import cv2 as cv
import numpy as np
import os, glob

in_folder_path = "testare/"
out_folder_path = "fisiere_solutie/351_Oanea_Tudor_Cosmin/"

column = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
width = 1500
height = 1500
patch_width = 100
patch_height = 100
template_features = []
orb = cv.ORB_create()
sdl = []
stl = []
sdc = []
stc = []
table = np.matrix([])
value = {}


def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, image)

# takes original image
# returns 4 points corresponding to game table corners
def extract_corners(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_m_blur = cv.medianBlur(image, 3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5)
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)

    _, thresh = cv.threshold(image_sharpened, 30, 255, cv.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.dilate(thresh, kernel)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if 3000 < cv.contourArea(c)]
    points = [cv.boundingRect(mask_contour) for mask_contour in contours]

    points.sort(key=lambda x: x[0])

    lateral_search = 4
    top_left = (points[0][0], points[0][1])
    for k in range(1, lateral_search):
        if points[k][1] < top_left[1]:
            top_left = (points[k][0], points[k][1])

    bottom_left = (points[0][0], points[0][1] + points[0][3])
    for k in range(1, lateral_search):
        if points[k][1] + points[k][3] > bottom_left[1]:
            bottom_left = (points[k][0], points[k][1] + points[k][3])

    top_right = (points[-1][0] + points[-1][2], points[-1][1])
    for k in range(1, lateral_search):
        if points[-1 - k][1] < top_right[1]:
            top_right = (points[-1 - k][0] + points[-1 - k][2], points[-1 - k][1])

    bottom_right = (points[-1][0] + points[-1][2], points[-1][1] + points[-1][3])
    for k in range(1, lateral_search):
        if points[-1 - k][1] + points[-1 - k][3] > bottom_right[1]:
            bottom_right = (points[-1 - k][0] + points[-1 - k][2], points[-1 - k][1] + points[-1 - k][3])

    return top_left, top_right, bottom_right, bottom_left

# takes original image and game table corners
# returns wraped perspective of gray-converted image
def wrap_conversion(image, top_left, top_right, bottom_right, bottom_left):
    global width, height
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)

    result = cv.warpPerspective(gray_img, M, (width, height))

    return result

# takes a gray image and the position of top-left corner of one patch
# returns if that patch contains a white piece and the patch itself
def check_white_piece(gray_image, i, j):
    global patch_width, patch_height
    _, thresh = cv.threshold(gray_image, 170, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh_img = cv.dilate(thresh, kernel)

    point1 = (i * patch_height, j * patch_width)
    point2 = ((i + 1) * patch_height, (j + 1) * patch_width)
    patch_gray = gray_image[point1[0]: point2[0], point1[1]: point2[1]]
    patch_thresh = thresh_img[point1[0]: point2[0], point1[1]: point2[1]]

    kernel_patch = np.ones((5, 5), np.uint8)
    patch_thresh = cv.erode(patch_thresh, kernel_patch)

    patch_mean = np.mean(patch_thresh)

    return patch_mean > 60, patch_gray

# computes features key_points and descriptors for known letter patches
# known letter patch = jpg with a patch of a letter named according to that letter and the picture it was cropped from
# (known letter patches are cropped from given train images)
def compute_template_features():
    global template_features, orb
    folder_path = "known letter patches gray"
    for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
        letter_image = cv.imread(filename)
        key_points, descriptor = orb.detectAndCompute(letter_image, None)
        if filename[-10] == "Q":
            template_features.append((key_points, descriptor, "?", filename))
        else:
            template_features.append((key_points, descriptor, filename[-10], filename))

# takes a letter patch and computes it's features key-points and descriptors
# returns letter of the most similar known letter patch
def letter_classification(letter_patch):
    global template_features

    # Brute Force Matching
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    key_points, descriptor = orb.detectAndCompute(letter_patch, None)
    distmin = np.inf

    letter = "JOKER"
    for temp in template_features:
        matches = bf.match(descriptor, temp[1])
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches):
            dist = np.mean([m.distance for m in matches[:10]])
            if dist < distmin:
                distmin = dist
                letter = temp[2]

    return letter

# initialization of game table and it's special positions
def init_table():
    global table, sdl, stl, sdc, stc
    reset = []
    for i in range(15):
        reset.append(["0" for j in range(15)])
    table = np.matrix(reset)

    sdl = [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14), (6, 2), (6, 6), (6, 8), (6, 12),
           (7, 3), (7, 11), (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14), (12, 6),
           (12, 8), (14, 3), (14, 11)]
    stl = [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13), (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)]
    sdc = [(1, 1), (1, 13), (2, 2), (2, 12), (3, 3), (3, 11), (4, 4), (4, 10), (7, 7),
           (10, 4), (10, 10), (11, 3), (11, 11), (12, 2), (12, 12), (13, 1), (13, 13)]
    stc = [(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)]

# initialization of letter values
def init_value():
    global value

    value["A"] = 1
    value["B"] = 9
    value["C"] = 1
    value["D"] = 2
    value["E"] = 1
    value["F"] = 8
    value["G"] = 9
    value["H"] = 10
    value["I"] = 1
    value["J"] = 10
    value["L"] = 1
    value["M"] = 4
    value["N"] = 1
    value["O"] = 1
    value["P"] = 2
    value["R"] = 1
    value["S"] = 1
    value["T"] = 1
    value["U"] = 1
    value["V"] = 8
    value["X"] = 10
    value["Z"] = 10
    value["?"] = 0

# takes position of a letter
# returns the position type
def check_special_position(i, j):
    if (i, j) in sdl:
        sdl.remove((i, j))
        return "sdl"
    elif (i, j) in stl:
        stl.remove((i, j))
        return "stl"
    elif (i, j) in sdc:
        sdc.remove((i, j))
        return "sdc"
    elif (i, j) in stc:
        stc.remove((i, j))
        return "stc"
    return "0"

# computes the score for a list of new pieces on the table
def compute_score(new_pieces):
    global value, table
    score = 0
    posj = new_pieces[0][1]
    posi = new_pieces[0][0]
    special = {}
    special[(posi, posj)] = check_special_position(posi, posj)

    # only one new piece
    if len(new_pieces) == 1:
        # search horizontally and vertically for new words
        found = False
        word_score = value[ table[posi, posj] ]
        if special[(posi, posj)] == "sdl":
            word_score *= 2
        elif special[(posi, posj)] == "stl":
            word_score *= 3

        # search right
        i = posi
        right = posj + 1
        while right <= 14 and table[i, right] != "0":
            letter = table[i, right]
            word_score += value[letter]
            right += 1
            found = True

        # search left
        i = posi
        left = posj - 1
        while left >= 0 and table[i, left] != "0":
            letter = table[i, left]
            word_score += value[letter]
            left -= 1
            found = True

        if special[(posi, posj)] == "sdc":
            word_score = word_score * 2
        elif special[(posi, posj)] == "stc":
            word_score = word_score * 3

        if found:
            score += word_score

        found = False
        word_score = value[ table[posi, posj]]
        if special[(posi, posj)] == "sdl":
            word_score *= 2
        elif special[(posi, posj)] == "stl":
            word_score *= 3
        # search down
        down = posi + 1
        j = posj
        while down <= 14 and table[down, j] != "0":
            letter = table[down, j]
            word_score += value[letter]
            down += 1
            found = True

        # search up
        up = posi - 1
        while up >= 0 and table[up, j] != "0":
            letter = table[up, j]
            word_score += value[letter]
            up -= 1
            found = True

        if special[(posi, posj)] == "sdc":
            word_score = word_score * 2
        elif special[(posi, posj)] == "stc":
            word_score = word_score * 3

        if found:
            score += word_score

    # word is written horizontally
    elif new_pieces[0][0] == new_pieces[1][0]:
        word_score = value[ table[posi, posj] ]
        if special[(posi, posj)] == "sdl":
            word_score *= 2
        elif special[(posi, posj)] == "stl":
            word_score *= 3

        i = posi
        # search right
        right = posj + 1
        while right <= 14 and table[i, right] != "0":
            letter = table[i, right]
            if (i, right) in new_pieces:
                if (i, right) not in special.keys():
                    special[(i, right)] = check_special_position(i, right)

                if special[(i, right)] == "sdl":
                    word_score += 2 * value[letter]
                elif special[(i, right)] == "stl":
                    word_score += 3 * value[letter]
                else:
                    word_score += value[letter]
            else:
                word_score += value[letter]
            right += 1
        right -= 1

        # search left
        left = posj - 1
        while left >= 0 and table[i, left] != "0":
            letter = table[i, left]
            if (i, left) in new_pieces:
                if (i, left) not in special.keys():
                    special[(i, left)] = check_special_position(i, left)

                if special[(i, left)] == "sdl":
                    word_score += 2 * value[letter]
                elif special[(i, left)] == "stl":
                    word_score += 3 * value[letter]
                else:
                    word_score += value[letter]
            else:
                word_score += value[letter]
            left -= 1
        left += 1

        for sp in special.values():
            if sp == "sdc":
                word_score = word_score * 2
            elif sp == "stc":
                word_score = word_score * 3

        score = word_score

        # for each new letter in between left and right search for vertically written words
        for j in range(left, right + 1):
            if (i, j) in new_pieces:
                # initialize vertical word score with current piece value
                found = False
                word_score = value[ table[i, j]]
                if special[(i, j)] == "sdl":
                    word_score *= 2
                elif special[(i, j)] == "stl":
                    word_score *= 3

                # search up and add normal values to word_score
                up = i - 1
                while up >= 0 and table[up, j] != "0":
                    letter = table[up, j]
                    word_score += value[letter]
                    up -= 1
                    found = True

                # search down and add normal values to word_score
                down = i + 1
                while down <= 14 and table[down, j] != "0":
                    letter = table[down, j]
                    word_score += value[letter]
                    down += 1
                    found = True

                if special[(i, j)] == "sdc":
                    word_score = word_score * 2
                elif special[(i, j)] == "stc":
                    word_score = word_score * 3

                if found:
                    score += word_score

    # word is written vertically
    else:
        word_score = value[ table[posi, posj]]
        if special[(posi, posj)] == "sdl":
            word_score *= 2
        elif special[(posi, posj)] == "stl":
            word_score *= 3

        j = posj
        # search down
        down = posi + 1
        while down <= 14 and table[down, j] != "0":
            letter = table[down, j]
            if (down, j) in new_pieces:
                if (down, j) not in special.keys():
                    special[(down, j)] = check_special_position(down, j)
                if special[(down, j)] == "sdl":
                    word_score += 2 * value[letter]
                elif special[(down, j)] == "stl":
                    word_score += 3 * value[letter]
                else:
                    word_score += value[letter]
            else:
                word_score += value[letter]
            down += 1
        down -= 1

        # search up
        up = posi - 1
        while up >= 0 and table[up, j] != "0":
            letter = table[up, j]
            if (up, j) in new_pieces:
                if (up, j) not in special.keys():
                    special[(up, j)] = check_special_position(up, j)
                if special[(up, j)] == "sdl":
                    word_score += 2 * value[letter]
                elif special[(up, j)] == "stl":
                    word_score += 3 * value[letter]
                else:
                    word_score += value[letter]
            else:
                word_score += value[letter]
            up -= 1
        up += 1

        for sp in special.values():
            if sp == "sdc":
                word_score = word_score * 2
            elif sp == "stc":
                word_score = word_score * 3

        score = word_score

        # for each new letter in between up and down search for horizontally written words
        for i in range(up, down + 1):
            if (i, j) in new_pieces:
                # initialize vertical word score with current piece value
                found = False
                word_score = value[ table[i, j]]
                if special[(i, j)] == "sdl":
                    word_score *= 2
                elif special[(i, j)] == "stl":
                    word_score *= 3

                # search left and add normal values to word_score
                left = j - 1
                while left >= 0 and table[i, left] != "0":
                    letter = table[i, left]
                    word_score += value[letter]
                    left -= 1
                    found = True

                # search right and add normal values to word_score
                right = j + 1
                while right <= 14 and table[i, right] != "0":
                    letter = table[i, right]
                    word_score += value[letter]
                    right += 1
                    found = True

                if special[(i, j)] == "sdc":
                    word_score = word_score * 2
                elif special[(i, j)] == "stc":
                    word_score = word_score * 3

                if found:
                    score += word_score

    if len(new_pieces) == 7:
        score += 50
    return score

def write_file(nume, final_file):
    text = ""
    for poz in final_file:
        text = text + poz + "\n"

    with open(out_folder_path + nume + ".txt", "w") as f:
        f.write(text)

def run():
    for pic_i in range(1, 6):
        white_pieces = []
        init_table()
        top_left, top_right, bottom_right, bottom_left = 0, 0, 0, 0

        for pic_j in range(1, 21):
            final_file = []

            if pic_j < 10:
                pic_name = str(pic_i) + "_0" + str(pic_j)
            else:
                pic_name = str(pic_i) + "_" + str(pic_j)

            image = cv.imread(in_folder_path + pic_name + ".jpg")
            crop_image = image[1200 : image.shape[0] - 700, 625 : image.shape[1] - 450]

            if pic_j == 1:
                top_left, top_right, bottom_right, bottom_left = extract_corners(crop_image)
            rez = wrap_conversion(crop_image, top_left, top_right, bottom_right, bottom_left)

            new_pieces = []

            for i in range(15):
                for j in range(15):
                    is_letter_piece, patch_gray = check_white_piece(rez, i, j)
                    if is_letter_piece and (i, j) not in white_pieces:
                        white_pieces.append((i, j))

                        letter = letter_classification(patch_gray)
                        if letter == "Q":
                            letter = "?"
                        final_file.append((str(i + 1) + column[j] + " " + letter))

                        table[i, j] = letter
                        new_pieces.append((i, j))

                        # print(final_file[-1])

            score = compute_score(new_pieces)
            final_file.append(str(score))
            write_file(pic_name, final_file)

            print(pic_name, "done\n")

init_value()
compute_template_features()
run()