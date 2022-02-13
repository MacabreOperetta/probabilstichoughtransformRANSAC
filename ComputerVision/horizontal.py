import cv2 as cv
import numpy as np


# Load image, convert to grayscale, Otsu's threshold
def horizontal_lines(image_name):
    # Load the image
    src = cv.imread(image_name, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image: ' + image_name)
        return -1
    # Show source image
    # [load_image]
    # [gray]
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)

    # Detect horizontal lines
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
    detect_horizontal = cv.morphologyEx(bw, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv.findContours(detect_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv.drawContours(src, [c], -1, (36, 255, 12), 2)
    filename = image_name.split(chr(92))
    filename ="Generated"+chr(92)+"Horizontal"+chr(92)+filename[2]
    filename = filename.replace("_normalized","")
    save_name = '.'.join(filename.split('.')[:-1]) + '_horizontal.png'
    print(save_name)
    cv.imwrite(save_name, src)
    return cnts