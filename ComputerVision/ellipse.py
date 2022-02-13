import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
from operator import itemgetter
import argparse


def img_to_binary_grey_scale(img, threshold):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # convert to grayscale
    ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    return thresh

def extract_notes(binary_image,image_name):
    #cv2.imshow('originale', binary_image)
    image_to_erode = cv2.bitwise_not(binary_image)
    output = binary_image.copy()
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(image_to_erode, kernel, iterations=2)
    dilatation = cv2.dilate(erosion, kernel, iterations=3)
    extracted_notes_image = cv2.bitwise_not(dilatation)
    # cv2.imshow('Morpho result', extracted_notes_image)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 80

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 460

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.6

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    # keypoints = detector.detect(binary_image)
    keypoints = detector.detect(extracted_notes_image)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(
        cv2.imread(image_name), keypoints,
        np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    filename = image_name.split(chr(92))
    filename ="Generated"+chr(92)+"Ellipse"+chr(92)+filename[2]
    filename = filename.replace("_normalized","")
    save_name = '.'.join(filename.split('.')[:-1]) + '_ellipse.png'
    cv2.imwrite(save_name, im_with_keypoints)
    #print(len(keypoints))
    notes_positions = [point.pt for point in keypoints]
    return notes_positions

def morpho_process(binary,image_name):
    notes_positions = extract_notes(binary,image_name)
    #print(notes_positions)
    return notes_positions


def ellipse_detection(image_name):
    img = cv2.imread(image_name)
    binary_morpho = img_to_binary_grey_scale(img, 127)

    note_positions = morpho_process(binary_morpho,image_name)
    return note_positions

