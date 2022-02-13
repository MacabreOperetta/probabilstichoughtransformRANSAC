from skimage import feature, color, transform, io
from skimage.transform import rescale, resize, downscale_local_mean
import cv2 as cv
import numpy as np

# load image and get dimensions
def normalize(image_name):
    img = cv.imread(image_name, cv.IMREAD_UNCHANGED)
    alpha_channel = img[:, :, 3]
    _, mask = cv.threshold(alpha_channel, 254, 255, cv.THRESH_BINARY)  # binarize mask
    color = img[:, :, :3]
    new_img = cv.bitwise_not(cv.bitwise_not(color, mask=mask))
    filename = image_name.split(chr(92))
    filename ="Generated"+chr(92)+"Normalized"+chr(92)+filename[2]
    filename = filename.replace("_rectified","")
    save_name = '.'.join(filename.split('.')[:-1]) + '_normalized.png'
    image_resized = cv.resize(new_img,(2500, 1500))
    io.imsave(save_name, image_resized)