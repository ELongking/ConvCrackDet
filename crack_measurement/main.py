import cv2

import numpy as np
from .utils import add_text
from .show import cv_show


def padding(img, pd=10):
    """
    For getting area-like contours, need to pad the original image to avoid crack at the edge of the window

    :param img: original image
    :param pd: padding distance
    :return: padded image

    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    minval, val, _, point = cv2.minMaxLoc(gray)
    padding_color = (int(img[point[1], point[0], 0]),
                     int(img[point[1], point[0], 1]),
                     int(img[point[1], point[0], 2]))
    img = cv2.copyMakeBorder(img, pd, pd, pd, pd, cv2.BORDER_CONSTANT, value=padding_color)
    return img, minval


def contrast_enhance(img):
    """
    Enhance image's contrast

    :param img: input image, support for gray image
    :return: contrast-enhanced gray image

    """

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(img)
    return dst


def remove_small_area(contour, gamma=0.1):
    """
    Maintain the crack area in contour list, thus remove the noise contour

    :param contour: output of cv2.findContours()[0]
    :param gamma: compared to the crack area (usually the max), what range of area you also want to maintain
    :return: contour array after removing

    """
    area = []
    for i in range(len(contour)):
        area.append(cv2.contourArea(contour[i]))
    maxArea = max(area)
    return [contour[i] for i in range(len(area)) if area[i] > gamma * maxArea]


def show_contours(img, thre):
    """
    Main function, for showing contour on the original image
    Following process:
    original -> gray -> contrast-enhance -> blur -> binary threshold -> dilate & erosion -> canny edge detection ->
    original contours -> new contours -> result

    :param img: original image, support for 3-channels image
    :return: contours array for calculating, and result image based on the original adding contours line

    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    _, thresh = cv2.threshold(gray, 1.01 * thre, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ero = cv2.erode(thresh, np.ones((3, 3), int), iterations=1)
    gray = cv2.dilate(ero, np.ones((3, 3), int), iterations=1)

    edge = cv2.Canny(gray, 1.01 * thre, 255, apertureSize=3, L2gradient=True)

    contours, hierarchy = cv2.findContours(edge,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    new_contours = remove_small_area(contours, gamma=0.2)
    for i in range(len(new_contours)):
        color = np.random.randint(0, 255, size=(1, 3), dtype=int)
        conImg = cv2.drawContours(img, new_contours, i, (int(color[0, 0]), int(color[0, 1]), int(color[0, 2])),
                                  lineType=1)
    return new_contours, conImg


def process(image):
    image, minVal = padding(image)
    SHAPE = image.shape
    image_copy = image.copy()
    cnt, cntImg = show_contours(image, minVal)
    cv_show(cntImg, 'edge detection')
    add_text(image_copy, cnt, SHAPE)
    return image_copy
