import math
from numpy.ma import cos, sin
import random
import cv2
import numpy as np
from .show import cv_show


def width_calculate(img_original, contour):
    """
    Main function
    Find the maximum internal connection circle in each crack area
    Will print each crack width (also the max)

    :param img_original: original 3-channels image
    :param contour: the contours point array after finding contours process
    :return: the original image marked with the maximum internal connection circle

    """
    expansion_circle_list = []
    for c in contour:
        left_x = min(c[:, 0, 0])
        right_x = max(c[:, 0, 0])
        down_y = max(c[:, 0, 1])
        up_y = min(c[:, 0, 1])
        upper_r = min(right_x - left_x, down_y - up_y) / 2
        precision = math.sqrt((right_x - left_x) ** 2 + (down_y - up_y) ** 2) / (2 ** 13)
        Nx = 2 ** 8
        Ny = 2 ** 8
        pixel_X = np.linspace(left_x, right_x, Nx)
        pixel_Y = np.linspace(up_y, down_y, Ny)
        xx, yy = np.meshgrid(pixel_X, pixel_Y)
        in_list = []
        for i in range(pixel_X.shape[0]):
            for j in range(pixel_X.shape[0]):
                if cv2.pointPolygonTest(c, (xx[i][j], yy[i][j]), False) > 0:
                    in_list.append((xx[i][j], yy[i][j]))
        in_point = np.array(in_list)
        N = len(in_point)
        rand_index = random.sample(range(N), N // 100)
        rand_index.sort()
        radius = 0
        big_r = upper_r
        center = None
        for id in rand_index:
            tr = iterated_optimal_incircle_radius_get(c, in_point[id][0], in_point[id][1], radius, big_r, precision)
            if tr > radius:
                radius = tr
                center = (in_point[id][0], in_point[id][1])
        loops_index = [i for i in range(N) if i not in rand_index]
        for id in loops_index:
            tr = iterated_optimal_incircle_radius_get(c, in_point[id][0], in_point[id][1], radius, big_r, precision)
            if tr > radius:
                radius = tr
                center = (in_point[id][0], in_point[id][1])

        expansion_circle_list.append([radius, center])
        print('Crack width：', round(radius * 2, 2))

    print('---------------')
    expansion_circle_radius_list = [i[0] for i in expansion_circle_list]
    max_radius = max(expansion_circle_radius_list)
    max_center = expansion_circle_list[expansion_circle_radius_list.index(max_radius)][1]
    print('Maximum width：', round(max_radius * 2, 2))

    cv2.drawContours(img_original, contour, -1, (0, 0, 255), lineType=1)

    for expansion_circle in expansion_circle_list:
        radius_s = expansion_circle[0]
        center_s = expansion_circle[1]
        if radius_s == max_radius:
            cv2.circle(img_original, (int(max_center[0]), int(max_center[1])), int(max_radius), (255, 0, 0), 2)
        else:
            cv2.circle(img_original, (int(center_s[0]), int(center_s[1])), int(radius_s), (255, 245, 0), 2)

    return img_original, expansion_circle_list


def iterated_optimal_incircle_radius_get(contour, pixelx, pixely, small_r, big_r, precision):
    """
    Calculating the radius of the maximum internal connection circle

    :param contour: contour point array
    :param pixelx: x pixel coordinates of center
    :param pixely: y pixel coordinates of center
    :param small_r: the maximum radius of the previously found inner tangent circle
    :param big_r: limit of circle radius
    :param precision: the accuracy, using dichotomous method to find maximum radius
    :return: radius

    """
    radius = small_r
    L = np.linspace(0, 2 * math.pi, 360)
    circle_X = pixelx + radius * cos(L)
    circle_Y = pixely + radius * sin(L)
    for i in range(len(circle_Y)):
        if cv2.pointPolygonTest(contour, (circle_X[i], circle_Y[i]), False) < 0:
            return 0
    while big_r - small_r >= precision:
        half_r = (small_r + big_r) / 2
        circle_X = pixelx + half_r * cos(L)
        circle_Y = pixely + half_r * sin(L)
        if_out = False
        for i in range(len(circle_Y)):
            if cv2.pointPolygonTest(contour, (circle_X[i], circle_Y[i]), False) < 0:
                big_r = half_r
                if_out = True
        if not if_out:
            small_r = half_r
    radius = small_r
    return radius