import cv2
from .length import length_calculate
from .width import width_calculate
from .show import cv_show


def add_text(img, contour, shape=None, ratio=1):
    """
    Add text mark to the image

    :param img: original image
    :param contour: contours from cv2.findContours function
    :param shape: image shape
    :param ratio: similarity ratio
    :return: result image with mark

    """
    W = width_calculate(img, contour)[1]
    L = length_calculate(contour, shape=shape, mode='max')
    for w, l in zip(W, L):
        text = 'L={}\nW={}'.format(l * ratio, round(w[0] * 2 * ratio, 2))
        x, y = int(w[1][0]) - 5, int(w[1][1])
        dy = 10
        for i, txt in enumerate(text.split('\n')):
            cv2.putText(img,
                        txt,
                        (x, y + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.25,
                        (255, 0, 255),
                        1)
        print(f'width={w[0]}, inner joint circle coordinate={w[1]}, length={l}')
