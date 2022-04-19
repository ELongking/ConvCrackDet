import cv2
import matplotlib.pyplot as plt


def cv_show(img, title=None, save=False):
    """
    Function for image-show during debugging

    :param img: input image
    :param title: window name
    :param save: bool type, if True, save the showing image
    :return: None

    """

    if save:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imsave(title + '.png', img, dpi=600)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
