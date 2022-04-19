import cv2
from skimage.morphology import skeletonize
import numpy as np
from skan import Skeleton, summarize
from .show import cv_show
import networkx as nx


def extract_skeleton(contour, shape=None):
    """
    Create a template image to get contour area's skeleton

    :param contour: contours from cv2.findContours function
    :param shape: image shape
    :return: skeleton image list

    """
    skeList = []
    for c in contour:
        template = np.zeros(shape, np.uint8)
        temp_img = cv2.drawContours(template, [c], -1, (255, 255, 255), thickness=-1)
        skeleton = skeletonize(temp_img)
        skeList.append(skeleton)
    return skeList


def length_calculate(contour, shape=None, mode='sum'):
    """
    Main function
    calculate the length from skeleton image

    :param contour: contours from cv2.findContours function
    :param shape: image shape
    :param mode: two ways of length calculating: 'sum' means sum the all skeletons' length directly; and 'max' means
    get the longest skeleton without branch
    :return: result length list

    """
    len_list = []
    skeleton = extract_skeleton(contour, shape=shape)
    for s in skeleton:
        ske_df = summarize(Skeleton(s))
        if mode == 'sum':
            len_list.append(round(sum(ske_df['branch-distance'].values.tolist()), 2))
        elif mode == 'max':
            graph = nx.DiGraph()
            for i in range(ske_df.shape[0]):
                graph.add_edge(str(ske_df.at[i, 'node-id-src']), str(ske_df.at[i, 'node-id-dst']),
                               weight=ske_df.at[i, 'branch-distance'])
            L = nx.dag_longest_path_length(graph)
            len_list.append(round(L, 2))
    return len_list
