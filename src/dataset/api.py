# -*- coding: utf-8 -*-


import xml.etree.ElementTree as ET

import cv2
import numpy as np
from const import DEFAULT_WIDTH, IMAGE_SIZE
from rendering import render_entity


class Bunch:
    """Dummy class"""
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def get_real_bbox(entity_bbox, entity_type, entity_size, entity_angle):
    assert entity_type != "none"
    center = (int(entity_bbox[1] * IMAGE_SIZE), int(entity_bbox[0] * IMAGE_SIZE))
    M = cv2.getRotationMatrix2D(center, entity_angle, 1)
    unit = min(entity_bbox[2], entity_bbox[3]) * IMAGE_SIZE / 2
    delta = DEFAULT_WIDTH * 1.5 / IMAGE_SIZE
    if entity_type == "circle":
        radius = unit * entity_size
        real_bbox = [center[1] * 1.0 / IMAGE_SIZE, center[0] * 1.0 / IMAGE_SIZE, 2 * radius / IMAGE_SIZE + delta, 2 * radius / IMAGE_SIZE + delta]
    else:
        if entity_type == "triangle":
            dl = int(unit * entity_size)
            homo_pts = np.array([[center[0], center[1] - dl, 1], 
                                 [center[0] + int(dl / 2.0 * np.sqrt(3)), center[1] + int(dl / 2.0), 1], 
                                 [center[0] - int(dl / 2.0 * np.sqrt(3)), center[1] + int(dl / 2.0), 1]], 
                                np.int32)
        if entity_type == "square":
            dl = int(unit / 2 * np.sqrt(2) * entity_size)
            homo_pts = np.array([[center[0] - dl, center[1] - dl, 1],
                                 [center[0] - dl, center[1] + dl, 1], 
                                 [center[0] + dl, center[1] + dl, 1], 
                                 [center[0] + dl, center[1] - dl, 1]],
                                np.int32)
        if entity_type == "pentagon":
            dl = int(unit * entity_size)
            homo_pts = np.array([[center[0], center[1] - dl, 1],
                                 [center[0] - int(dl * np.cos(np.pi / 10)), center[1] - int(dl * np.sin(np.pi / 10)), 1],
                                 [center[0] - int(dl * np.sin(np.pi / 5)), center[1] + int(dl * np.cos(np.pi / 5)), 1],
                                 [center[0] + int(dl * np.sin(np.pi / 5)), center[1] + int(dl * np.cos(np.pi / 5)), 1],
                                 [center[0] + int(dl * np.cos(np.pi / 10)), center[1] - int(dl * np.sin(np.pi / 10)), 1]],
                                np.int32)
        if entity_type == "hexagon":
            dl = int(unit * entity_size)
            homo_pts = np.array([[center[0], center[1] - dl, 1],
                                 [center[0] - int(dl / 2.0 * np.sqrt(3)), center[1] - int(dl / 2.0), 1],
                                 [center[0] - int(dl / 2.0 * np.sqrt(3)), center[1] + int(dl / 2.0), 1],
                                 [center[0], center[1] + dl, 1],
                                 [center[0] + int(dl / 2.0 * np.sqrt(3)), center[1] + int(dl / 2.0), 1],
                                 [center[0] + int(dl / 2.0 * np.sqrt(3)), center[1] - int(dl / 2.0), 1]],
                                np.int32)
        after_pts = np.dot(M, homo_pts.T)
        min_x = min(after_pts[1, :]) / IMAGE_SIZE
        max_x = max(after_pts[1, :]) / IMAGE_SIZE
        min_y = min(after_pts[0, :]) / IMAGE_SIZE
        max_y = max(after_pts[0, :]) / IMAGE_SIZE
        real_bbox = [(min_x + max_x) / 2, (min_y + max_y) / 2, max_x - min_x + delta, max_y - min_y + delta] 
    return list(np.round(real_bbox, 4))


def get_mask(entity_bbox, entity_type, entity_size, entity_angle):
    dummy_entity = Bunch()
    dummy_entity.bbox = entity_bbox
    dummy_entity.type = Bunch(get_value=lambda : entity_type)
    dummy_entity.size = Bunch(get_value=lambda : entity_size)
    dummy_entity.color = Bunch(get_value=lambda : 0)
    dummy_entity.angle = Bunch(get_value=lambda : entity_angle)
    mask = render_entity(dummy_entity) / 255
    return mask


# ref: https://www.kaggle.com/stainsby/fast-tested-rle
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return "[" + ",".join(str(x) for x in runs) + "]"
 

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle[1:-1].split(",")
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
