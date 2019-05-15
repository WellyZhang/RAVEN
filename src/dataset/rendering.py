# -*- coding: utf-8 -*-


import cv2
import numpy as np
from PIL import Image

from AoT import Root
from const import CENTER, DEFAULT_WIDTH, IMAGE_SIZE


def imshow(array):
    image = Image.fromarray(array)
    image.show()


def imsave(array, filepath):
    image = Image.fromarray(array)
    image.save(filepath)


def generate_matrix(array_list):
    # row-major array_list
    assert len(array_list) <= 9
    img_grid = np.zeros((IMAGE_SIZE * 3, IMAGE_SIZE * 3), np.uint8)
    for idx in range(len(array_list)):
        i, j = divmod(idx, 3)
        img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE, j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = array_list[idx]
    # draw grid
    for x in [0.33, 0.67]:
        img_grid[int(x * IMAGE_SIZE * 3) - 1:int(x * IMAGE_SIZE * 3) + 1, :] = 0
    for y in [0.33, 0.67]:
        img_grid[:, int(y * IMAGE_SIZE * 3) - 1:int(y * IMAGE_SIZE * 3) + 1] = 0
    return img_grid


def generate_answers(array_list):
    assert len(array_list) <= 8
    img_grid = np.zeros((IMAGE_SIZE * 2, IMAGE_SIZE * 4), np.uint8)
    for idx in range(len(array_list)):
        i, j = divmod(idx, 4)
        img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE, j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = array_list[idx]
    # draw grid
    for x in [0.5]:
        img_grid[int(x * IMAGE_SIZE * 2) - 1:int(x * IMAGE_SIZE * 2) + 1, :] = 0
    for y in [0.25, 0.5, 0.75]:
        img_grid[:, int(y * IMAGE_SIZE * 4) - 1:int(y * IMAGE_SIZE * 4) + 1] = 0
    return img_grid


def generate_matrix_answer(array_list):
    # row-major array_list
    assert len(array_list) <= 18
    img_grid = np.zeros((IMAGE_SIZE * 6, IMAGE_SIZE * 3), np.uint8)
    for idx in range(len(array_list)):
        i, j = divmod(idx, 3)
        img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE, j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = array_list[idx]
    # draw grid
    for x in [0.33, 0.67, 1.00, 1.33, 1.67]:
        img_grid[int(x * IMAGE_SIZE * 3), :] = 0
    for y in [0.33, 0.67]:
        img_grid[:, int(y * IMAGE_SIZE * 3)] = 0
    return img_grid


def merge_matrix_answer(matrix, answer):
    matrix_image = generate_matrix(matrix)
    answer_image = generate_answers(answer)
    img_grid = np.ones((IMAGE_SIZE * 5 + 20, IMAGE_SIZE * 4), np.uint8) * 255
    img_grid[:IMAGE_SIZE * 3, int(0.5 * IMAGE_SIZE):int(3.5 * IMAGE_SIZE)] = matrix_image
    img_grid[-(IMAGE_SIZE * 2):, :] = answer_image
    return img_grid

def render_panel(root):
    # Decompose the panel into a structure and its entities
    assert isinstance(root, Root)
    canvas = np.ones((IMAGE_SIZE, IMAGE_SIZE), np.uint8) * 255
    structure, entities = root.prepare()
    structure_img = render_structure(structure)
    background = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    # note left components entities are in the lower layer
    for entity in entities:
        entity_img = render_entity(entity)
        background = layer_add(background, entity_img)
    background = layer_add(background, structure_img)
    return canvas - background


def render_structure(structure_name):
    ret = None
    if structure_name == "Left_Right":
        ret = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        ret[:, int(0.5 * IMAGE_SIZE)] = 255.0
    elif structure_name == "Up_Down":
        ret = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        ret[int(0.5 * IMAGE_SIZE), :] = 255.0
    else:
        ret = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    return ret


def render_entity(entity):
    entity_bbox = entity.bbox
    entity_type = entity.type.get_value()
    entity_size = entity.size.get_value()
    entity_color = entity.color.get_value()
    entity_angle = entity.angle.get_value()
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)

    # planar position: [x, y, w, h]
    # angular position: [x, y, w, h, x_c, y_c, omega]
    # center: (columns, rows)
    center = (int(entity_bbox[1] * IMAGE_SIZE), int(entity_bbox[0] * IMAGE_SIZE))
    if entity_type == "triangle":
        unit = min(entity_bbox[2], entity_bbox[3]) * IMAGE_SIZE / 2
        dl = int(unit * entity_size)
        pts = np.array([[center[0], center[1] - dl], 
                        [center[0] + int(dl / 2.0 * np.sqrt(3)), center[1] + int(dl / 2.0)], 
                        [center[0] - int(dl / 2.0 * np.sqrt(3)), center[1] + int(dl / 2.0)]], 
                       np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = 255 - entity_color
        width = DEFAULT_WIDTH
        draw_triangle(img, pts, color, width)
    elif entity_type == "square":
        unit = min(entity_bbox[2], entity_bbox[3]) * IMAGE_SIZE / 2
        dl = int(unit / 2 * np.sqrt(2) * entity_size)
        pt1 = (center[0] - dl, center[1] - dl)
        pt2 = (center[0] + dl, center[1] + dl)
        color = 255 - entity_color
        width = DEFAULT_WIDTH
        draw_square(img, pt1, pt2, color, width)
    elif entity_type == "pentagon":
        unit = min(entity_bbox[2], entity_bbox[3]) * IMAGE_SIZE / 2
        dl = int(unit * entity_size)
        pts = np.array([[center[0], center[1] - dl],
                        [center[0] - int(dl * np.cos(np.pi / 10)), center[1] - int(dl * np.sin(np.pi / 10))],
                        [center[0] - int(dl * np.sin(np.pi / 5)), center[1] + int(dl * np.cos(np.pi / 5))],
                        [center[0] + int(dl * np.sin(np.pi / 5)), center[1] + int(dl * np.cos(np.pi / 5))],
                        [center[0] + int(dl * np.cos(np.pi / 10)), center[1] - int(dl * np.sin(np.pi / 10))]],
                       np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = 255 - entity_color
        width = DEFAULT_WIDTH
        draw_pentagon(img, pts, color, width)
    elif entity_type == "hexagon":
        unit = min(entity_bbox[2], entity_bbox[3]) * IMAGE_SIZE / 2
        dl = int(unit * entity_size)
        pts = np.array([[center[0], center[1] - dl],
                        [center[0] - int(dl / 2.0 * np.sqrt(3)), center[1] - int(dl / 2.0)],
                        [center[0] - int(dl / 2.0 * np.sqrt(3)), center[1] + int(dl / 2.0)],
                        [center[0], center[1] + dl],
                        [center[0] + int(dl / 2.0 * np.sqrt(3)), center[1] + int(dl / 2.0)],
                        [center[0] + int(dl / 2.0 * np.sqrt(3)), center[1] - int(dl / 2.0)]],
                       np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = 255 - entity_color
        width = DEFAULT_WIDTH
        draw_hexagon(img, pts, color, width)
    elif entity_type == "circle":
        # Minus because of the way we show the image. See: render_panel's return
        color = 255 - entity_color
        unit = min(entity_bbox[2], entity_bbox[3]) * IMAGE_SIZE / 2
        radius = int(unit * entity_size)
        width = DEFAULT_WIDTH
        draw_circle(img, center, radius, color, width)
    elif entity_type == "none":
        pass
    # angular
    if len(entity_bbox) > 4:
        # [x, y, w, h, x_c, y_c, omega]
        entity_angle = entity_bbox[6]
        center = (int(entity_bbox[5] * IMAGE_SIZE), int(entity_bbox[4] * IMAGE_SIZE))
        img = rotate(img, entity_angle, center=center)
    # planar 
    else:
        img = rotate(img, entity_angle, center=center)
    # img = shift(img, *entity_position)

    return img


def shift(img, dx, dy):
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    img = cv2.warpAffine(img, M, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR)
    return img


def rotate(img, angle, center=CENTER):
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img = cv2.warpAffine(img, M, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR)
    return img


def scale(img, tx, ty, center=CENTER):
    M = np.array([[tx, 0, center[0] * (1 - tx)], [0, ty, center[1] * (1 - ty)]], np.float32)
    img = cv2.warpAffine(img, M, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR)
    return img


def layer_add(lower_layer_np, higher_layer_np):
    # higher_layer_np is superimposed on lower_layer_np
    # new_np = lower_layer_np.copy()
    # lower_layer_np is modified
    lower_layer_np[higher_layer_np > 0] = 0
    return lower_layer_np + higher_layer_np


# Draw primitives
def draw_triangle(img, pts, color, width):
    # if filled
    if color != 0:
        # fill the interior
        cv2.fillConvexPoly(img, pts, color)
        # draw the edge
        cv2.polylines(img, [pts], True, 255, width)
    # if not filled
    else:
        cv2.polylines(img, [pts], True, 255, width)


def draw_square(img, pt1, pt2, color, width):
    # if filled
    if color != 0:
        # fill the interior
        cv2.rectangle(img,
                      pt1,
                      pt2,
                      color, 
                      -1)
        # draw the edge
        cv2.rectangle(img, 
                      pt1,
                      pt2,
                      255,
                      width)
    # if not filled
    else:
        cv2.rectangle(img, 
                      pt1,
                      pt2,
                      255,
                      width)


def draw_pentagon(img, pts, color, width):
    # if filled
    if color != 0:
        # fill the interior
        cv2.fillConvexPoly(img, pts, color)
        # draw the edge
        cv2.polylines(img, [pts], True, 255, width)
    # if not filled
    else:
        cv2.polylines(img, [pts], True, 255, width)


def draw_hexagon(img, pts, color, width):
    # if filled
    if color != 0:
        # fill the interior
        cv2.fillConvexPoly(img, pts, color)
        # draw the edge
        cv2.polylines(img, [pts], True, 255, width)
    # if not filled
    else:
        cv2.polylines(img, [pts], True, 255, width)


def draw_circle(img, center, radius, color, width):
    # if filled
    if color != 0:
        # fill the interior
        cv2.circle(img,
                   center,
                   radius,
                   color,
                   -1)
        # draw the edge
        cv2.circle(img,
                   center,
                   radius,
                   255,
                   width)
    # if not filled
    else:
        cv2.circle(img,
                   center,
                   radius,
                   255,
                   width)
