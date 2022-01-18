#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np
from numpy import random
from collections import deque
import cvzone



pts = {}

__all__ = ["vis"]

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)



def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def vis2(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        
        UI_box(box, img, color=compute_color_for_labels(cls_id),label=class_names[cls_id],line_thickness=2)
        

    return img

def vid_to_frames(path):
    frames = []
    cap = cv2.VideoCapture(path)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frames.append(img)
    return frames

emojidict = dict(
    happy = vid_to_frames('assets/happy.png'),
    sad = vid_to_frames('assets/sad.png'),
    surprised = vid_to_frames('assets/surprised.png')       
    )
current = None
count = 0

def add_image(img, src2, x, y, ):
    # x=  x+90
    # y = y-10
    w = 80
    h = 80

    initial = img[y:y+h,x:x+w]
    src1 = initial
    src2 = cv2.resize(src2, src1.shape[1::-1])
    u_green = np.array([1, 1, 1])
    l_green = np.array([0, 0, 0])
    mask = cv2.inRange(src2, l_green, u_green)
    res = cv2.bitwise_and(src2, src2, mask = mask)
    f = src2 - res
    f = np.where(f == 0, src1, f)

    # src2[np.where((src2 ==[0,0,0]).all(axis=2))] = [255,255,255]
    # dst = cv2.bitwise_and(src1, src2)
    # dst = cv2.addWeighted(src1, 0.2, src2, 0.8, 0)
    img[y:y+h,x:x+w] = f
    return img

def vis10(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    global current
    global count

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        label = class_names[cls_id]
        UI_box2(box, img, color=compute_color_for_labels(cls_id),label=label,line_thickness=2)
        if count == len(emojidict[label]):
            count = 0 
        if label == current:
            count +=1
        elif label != current:
            count = 0 
        width = int(box[0]) - int(box[0])
        height = int(box[1]) - int(box[1])
        try:
            img = add_image(img, emojidict[label][count] ,int(box[0] + width/2), int(box[1]) )
        except:
            img = add_image(img, emojidict[label][0] ,int(box[0]), int(box[1]) )

        current = label
        
    return img



def vis3(img, boxes, scores, cls_ids, conf=0.5, class_names=None, reader = None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        
        if reader:
            a,b,c,d = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            im = img[b:d, a:c]
            gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            result = reader.readtext(gray)
            text = ""

            if len(result) >=1:
                for res in result:
                    if len(res[1]) > 5 and res[2] > 0.2:
                        text = res[1]

        UI_box(box, img, color=compute_color_for_labels(cls_id),label=class_names[cls_id] + text ,line_thickness=2)
        
    return img



_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def vis_track(img, boxes):

    for key in list(pts):
      if key not in boxes[:, -2]:
        pts.pop(key)
    
    for i in range(len(boxes)):
        box = boxes[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        id = box[4]
        if id not in pts:
          pts[id] = deque(maxlen= 64)

        clsid = box[5]

        color = compute_color_for_labels(clsid)
        text = '%d'%(id)
        txt_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #bbox_center_point(x,y)
        center = (int(((box[0])+(box[2]))/2),int(((box[1])+(box[3]))/2))

        pts[id].append(center)

        thickness = 4
        cv2.circle(img,  (center), 1, color, thickness)

    #draw motion path
        for j in range(1, len(pts[id])):
            if pts[id][j - 1] is None or pts[id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + 1)) * 3)
            cv2.line(img,(pts[id][j-1]), (pts[id][j]),(color),4)

    return img


def vis_track8(img, boxes):

    for key in list(pts):
      if key not in boxes[:, -2]:
        pts.pop(key)
    
    for i in range(len(boxes)):
        box = boxes[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        id = box[4]
        if id not in pts:
          pts[id] = deque(maxlen= 64)
        clsid = 1
        color = compute_color_for_labels(clsid)
        text = '%d'%(id)
        txt_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #bbox_center_point(x,y)
        center = (int(((box[0])+(box[2]))/2),int(((box[1])+(box[3]))/2))
        # print(center)
        pts[id].append(center)
        thickness = 5
        cv2.circle(img,  (center), 1, color, thickness)
        for j in range(1, len(pts[id])):
            if pts[id][j - 1] is None or pts[id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + 1)) * 2)
            cv2.line(img,(pts[id][j-1]), (pts[id][j]),(color),5)

    return img

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person  #BGR
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def UI_box(x, img, color=None,label=None,line_thickness=None, boundingbox = True):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    if boundingbox:
        cv2.rectangle(img, c1, c2, color, 2)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)




def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box2(x, img, color=None,label=None,line_thickness=None, boundingbox = True):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.30 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    if boundingbox:
        # cv2.rectangle(img, c1, c2, color, 2)
        draw_disconnected_rect(img, c1, c2, color, tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, tf, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



def draw_disconnected_rect( img, pt1, pt2, color, thickness):
    width = pt2[0] - pt1[0]
    height = pt2[1] - pt1[1]
    line_width = min(30, width // 3)
    line_height = min(30, height // 3)
    line_length = max(line_width, line_height)
    cv2.line(img, pt1, (pt1[0] + line_length, pt1[1]), color, thickness)
    cv2.line(img, pt1, (pt1[0], pt1[1] + line_length), color, thickness)
    cv2.line(
        img, (pt2[0] - line_length, pt1[1]), (pt2[0], pt1[1]), color, thickness
    )
    cv2.line(
        img, (pt2[0], pt1[1]), (pt2[0], pt1[1] + line_length), color, thickness
    )
    cv2.line(
        img, (pt1[0], pt2[1]), (pt1[0] + line_length, pt2[1]), color, thickness
    )
    cv2.line(
        img, (pt1[0], pt2[1] - line_length), (pt1[0], pt2[1]), color, thickness
    )
    cv2.line(img, pt2, (pt2[0] - line_length, pt2[1]), color, thickness)
    cv2.line(img, (pt2[0], pt2[1] - line_length), pt2, color, thickness)