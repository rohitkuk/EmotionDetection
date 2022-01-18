import sys
sys.path.insert(0, './YOLOX')
from yolox.data.datasets.coco_classes import COCO_CLASSES
from detector import Predictor
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
from yolox.utils import vis
from yolox.utils.visualize import vis_track, draw_border, UI_box, vis_track8, compute_color_for_labels
import time
from yolox.exp import get_exp
import numpy as np
from sort import Sort
from LineMappings import LM

try:
    LineMapping = LM[sys.argv[1].split('/')[-1]]
except:
    pass

import pandas as pd
import torch
import time
import datetime
import random 
from intersect_ import Point, doIntersect
import math

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
class_names = COCO_CLASSES

filter_classes = ['car', 'bus', 'truck']
TRAIL_LEN = 64

from collections import deque
from collections import Counter

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# speed_four_line_queue = {}

data_deque = {}


object_counter = {
    'North' : Counter(),
    'South' : Counter(),
    'East'  : Counter(),
    'West'  : Counter()
 }
MainDF = []

#  Do it for one video then save it in the config file , create a class initiate if mappings not present and start the canvas



names = COCO_CLASSES
# Functions from the Other files some may be redundant will clean it out once sure that things work expectedly/

def estimateSpeed(location1, location2):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	ppm = 8 #Pixels per Meter s
	d_meters = d_pixels / ppm
	time_constant = 15 * 3.6
	speed = d_meters * time_constant
	return int(speed)


def draw_boxes(img, bbox, object_id, identities=None, offset=(0, 0), LineMapping={}):


    height, width, _ = img.shape 
    
    # Clear mappings already in the deque , Check for without this
    [data_deque.pop(key) for key in set(data_deque) if key not in identities]
    data = []
    for i, box in enumerate(bbox):
        if class_names[object_id[i]] not in set(filter_classes):
            continue
        x1, y1, x2, y2 = [int(i) +offset[0]  for i in box]  
        box_height = (y2-y1)
        center = (int((x2+x1)/ 2), int((y2+y2)/2))
        id = int(identities[i]) if identities is not None else 0

        if id not in set(data_deque):  
          data_deque[id] = deque(maxlen= 100)
        #   speed_four_line_queue[id] = [] 

        color = compute_color_for_labels(object_id[i])
        obj_name = class_names[object_id[i]]
        label = '%s' % (obj_name)
        data_deque[id].appendleft(center) #appending left to speed up the check we will check the latest map
        UI_box(box, img, label=label, color=(0,255,0), line_thickness=3, boundingbox=True)
        if len(data_deque[id]) >= 2: # just to make sure weh have enough of tracking
            for dir_, line in LineMapping.items():
                if doIntersect(Point(*data_deque[id][0]), Point(*data_deque[id][1]), Point(*line[0]), Point(*line[1])):#
                    cv2.line(img, line[0], line[1], (255,255,255), 3)
                    object_counter[dir_].update([obj_name])
                    data.append({
                    'Category' : obj_name,
                    'direction': dir_,
                    'Time'     : datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'Speed'    : estimateSpeed(data_deque[id][1], data_deque[id][0]),
                    # 'Speed'    : random.randint(30,60),
                    'id'       : id
                    })    
    return img,data


class Tracker():
    def __init__(self, filter_class=None, model='yolox-s', ckpt='wieghts/yolox_s.pth', LineMapping=None):
        self.LineMapping = LineMapping
        self.detector = Predictor(model, ckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.filter_class = COCO_CLASSES
    def update(self, image, visual = True, logger_=True):
        height, width, _ = image.shape 
        _,info = self.detector.inference(image, visual=True, logger_=logger_)
        outputs = []
        for dir_, line in self.LineMapping.items():
            cv2.line(image, line[0], line[1], (46,162,112), thickness=8, lineType=cv2.LINE_AA)

        if info['box_nums']>0:
            bbox_xywh = []
            scores = []
            objectids = []
            #bbox_xywh = torch.zeros((info['box_nums'], 4))
            for [x1, y1, x2, y2], class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                # if self.filter_class and class_names[int(class_id)] not in self.filter_class:
                #     continue
                # if score < 0.9 and class_names[int(class_id)]  == "bus":
                #     continue
                # color = compute_color_for_labels(int(class_id))
                bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])  
                objectids.append(info['class_ids'])             
                scores.append(score)
                
            bbox_xywh = torch.Tensor(bbox_xywh)
            outputs = self.deepsort.update(bbox_xywh, scores, info['class_ids'],image)
            data = []
            if len(outputs) > 0:
                if visual:
                    if len(outputs) > 0:
                        bbox_xyxy =outputs[:, :4]
                        identities =outputs[:, -2]
                        object_id =outputs[:, -1]
                        if self.LineMapping:
                            image, data = draw_boxes(image, bbox_xyxy, object_id,identities, LineMapping = self.LineMapping)
                            image = vis_track(image, outputs)
            return image, outputs , data

class SORTTracker():
    def __init__(self, filter_class=None, model='yolox-s', ckpt='yolox_s.pth'):
        self.detector = Predictor(model, ckpt)
        self.filter_class = filter_class
        self.sort = Sort()
    def update(self, image, visual = True, logger_=True):
        _,info = self.detector.inference(image, visual=True, logger_=logger_)
        outputs = []
        if info['box_nums']>0:
            bbox_xywh = []
            scores = []
            objectids = []
            #bbox_xywh = torch.zeros((info['box_nums'], 4))
            for [x1, y1, x2, y2], class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                if self.filter_class and class_names[int(class_id)] not in self.filter_class:
                    continue
                # color = compute_color_for_labels(int(class_id))
                bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])  
                objectids.append(info['class_ids'])             
                scores.append(score)
                
            bbox_xywh = torch.Tensor(bbox_xywh)
            outputs = self.sort.update(bbox_xywh)
            if len(outputs) > 0:
                if visual:
                    image = vis_track8(image, outputs)

            return image, outputs

if __name__=='__main__':
    from LineMappings import LM
    try:
        LineMapping = LM[sys.argv[1].split('/')[-1]]    
    except:
        LineMapping = None
        
    tracker = Tracker(filter_class=None, model='yolox-x', ckpt='weights/yolox_x.pth', LineMapping=LineMapping)    # instantiate Tracker

    cap = cv2.VideoCapture(sys.argv[1]) 
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    length = int(cv2.VideoCapture.get(cap, property_id))

    vid_writer = cv2.VideoWriter(
        f'demo_{sys.argv[1]}', cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    ) # open one video
    frame_count = 0
    fps = 0.0
    while True:
        ret_val, frame = cap.read() # read frame from video
        x = [100, 100, 200, 200]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        label = "FPS: %.2f"%(fps)
        UI_box(x, frame, (211, 232, 21), label, 4, False)
        t1 = time_synchronized()
        if ret_val:
            # try:
            frame, bbox, data_dict = tracker.update(frame, visual=True, logger_=False)  # feed one frame and get result
            # except:
            #     print("Error")
            #     pass
            vid_writer.write(frame)
            if frame_count == 1000:
                break
            frame_count +=1
            print(frame_count, end="\r")
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
            fps  = ( fps + (1./(time_synchronized()-t1)) ) / 2
        else:
            break

    cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()
