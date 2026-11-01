import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의

def convert_bbox_coord(bbox):
    x,y,w,h = bbox
    x1, y1, x2, y2 = x, y, x+w, y+h

    return x1, y1, x2, y2


# TODO: 2) IoU 계산 함수 정의

def calculate_iou(bbox_a, bbox_b):
    xa_1, ya_1, xa_2, ya_2 = convert_bbox_coord(bbox_a)
    xb_1, yb_1, xb_2, yb_2 = convert_bbox_coord(bbox_b)

    # IoU의 좌상단 우하단 좌표
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    # IoU의 너비와 높이
    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)

    intersection_area = w * h
    union_area = ((xa_2 - xa_1) * (ya_2 - ya_1) + (xb_2 - xb_1) * (yb_2 - yb_1)) - intersection_area

    # TODO: ZeroDivisionError 주의
    return intersection_area / union_area if union_area > 0 else 0


# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거

def prune_low_conf(boxes, scores, conf_threshold=0.5):

    pruned_boxes = []
    pruned_scores = []

    for box, score in zip(boxes,scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)

    return pruned_boxes, pruned_scores

# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환

def apply_nms(boxes, scores, iou_threshold=0.5):

    # pop() 함수 사용으로 일반 리스트로 변환
    score_ordered_idx = np.argsort(scores).tolist()

    nms_boxes = []
    nms_scores = []

    while len(score_ordered_idx) > 0:

        highest_idx = score_ordered_idx.pop()

        best_box = boxes[highest_idx]
        best_score = scores[highest_idx]

        nms_boxes.append(best_box)
        nms_scores.append(best_score)

        # TODO: 나머지 BBox와 IoU 계산
        # TODO: IoU가 임계값보다 높은 BBox 제거
        remove_idx = []

        for idx in score_ordered_idx:
            iou = calculate_iou(best_box, boxes[idx])

            if iou >= iou_threshold:
                remove_idx.append(idx)

        for idx in remove_idx:
            score_ordered_idx.remove(idx)

    return nms_boxes, nms_scores

        