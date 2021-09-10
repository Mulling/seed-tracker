# See LICENSE for license details

#!/usr/bin/env python3

"""
simple_sort: A simplified version of the SORT algorithm made by Alex Bewley,
for the original sort algorithm see: github.com/abewley/sort. This tracker uses
the OpenCV Kalman filter and does not handle object occlusion. Running this
file directly will start a testing tool.
"""

import cv2 as cv
import random
import numpy as np

from scipy.optimize import linear_sum_assignment

use_centroid_model = True
use_manhattan_distance = False
use_iou_model = False


def iou(bb_test, bb_gt):
    """
    Computes IOU between two 'bboxes' in the form [x1, y1, x2, y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)


def bbox_to_arr(bbox):
    """
    Convert a bounding box into a array in the form [x, y, s, r] where:
    x, y = coordinates of the center of the bounding box,
    s    = scale, or the area of the bounding box,
    r    = aspect ration of the bounding box.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)

    return np.array([x, y, s, r], np.float32).reshape((4, 1))


def arr_to_bbox(arr):
    """
    Convert a arr into a bounding box of format [x1, y1, x2, y2] where:
    x1, y1 = coordinates of the top left corner of the bounding box,
    x2, y2 = coordinates of the bottom right corner of the bounding box.
    """
    w = np.sqrt(arr[3] * arr[2])
    h = arr[2] / w

    return np.array([arr[0] - w / 2.0,
                     arr[1] - h / 2.0,
                     arr[0] + w / 2.0,
                     arr[1] + h / 2.0], np.float32).reshape((4, 1))


class Tracker_s(object):
    """
    Internal state of each tracked object. This uses a model that is based only
    in the x and y position of the centroid of the object.
    """
    count = 0

    def __init__(self, c):
        self.kalman = cv.KalmanFilter(4, 2, 0)

        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], np.float32)

        self.kalman.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32)

        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.kalman.errorCovPost *= 0.1

        self.kalman.processNoiseCov *= 1
        self.kalman.processNoiseCov[2:, 2:] *= 10

        a = self.kalman.statePost

        a[:2] = c.reshape((2, 1))

        self.kalman.statePost = a

        self.age = 0
        self.hit = 0

        self.id = Tracker.count

        Tracker.count += 1

    def update(self, c):
        """
        Update the state vector with the new bounding box.
        """
        self.hit += 1
        self.age = 0
        self.kalman.correct(c.reshape((2, 1)).astype(np.float32))

    def predict(self):
        """
        Advance the state vector and return the predicted bounding box.
        """
        self.age += 1
        return self.kalman.predict()[:2]


class Tracker(object):
    """
    Internal state of each tracked object.
    """
    count = 0

    def __init__(self, bbox):
        self.kalman = cv.KalmanFilter(6, 4, 0)

        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]], np.float32)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]], np.float32)

        self.kalman.processNoiseCov = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]], np.float32)

        self.kalman.errorCovPost = np.eye(6, dtype=np.float32)
        self.kalman.errorCovPost *= 0.1

        self.kalman.processNoiseCov *= 1
        self.kalman.processNoiseCov[4:, 4:] *= 10

        a = self.kalman.statePost

        a[:4] = bbox_to_arr(bbox)

        self.kalman.statePost = a

        self.age = 0
        self.hit = 0

        self.id = Tracker.count

        Tracker.count += 1

    def update(self, bbox):
        """
        Update the state vector with the new bounding box.
        """
        self.hit += 1
        self.age = 0
        self.kalman.correct(bbox_to_arr(bbox))

    def predict(self):
        """
        Advance the state vector and return the predicted bounding box.
        """
        self.age += 1
        return arr_to_bbox(self.kalman.predict())


class SimpleSort(object):
    """
    Simplified implementation of the sort tracker.
    """
    def __init__(self, max_age=5, min_hits=5, max_dist=50):
        """
        'max_age' is the maximum amount of frames a tracker can be alive
        without a match. 'min_hits' the minimum amount of frames for a tracker
        to become active.
        """
        self.trackers = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_dist = max_dist

    def update(self, detections):
        """
        Update the state vector with 'detections'. Return will be a array of
        bounding boxes (of the detections) in the form: [x1, y1, x2, y2, ID],
        if using centroid model: [x1, y1], return will be in the form [x1, y1,
        ID].
        """
        trks = np.zeros((len(self.trackers), 2 if use_centroid_model else 4))

        to_del = []

        for t, trk in enumerate(trks):
            det = self.trackers[t].predict()

            if use_centroid_model:
                trk[:] = [det[0][0], det[1][0]]
            else:
                trk[:] = [det[0], det[1], det[2], det[3]]

            if np.any(np.isnan(trk)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        matched, unmatched_d = associate(trks, detections, self.max_dist)

        ret = []
        # update the matched trackers
        for t, trk in enumerate(self.trackers):
            if t in matched[:, 1]:
                # different than the original sort here we return the actual
                # detection, not the prediction.
                det = detections[matched[np.where(
                    matched[:, 1] == t
                )[0], 0], :][0]
                if trk.hit > self.min_hits:
                    # print(f"{det} associated with {trks[t]}")
                    ret.append(np.concatenate((det, [trk.id])).reshape(1, -1))
                trk.update(det)

        # instantiate the unmatched detections
        for i in unmatched_d:
            if use_centroid_model:
                self.trackers.append(Tracker_s(detections[i, :]))
            else:
                self.trackers.append(Tracker(detections[i, :]))

        # remove old trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.age > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 3)) if use_centroid_model else np.empty((0, 5))


# NOTE: lower max_dist will affect the ability of the tracker
# to work at higher speeds
def associate(trackers, detections, max_dist=25):
    """
    Try and associate each tracker with the detections.
    Return matched indices and unmatched detections.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=np.int32), np.arange(len(detections))

    mat = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            if use_centroid_model:
                if use_manhattan_distance:
                    mat[d, t] = np.abs(det - trk).sum(-1)
                else:
                    mat[d, t] = np.linalg.norm(np.array([det[0], det[1]]) -
                                               np.array([trk[0], trk[1]]))
            else:
                if use_iou_model:
                    ioveru = iou(det, trk)
                    if ioveru < 0.1:
                        mat[d, t] = 10 / np.linalg.norm(
                            np.array([(det[0] + det[2]) / 2.0,
                                      (det[1] + det[3]) / 2.0]) -
                            np.array([(trk[0] + trk[2]) / 2.0,
                                      (trk[1] + trk[3]) / 2.0])
                        )
                    else:
                        mat[d, t] = ioveru
                else:
                    mat[d, t] = np.linalg.norm(
                        np.array([(det[0] + det[2]) / 2.0,
                                  (det[1] + det[3]) / 2.0]) -
                        np.array([(trk[0] + trk[2]) / 2.0,
                                  (trk[1] + trk[3]) / 2.0])
                    )
            # print(f"{d}:{t} = {mat[d, t]}")

    if use_iou_model:
        matched_indices = np.transpose(np.asarray(linear_sum_assignment(-mat)))
    else:
        matched_indices = np.transpose(np.asarray(linear_sum_assignment(mat)))

    unmatched_dets = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    matches = []
    for m in matched_indices:
        if use_iou_model and mat[m[0], m[1]] < 0.001:
            unmatched_dets.append(m[0])
        elif not use_iou_model and mat[m[0], m[1]] > max_dist:
            unmatched_dets.append(m[0])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        return np.empty((0, 2), dtype=np.int32), np.array(unmatched_dets)

    return np.concatenate(matches, axis=0), np.array(unmatched_dets)


def __test_kalman_filter():
    """
    Test the Kalman filter.
    """
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    tracker = None
    pred = None

    def process_frame(img):
        def to_bounding_rec(c):
            x, y, w, h = cv.boundingRect(c)
            return [x, y, x + w, y + h]

        contours, _ = \
            cv.findContours(cv.cvtColor(img, cv.COLOR_RGB2GRAY), cv.RETR_LIST,
                            cv.CHAIN_APPROX_SIMPLE)

        return [to_bounding_rec(c) for c in contours]

    def draw_circle(event, x, y, f, p):
        img[::] = 0
        if event == cv.EVENT_MOUSEMOVE:
            cv.circle(img, (x, y), 5, (255, 255, 255), -1)


    cv.namedWindow('test')
    cv.setMouseCallback('test', draw_circle)

    while(True):
        img_show = img.copy()

        bboxes = process_frame(img_show)

        for x1, y1, x2, y2 in bboxes:
            cv.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if tracker is None and len(bboxes) > 0:
            if use_centroid_model:
                tracker = Tracker_s(
                    np.array([(bboxes[0][0] + bboxes[0][2]) / 2,
                              (bboxes[0][1] + bboxes[0][3]) / 2])
                )
            else:
                tracker = Tracker(bboxes[0])

            pred = tracker.predict()
        elif len(bboxes) > 0:
            if use_centroid_model:
                tracker.update(np.array([(bboxes[0][0] + bboxes[0][2]) / 2,
                                         (bboxes[0][1] + bboxes[0][3]) / 2]))
            else:
                tracker.update(bboxes[0])
            pred = tracker.predict()

        if pred is not None and not np.any(np.isnan(pred)):
            if use_centroid_model:
                cv.circle(img, (int(pred[0][0]), int(pred[1][0])), 2,
                          (0, 0, 255), -1)
            else:
                p1 = (int(pred[0]), int(pred[1]))
                p2 = (int(pred[2]), int(pred[3]))
                cv.rectangle(img_show, p1, p2, (0, 0, 255), 1)

        cv.imshow('test', img_show)

        key = cv.waitKey(1)

        if key & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


def __test_simple_sort():
    """
    Test the tracker.
    """

    img = np.zeros((500, 1000, 3), dtype=np.uint8)

    t = SimpleSort()

    def process_frame(img):
        def to_bounding_rec(c):
            x, y, w, h = cv.boundingRect(c)
            return [x, y, x + w, y + h]

        contours, _ = \
            cv.findContours(cv.cvtColor(img, cv.COLOR_RGB2GRAY), cv.RETR_LIST,
                            cv.CHAIN_APPROX_SIMPLE)
        if use_centroid_model:
            ret = []
            for c in contours:
                x1, y1, x2, y2 = to_bounding_rec(c)
                ret.append([((x1 + x2) / 2),
                            ((y1 + y2) / 2)])
            return ret
        else:
            return [to_bounding_rec(c) for c in contours]

    def draw_circle(event, x, y, f, p):
        img[::] = 0
        if event == cv.EVENT_MOUSEMOVE:
            for i in range(10):
                cv.circle(img, (x-(i*100) + random.randint(1, 5),
                                y + random.randint(1, 5)),
                          5, (255, 255, 255), -1)

    cv.namedWindow('test')
    cv.setMouseCallback('test', draw_circle,)

    while(True):
        img_show = img.copy()

        bboxes = process_frame(img_show)

        res = t.update(np.array(bboxes))
        if use_centroid_model:
            for x1, y1, id in res:
                cv.putText(img_show, str(id), (int(x1), int(y1)),
                           cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                cv.circle(img_show, (int(x1), int(y1)), 3, (0, 0, 255), 1)

        else:

            for x1, y1, x2, y2, id in res:
                p1 = (x1, y1)
                p2 = (x2, y2)
                cv.putText(img_show, str(id), p2,
                           cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                cv.rectangle(img_show, p1, p2, (0, 0, 255), 1)

        cv.imshow('test', img_show)

        key = 0xFF & cv.waitKey(1)

        if key == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    print(__doc__)
    __test_kalman_filter()
    __test_simple_sort()
