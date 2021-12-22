from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import argparse
import pyflow

# Flow Options:
ALPHA = 0.012
RATIO = 0.75
MIN_WIDTH = 20
N_OUTER_FP_ITERATIONS = 7
N_INNER_FP_ITERATIONS = 1
N_SORT_ITERATIONS = 30
COL_TYP = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--video', type=str, default='examples/bclip1.mp4')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
_, prev_frame = cap.read()

while(cap.isOpened()):
    _, frame = cap.read()

    cv2.imshow("Input", frame)

    u, v, im2W = pyflow.coarse2fine_flow(
        prev_frame.astype(float) / 255., frame.astype(float) / 255., ALPHA, RATIO, MIN_WIDTH, 
        N_OUTER_FP_ITERATIONS, N_INNER_FP_ITERATIONS, N_SORT_ITERATIONS, COL_TYP
    )

    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    hsv = np.zeros(frame.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Optical Flow", rgb)

    prev_frame = frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()