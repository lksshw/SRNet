"""
Skeletonization of text mask.
Change the original code to Python3 support.
Original project: https://github.com/anupamwadhwa/Skeletonization-of-Digital-Patterns
"""

import os
import cv2
import math
import numpy as np

def skeletonization(img, threshold = 127):

    A = img.copy()
    if len(A.shape) == 3:
        A = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
    A = np.squeeze((A > threshold).astype(np.uint8))
    h, w = A.shape

    while True:
        C = 0
        M = np.zeros((h, w))

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if A[i, j] == 0:
                    continue
                P = [A[i, j], A[i-1, j], A[i-1, j+1], A[i, j+1], A[i+1, j+1], A[i+1, j], A[i+1, j-1], A[i, j-1], A[i-1, j-1]]

                # condition 1
                b = 0
                for k in range(1, 9):
                    b = b + P[k]
                if b < 2 or b > 6:
                    continue

                # condition 2
                a = 0
                for k in range(1, 8):
                    if P[k] == 0 and P[k + 1] == 1:
                        a = a + 1
                if P[8] == 0 and P[1] == 1:
                    a = a + 1
                if not a == 1:
                    continue

                # condition 3 & 4
                if P[1] * P[3] * P[5] == 0 and P[3] * P[5] * P[7] == 0:
                    M[i, j] = 1
                    C = C + 1
        
        A = A - M
        M = np.zeros((h, w))
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if A[i, j] == 0:
                    continue
                P = [A[i, j], A[i-1, j], A[i-1, j+1], A[i, j+1], A[i+1, j+1], A[i+1, j], A[i+1, j-1], A[i, j-1], A[i-1, j-1]]

                # condition 1
                b = 0
                for k in range(1, 9):
                    b = b + P[k]
                if b < 2 or b > 6:
                    continue

                # condition 2
                a = 0
                for k in range(1, 8):
                    if P[k] == 0 and P[k + 1] == 1:
                        a = a + 1
                if P[8] == 0 and P[1] == 1:
                    a = a + 1
                if not a == 1:
                    continue

                # condition 3 & 4
                if P[1] * P[3] * P[7] == 0 and P[1] * P[5] * P[7] == 0:
                    M[i, j] = 1
                    C = C + 1

        A = A - M
        if C == 0:
            break
    return (A * 255).astype(np.uint8)