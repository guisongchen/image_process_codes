from cv2 import cv2
import numpy as np 
import matplotlib.pyplot as plt

def otsu_binarization(img, th=128):
    H, W = img.shape
    out = img.copy()


    # find bestTh to get max between variance(aka min within variance)
    maxSigma = 0.0
    maxTh = 0

    # determine threshold
    for _t in range(1, 255):
        subArray_0 = out[np.where(out < _t)]
        mean_0 = np.mean(subArray_0) if len(subArray_0) > 0 else 0.0
        wt_0 = float(len(subArray_0)) / (H * W)

        subArray_1 = out[np.where(out >= _t)]
        mean_1 = np.mean(subArray_1) if len(subArray_1) > 0 else 0.0
        wt_1 = float(len(subArray_1)) / (H * W)

        #print(mean_0, wt_0, mean_1, wt_1)

        curSigma = wt_0 * wt_1 * ((mean_0 - mean_1) ** 2)
        if curSigma > maxSigma:
            maxSigma = curSigma
            maxTh = _t

    print("threhold: {}".format(maxTh))

    out[out < maxTh] = 0
    out[out >= th] = 255

    return out





path = "./git/repositories/image_process_codes/data/{}".format(input("file name: "))
src = cv2.imread(path, 0)

cv2.imshow('source img', src)
cv2.waitKey(0)

out = otsu_binarization(src)
cv2.imwrite("out.jpg", out)
cv2.imshow('out img', out)
cv2.waitKey(0)
cv2.destroyAllWindows()