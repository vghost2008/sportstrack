import os.path

import h5py
import numpy as np
import cv2
path = r"/home/yangxiaodong/Data/002.h5"
file = h5py.File(path, 'r')
print(file)
a = np.array(file['events']['event_gs'])
x = np.array(file['events']['xs'])
y = np.array(file['events']['ys'])
t = np.array(file['events']['ts'])

length = len(set(t))
frame_rate = len(a) // length

for i in range(frame_rate):
    x_tmp = x[i*length:(i+1)* length]
    y_tmp = y[i*length:(i+1)* length]
    a_tmp = a[i*length:(i+1)* length]
    image = np.zeros((np.max(y) + 1, np.max(x) + 1))
    for j in range(len(x_tmp)):
        image[y_tmp[j], x_tmp[j]] = a_tmp[j]
    image = image.astype(np.uint8)
    cv2.imwrite(os.path.join("/home/yangxiaodong/Data/tmp3", str(i)+'.jpg'), image)
# image = a.reshape(x, y)
#     image = np.array(image).astype(np.uint8)
#     cv2.imshow("image", image)
#     cv2.waitKey(0)