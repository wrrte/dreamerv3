import numpy as np
import cv2
vid = np.random.randint(0, 255, (2, 68, 200, 3), dtype=np.uint8)
target_height = 512
scale_w = target_height / vid.shape[1]
target_width = int(vid.shape[2] * scale_w)
new_vid = np.empty((vid.shape[0], target_height, target_width, vid.shape[3]), dtype=vid.dtype)
for i in range(vid.shape[0]):
    new_vid[i] = cv2.resize(vid[i], (target_width, target_height), interpolation=cv2.INTER_NEAREST)
print(new_vid.shape)
