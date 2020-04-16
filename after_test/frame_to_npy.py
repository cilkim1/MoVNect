import os
import cv2
import glob
import numpy as np

path = "D:/mpi_inf_3dhp/S1/Seq1/imageSequence"
new_path = "D:/mpi_inf_3dhp/S1/Seq1/imageSequence_npy"
root = glob.glob("{}/*.{}".format(path, "avi"))
if os.path.isdir(new_path) is False:
    os.mkdir(new_path)
for i in range(len(root)):
    vidcap = cv2.VideoCapture(root[i])
    success,image = vidcap.read()
    count = 0
    current_name, _ = os.path.splitext(os.path.basename(root[i]))
    while success:
        image = cv2.resize(image, dsize=(256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.expand_dims(image, -1)
        if count is 0:
            total_image = image
        else:
            total_image = np.concatenate([total_image, image], -1)
        success,image = vidcap.read()
        count += 1
    # print(total_image.shape, type(total_image))
    np.save("{}/{}.npy".format(new_path, current_name), total_image)