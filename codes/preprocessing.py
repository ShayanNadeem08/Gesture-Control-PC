
import cv2
from torch import from_numpy

def convert_batch_grey(batch):
    batch = batch.numpy()
    batch = batch.transpose(0,1,3,4,2)
    grey_batch = []
    for video in batch
        grey_video = []
        for frame in video:
            grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey_video.append(grey_frame)
        grey_batch.append(grey_video)
    return from_numpy(grey_batch) 