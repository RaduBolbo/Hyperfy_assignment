import subprocess # pt Git commands
import cv2
import numpy as np
import torch

# pentru masurarea timpului de inferenta
import time





class Yolo():
    def __init__(self):
        super(Yolo, self).__init__()
        self.model = None

    def load_weights(self):
        # Model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def inference(self, imgs, showresult=True):
        if self.model is None: # nu inteleg de ce nu il recunoaste
            print('You hae to first call "load_weights" method')
        else:
            results = self.model(imgs)
            if showresult:
                results.show()
            return results

        

            




