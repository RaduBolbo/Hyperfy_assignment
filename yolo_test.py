import subprocess # pt Git commands
import cv2
import numpy as np
import torch

# pentru masurarea timpului de inferenta
import time


"""
######
# Clone darknet repo pretrained pe COCO
######
repo_url = 'https://github.com/AlexeyAB/darknet.git'
clone_dir = 'E:\Hyperfy_test\dir_yolo_darknet'

subprocess.run(['git', 'clone', repo_url, clone_dir])
"""


from yolo import Yolo

model = Yolo()
model.load_weights()
img = cv2.imread('img1.PNG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # se ordoneaza corect planele de culoare
imgs = [img]  # batch of images
model.inference(imgs)


'''
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
img = cv2.imread('img1.PNG')
imgs = [img]  # batch of images

# Start the timer
start_time = time.time()
# inferenta
results = model(imgs) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
print(f"Timp inferernta: {elapsed_time:.2f} seconds")

print(type(results))
print(results) # Speed: 2ms
results.show()
'''




