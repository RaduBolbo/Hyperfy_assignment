
# Din pacate are acelasi timp de inferenta ca yolo

import cv2
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

import numpy as np

# pentru masurarea timpului de inferenta
import time

from fasterrcnn import Fasterrcnn


# Load your input image
#image = Image.open('img1.PNG') # Load your image here
#image = image.convert("RGB")
#print(np.unique(image))

# ********
# Open the image file
image = cv2.imread('img1.png')
h, w, c = image.shape

image=image[:,:,0:3]
image = np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# ********

##################################### predictia ###################################

# Instantirre retea (de fapt inauntru nu e insttantiata reteaua, doar ca instantiez un obiect definit de mine)
fasterrcnn = Fasterrcnn()
# incarcare model
fasterrcnn.load_weights()
# aplic inferenta
boxes, labels = fasterrcnn.inference(image)

###################################################################################

# ********
# e nevoide sa transform img in PIL
image = Image.fromarray(image)
# ********


# Draw the bounding boxes and labels on the image
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 16)

#print('lllllllllllllllllll')
#print(boxes)
#print(labels)
for box, label in zip(boxes, labels):
    #print(box.tolist())
    #print(type(box.tolist()))
    draw.rectangle(box.tolist(), outline="red")
    draw.text((box[0], box[1]), label, font=font, fill="red")

# Display the image
image.show()



'''
#model = torch.hub.load('pytorch/vision:v0.10.0', 'faster_rcnn_resnet50_fpn', pretrained=True)
# Load the pre-trained Faster R-CNN model from the PyTorch Hub
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Labels COCO_V1
class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']

# Load your input image
image = Image.open('img1.PNG') # Load your image here
image = image.convert("RGB")


# Convert the image to a PyTorch tensor
image_tensor = torchvision.transforms.functional.to_tensor(image)

# Add a batch dimension to the tensor
image_tensor = image_tensor.unsqueeze(0)

# Pass the input tensor through the model
with torch.no_grad():
    start_time = time.time()
    # inferenta
    outputs = model(image_tensor)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Timp inferernta: {elapsed_time:.2f} seconds")

# Extract the bounding boxes and class labels from the model output
boxes = outputs[0]["boxes"]
labels = outputs[0]["labels"]

# Print the bounding boxes and class labels
print(boxes)
print(labels)

# Draw the bounding boxes and labels on the image
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 16)

for box, label in zip(boxes, labels):
    print(box.tolist())
    print(type(box.tolist()))
    draw.rectangle(box.tolist(), outline="red")
    draw.text((box[0], box[1]), str(class_labels[label.item()-1]), font=font, fill="red")

# Display the image
image.show()

'''





'''
import os
import numpy as np
import pandas as pd

import torch
import torchvision

import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

images_dir = '/kaggle/input/ship-detection/images/'
annotations_dir = '/kaggle/input/ship-detection/annotations/'

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

sample_id = 44

sample_image_path = f'img1.png'
sample_annot_path = f'img1.xml'

sample_image = Image.open(sample_image_path)
sample_image

with open(sample_annot_path) as annot_file:
    print(''.join(annot_file.readlines()))

tree = ET.parse(sample_annot_path)
root = tree.getroot()

sample_annotations = []

for neighbor in root.iter('bndbox'):
    xmin = int(neighbor.find('xmin').text)
    ymin = int(neighbor.find('ymin').text)
    xmax = int(neighbor.find('xmax').text)
    ymax = int(neighbor.find('ymax').text)
    
    sample_annotations.append([xmin, ymin, xmax, ymax])
    
print('Ground-truth annotations:', sample_annotations)

sample_image_annotated = sample_image.copy()

img_bbox = ImageDraw.Draw(sample_image_annotated)

for bbox in sample_annotations:
    img_bbox.rectangle(bbox, outline="white") 
    
sample_image_annotated
'''




