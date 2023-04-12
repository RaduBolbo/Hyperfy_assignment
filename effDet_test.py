
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import urllib.request

import tensorflow as tf

import tensorflow as tf
import numpy as np
import cv2
import urllib.request
#import hub
import tensorflow_hub as hub

from effDet import EffDet

from PIL import Image, ImageDraw, ImageFont


# Open the image file
img = cv2.imread('img1.png')
h, w, c = img.shape

img=img[:,:,0:3]
img = np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img_batched = [img] # simulez batch

##################################### predictia ###################################

# Instantirre retea (de fapt inauntru nu e insttantiata reteaua, doar ca instantiez un obiect definit de mine)
eff_det = EffDet()
# incarcare model
eff_det.load_weights()
# aplic inferenta
boxes, class_ids = eff_det.inference(img_batched)

###################################################################################

# e nevoide sa transform img in PIL
img = Image.fromarray(img)

# Draw the bounding boxes and labels on the image
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", 16)


for box, label in zip(boxes, class_ids):
    #draw.rectangle(box[0], outline="red")
    #print([box[0]*h, box[1]*w, box[2]*h, box[3]*w])
    #print(box[0])
    draw.rectangle([box[1]*w, box[0]*h, box[3]*w, box[2]*h], outline="red")
    #print(int(label-1))
    draw.text((box[0]*h, box[1]*w), str(int(label)), font=font, fill="red")

# Display the image
img.show()

'''

# img to tensor
tensor = tf.convert_to_tensor(img_batched, dtype=tf.uint8) # se pare ca modelul cere sa fie uint8

# Incarcare retea
detector = hub.load(r"https://tfhub.dev/tensorflow/efficientdet/d6/1")

detector_output = detector(tensor)
class_ids = detector_output["detection_classes"]
boxes = detector_output["detection_boxes"]

#output_img = detector_output

###############################################################

# le transform in liste?
class_ids = class_ids.numpy().tolist()
boxes = boxes.numpy().tolist()

# e nevoide sa transform img in PIL
img = Image.fromarray(img)

# Draw the bounding boxes and labels on the image
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", 16)

for box, label in zip(boxes[0], class_ids[0]):
    #draw.rectangle(box[0], outline="red")
    #print([box[0]*h, box[1]*w, box[2]*h, box[3]*w])
    #print(box[0])
    draw.rectangle([box[1]*w, box[0]*h, box[3]*w, box[2]*h], outline="red")
    print(int(label-1))
    draw.text((box[0]*h, box[1]*w), str(class_labels[int(label)]), font=font, fill="red")

# Display the image
img.show()

'''












'''

import torch
import torchvision.transforms as T
import numpy as np
import cv2
import urllib.request

import tensorflow as tf

import tensorflow as tf
import numpy as np
import cv2
import urllib.request
#import hub
import tensorflow_hub as hub

from PIL import Image, ImageDraw, ImageFont



# Labels COCO_V1
class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',    'teddy bear', 'hair drier', 'toothbrush']
print(len(class_labels))


# Open the image file
img = cv2.imread('img1.png')
h, w, c = img.shape

img=img[:,:,0:3]
img = np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img_batched = [img] # simulez batch



#print(np.unique(img))



# Convert the image to a TensorFlow tensor
tensor = tf.convert_to_tensor(img_batched, dtype=tf.uint8) # se pare ca modelul cere sa fie uint8

# Print the shape of the tensor
#print(tensor)
#print(tensor.shape)
#print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')

# Apply image detector on a single image.
detector = hub.load(r"https://tfhub.dev/tensorflow/efficientdet/d6/1")

detector_output = detector(tensor)
class_ids = detector_output["detection_classes"]
boxes = detector_output["detection_boxes"]

#print('lllllllllllllllllllllllllllllllllllllllllll')
#print(boxes)
#print(class_ids)

#output_img = detector_output

###############################################################

# le transform in liste?
class_ids = class_ids.numpy().tolist()
boxes = boxes.numpy().tolist()

# e nevoide sa transform img in PIL
img = Image.fromarray(img)

# Draw the bounding boxes and labels on the image
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", 16)

for box, label in zip(boxes[0], class_ids[0]):
    #draw.rectangle(box[0], outline="red")
    #print([box[0]*h, box[1]*w, box[2]*h, box[3]*w])
    #print(box[0])
    draw.rectangle([box[1]*w, box[0]*h, box[3]*w, box[2]*h], outline="red")
    print(int(label-1))
    draw.text((box[0]*h, box[1]*w), str(class_labels[int(label)]), font=font, fill="red")

# Display the image
img.show()


'''











