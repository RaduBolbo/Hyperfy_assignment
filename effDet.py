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



class EffDet():
    def __init__(self):
        super(EffDet, self).__init__()
        self.model = None
        # astea ar trebui sa fie labels de pe COCO (desi vad ca nu sunt)
        self.class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',    'teddy bear', 'hair drier', 'toothbrush']


    def load_weights(self):
        # Model
        self.model = hub.load(r"https://tfhub.dev/tensorflow/efficientdet/d6/1")

    def inference(self, imgs, showresult=True):
        # se presupune ca img e pus in batch, ca are ordinea canalelor RGB si ca ARE VALORI IN 0, 255

        if self.model is None:
            print('You have to first call "load_weights" method')
            return 1, 2
        else:
            # img to tensor
            tensor = tf.convert_to_tensor(imgs, dtype=tf.uint8) # se pare ca modelul cere sa fie uint8
            # APLIC INFERENTA
            detector_output = self.model(tensor)
            
            class_ids = detector_output["detection_classes"]
            boxes = detector_output["detection_boxes"]
            # le transform in liste
            class_ids = class_ids.numpy().tolist()
            boxes = boxes.numpy().tolist()

            return boxes[0], class_ids[0] # asa au ele formatul si trebuie sa fac [0]
        
        


'''
# Open the image file
img = cv2.imread('img1.png')
h, w, c = img.shape

img=img[:,:,0:3]
img = np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img_batched = [img] # simulez batch

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

# Load the model
model = torch.hub.load('rwightman/efficientdet-pytorch', 'efficientdet_d5', pretrained=True)

# Define the input image transformation
transform = T.Compose([
    T.Resize(512),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the input image from URL
url = 'https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg'
img = cv2.imdecode(np.asarray(bytearray(urllib.request.urlopen(url).read()), dtype=np.uint8), cv2.IMREAD_COLOR)

# Apply the input image transformation
input_img = transform(img).unsqueeze(0)

# Put the model in evaluation mode
model.eval()

# Make a prediction
with torch.no_grad():
    output = model(input_img)

# Get the boxes, scores, and labels from the output
boxes = output[0]['boxes'].cpu().numpy()
scores = output[0]['scores'].cpu().numpy()
labels = output[0]['labels'].cpu().numpy()

# Print the boxes, scores, and labels
print('Boxes:', boxes)
print('Scores:', scores)
print('Labels:', labels)

# Draw the boxes on the image
for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes[i]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Display the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''


