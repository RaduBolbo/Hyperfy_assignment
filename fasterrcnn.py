

# Din pacate are acelasi timp de inferenta ca yolo

import cv2
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

# pentru masurarea timpului de inferenta
import time



class Fasterrcnn():
    def __init__(self):
        super(Fasterrcnn, self).__init__()
        self.model = None
        # Labels COCO_V1
        self.class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']


    def load_weights(self):
        # Load the pre-trained Faster R-CNN model from the PyTorch Hub
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def inference(self, imgs, showresult=True):
        # se presupune ca img e unbatched, ca are ordinea canalelor RGB si CRED ca are valori in 0, 1

        if self.model is None:
            print('You have to first call "load_weights" method')
            return 1, 2
        else:
            # Convert the image to a PyTorch tensor
            image_tensor = torchvision.transforms.functional.to_tensor(imgs)
            # Add a batch dimension to the tensor
            image_tensor = image_tensor.unsqueeze(0)

            # Pass the input tensor through the model
            with torch.no_grad():
                start_time = time.time()
                # inferenta
                outputs = self.model(image_tensor)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Timp inferernta: {elapsed_time:.2f} seconds")
            
            # Extract the bounding boxes and class labels from the model output
            boxes = outputs[0]["boxes"]
            labels = outputs[0]["labels"]

            label_names = []
            for label in labels:
                #print('label.item:')
                #print(label.item())
                if label < len(self.class_labels): # nu inteleg de ce uneori ieise din range
                    label_names.append(self.class_labels[label.item()-1])

            return boxes, label_names # asa au ele formatul si trebuie sa fac [0]








'''


# Din pacate are acelasi timp de inferenta ca yolo

import cv2
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

# pentru masurarea timpului de inferenta
import time



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
















