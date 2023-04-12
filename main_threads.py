'''

Am descoperit ca in Python 2 threaduri nu ruleaza in paralel. Genial

'''

import cv2
from imutils.video import VideoStream
import imutils

import threading
import time

import urllib.request
import cv2
from PIL import Image
from io import BytesIO

import numpy as np

from fasterrcnn import Fasterrcnn

from PIL import Image, ImageDraw, ImageFont

import multiprocessing as mp




#########
# Link-uri Livecam
#########


#########
# .mp4 local
#########

url = r'E:\Hyperfy_test\v1.mp4'




# Instantiez un obiuect din clasa VideoCapture
cap = cv2.VideoCapture(url)
print(cap.isOpened())

b = 0
a = 0

a_boxes = 0
b_boxes = 0

a_labels = 0
b_labels= 0

frame = 0


# creez si niste lock-uri
a_lock = threading.Lock()
b_lock = threading.Lock()

# 
def thread1():
    global frame
    ########################################## CRED CA SIA ICI AR TREBUI O RERGIUEN CRITICA  PENTR frame
    print(type(frame))
    image = frame.copy() ######## ca sa nu modifice img in master
    # treb sa fie global ca sa fie viziboil
    global a_boxes
    global a_labels

    print(type(a_boxes), type(a_labels))

    image=image[:,:,0:3]
    image = np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    with a_lock:
        # aplic inferenta
        a_boxes, a_labels = fasterrcnn.inference(image)

# simuleaza reteaua cu interenta mai lenta
def thread2():
    # treb sa fie global ca sa fie viziboil
    global b
    time.sleep(0.15) # simulare workload
    with b_lock:
        b += 1

"""
Aici undeva ar trebui sa vina INCARCAREA MODELELOR, INAINTEA incvarcarii filmului
"""

t1 = threading.Thread(target=thread1)
t2 = threading.Thread(target=thread2)
t1.start()
t2.start()

# Instantirre retea (de fapt inauntru nu e insttantiata reteaua, doar ca instantiez un obiect definit de mine)
fasterrcnn = Fasterrcnn()
# incarcare model
fasterrcnn.load_weights()


while True:

    # citire frame cu frame
    ret, frame = cap.read()
    
    if not t1.is_alive(): # se verifica daca th1 si-a termiant trreaba, ca sa pot trimite un alt frame spre inferenta
        t1 = threading.Thread(target=thread1)
        t1.start()

    if not t2.is_alive(): # se verifica daca th1 si-a termiant trreaba, ca sa pot trimite un alt frame spre inferenta
        t2 = threading.Thread(target=thread2)
        t2.start()

    # iese din while, daca nu se returneaza corect frame-ul
    if not ret:
        #print("Error frame (1)")
        #? poate n-ar fi bine sa dau chiar break. Vedem mai incolo. Poate dau continue
        break
        #continue


    # Daca exista detection box-uri:
    if a_boxes is not 0:
            print('***************************************************************')

            # ********
            # e nevoide sa transform img in PIL
            frame = Image.fromarray(frame)
            # ********
            # Draw the bounding boxes and labels on the image
            draw = ImageDraw.Draw(frame)
            font = ImageFont.truetype("arial.ttf", 16)

            # am nevoie de regiune critica daca doar citesc? Daca cumva sunt rezultate scrise doaer partial? Hai sa pun
            with a_lock:
                #print(a_boxes)
                #print(a_labels)
                for box, label in zip(a_boxes, a_labels):
                    #print(box.tolist())
                    #print(type(box.tolist()))
                    draw.rectangle(box.tolist(), outline="red")
                    draw.text((box[0], box[1]), label, font=font, fill="red")

            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    
    # Sdisplay frame
    cv2.imshow("Public Camera", frame)

    # q => quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release VideoCapture
cap.release()
cv2.destroyAllWindows() # elibereaza memoria