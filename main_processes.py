'''

Am descoperit ca in Python 2 threaduri nu ruleaza in paralel. GEnial

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

def process1(frame, a_boxes, a_labels, a_lock):
    ########################################## CRED CA SIA ICI AR TREBUI O RERGIUEN CRITICA  PENTR frame
    #print(type(frame))




    while True:
        #print(type(frame))
        if len(frame) != 0:
            image = frame.copy() ######## ca sa nu modifice img in master # ju mai e nevoie,m ca procesele nu mai impart memoria

            print('entered_thread ----------------------- 1')

            print(type(a_boxes), type(a_labels))

            print('entered_thread ----------------------- 2')

            image=image[:,:,0:3]
            image = np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print('entered_thread ----------------------- 3')

            with a_lock:
                print('entered_thread ----------------------- 4')
                # aplic inferenta
                start_time = time.time()
                a_boxes, a_labels = fasterrcnn.inference(image)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Timp inferernta: {elapsed_time:.2f} seconds")
                print('entered_thread ----------------------- 5')
                print('inferenta')


# simuleaza reteaua cu interenta mai lenta
def process2(b, b_lock):
    time.sleep(0.15) # simulare workload
    with b_lock:
        b.value += 1


if __name__ == '__main__':

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

    b = mp.Value('i', 0)

    a_boxes = mp.Manager().list([])
    b_boxes = mp.Manager().list([])

    a_labels = mp.Manager().list([])
    b_labels= mp.Manager().list([])



    frame = mp.RawArray('i', np.array([]))


    # creez si niste lock-uri
    a_lock = mp.Lock()
    b_lock = mp.Lock()
    frame_lock = mp.Lock()

    """
    Aici undeva ar trebui sa vina INCARCAREA MODELELOR, INAINTEA incvarcarii filmului
    """


    
    # poate il mut in interiorul procesului, in afara while True
    # Instantirre retea (de fapt inauntru nu e insttantiata reteaua, doar ca instantiez un obiect definit de mine)
    fasterrcnn = Fasterrcnn()
    # incarcare model
    fasterrcnn.load_weights()
    

    
    # create processes
    p1 = mp.Process(target=process1, args=(frame, a_boxes, a_labels, a_lock))
    p2 = mp.Process(target=process2, args=(b, b_lock))
    p1.start()
    #p2.start()
    

    first=True
    while True:

        # citire frame cu frame
        with frame_lock:
            ret, frame = cap.read()

        """
        if first:
            first = False
            # create processes
        p1 = mp.Process(target=process1, args=(frame, a_boxes, a_labels, a_lock))
        p2 = mp.Process(target=process2, args=(b, b_lock))
        p1.start()
        #p2.start()
        """

        """
        if not p1.is_alive(): # se verifica daca th1 si-a termiant trreaba, ca sa pot trimite un alt frame spre inferenta
            p1 = threading.Thread(target=process1, args=(frame, a_boxes, a_labels))
            p1.start()

        if not p2.is_alive(): # se verifica daca th1 si-a termiant trreaba, ca sa pot trimite un alt frame spre inferenta
            p2 = threading.Thread(target=process2, args=(b,))
            p2.start()
        """
        # iese din while, daca nu se returneaza corect frame-ul
        if not ret:
            #print("Error frame (1)")
            #? poate n-ar fi bine sa dau chiar break. Vedem mai incolo. Poate dau continue
            break
            #continue


        # Daca exista detection box-uri, se deseneaza peste imagine:
        if len(a_boxes) !=0 :

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



'''
# 
def process1(frame, a_boxes, a_labels, a_lock):
    ########################################## CRED CA SIA ICI AR TREBUI O RERGIUEN CRITICA  PENTR frame
    #print(type(frame))




    while True:
        #print(type(frame))
        if len(frame) != 0:
            image = frame.copy() ######## ca sa nu modifice img in master # ju mai e nevoie,m ca procesele nu mai impart memoria

            print('entered_thread ----------------------- 1')

            print(type(a_boxes), type(a_labels))

            print('entered_thread ----------------------- 2')

            image=image[:,:,0:3]
            image = np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print('entered_thread ----------------------- 3')

            with a_lock:
                print('entered_thread ----------------------- 4')
                # aplic inferenta
                start_time = time.time()
                a_boxes, a_labels = fasterrcnn.inference(image)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Timp inferernta: {elapsed_time:.2f} seconds")
                print('entered_thread ----------------------- 5')
                print('inferentainferentainferentainferentainferentainferentainferentainferentainferentainferentainferenta')


# simuleaza reteaua cu interenta mai lenta
def process2(b, b_lock):
    time.sleep(0.15) # simulare workload
    with b_lock:
        b.value += 1


if __name__ == '__main__':

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

    b = mp.Value('i', 0)

    a_boxes = mp.Manager().list([])
    b_boxes = mp.Manager().list([])

    a_labels = mp.Manager().list([])
    b_labels= mp.Manager().list([])



    frame = mp.RawArray('i', np.array([]))


    # creez si niste lock-uri
    a_lock = mp.Lock()
    b_lock = mp.Lock()
    frame_lock = mp.Lock()

    """
    Aici undeva ar trebui sa vina INCARCAREA MODELELOR, INAINTEA incvarcarii filmului
    """


    
    # il muty in interiorul procesului
    # Instantirre retea (de fapt inauntru nu e insttantiata reteaua, doar ca instantiez un obiect definit de mine)
    fasterrcnn = Fasterrcnn()
    # incarcare model
    fasterrcnn.load_weights()
    

    
    # create processes
    p1 = mp.Process(target=process1, args=(frame, a_boxes, a_labels, a_lock))
    p2 = mp.Process(target=process2, args=(b, b_lock))
    p1.start()
    #p2.start()
    

    first=True
    while True:

        # citire frame cu frame
        with frame_lock:
            ret, frame = cap.read()

        """
        if first:
            first = False
            # create processes
        p1 = mp.Process(target=process1, args=(frame, a_boxes, a_labels, a_lock))
        p2 = mp.Process(target=process2, args=(b, b_lock))
        p1.start()
        #p2.start()
        """

        #print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
        #print(type(frame))
        """
        if not p1.is_alive(): # se verifica daca th1 si-a termiant trreaba, ca sa pot trimite un alt frame spre inferenta
            p1 = threading.Thread(target=process1, args=(frame, a_boxes, a_labels))
            p1.start()

        if not p2.is_alive(): # se verifica daca th1 si-a termiant trreaba, ca sa pot trimite un alt frame spre inferenta
            p2 = threading.Thread(target=process2, args=(b,))
            p2.start()
        """
        # iese din while, daca nu se returneaza corect frame-ul
        if not ret:
            #print("Error frame (1)")
            #? poate n-ar fi bine sa dau chiar break. Vedem mai incolo. Poate dau continue
            break
            #continue


        # Daca exista detection box-uri, se deseneaza peste imagine:
        if len(a_boxes) !=0 :
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

'''



########################################## main cu threaduri ###############################################################

'''

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

'''










