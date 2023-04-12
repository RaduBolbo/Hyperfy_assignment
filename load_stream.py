import cv2
from imutils.video import VideoStream
import imutils

import os



import threading

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
#rtsp1 = 'rtsp://userid:password@192.168.1.108:8080/h264_ulaw.sdp'
#rtsp='rtsp://userid:password@192.168.1.136:10554/udp/av0_0'

url = "https://play.webcamromania.ro/b3p4l5g5v534o20313"

def getPicture(url):      

    cam = cv2.VideoCapture(url)
    a=cam.get(cv2.CAP_PROP_BUFFERSIZE)
    cam.set(cv2.CAP_PROP_BUFFERSIZE,3)
    start_frame_number = 20
    cam.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

    print("buffer"+str(a))

    while True:        
        ret, frame = cam.read()
        cam.release()
        #print(ret, frame)
        if (ret==False):
            continue
        small_frame = cv2.resize(frame, (0, 0), fx=.50, fy=.50)
        small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("camera",small_frame)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    # When everything done, release the video capture object
    cam.release()


        # Closes all the frames
    cv2.destroyAllWindows() 

if __name__ == '__main__':

    #t1 = threading.Thread(target=getPicture, args=(rtsp,))
    #t2 = threading.Thread(target=getPicture, args=(rtsp1,))
    t = threading.Thread(target=getPicture, args=(url,))

    #t1.start()
    #t2.start()
    t.start()


















################################## asta ar trebui sa mearga
'''
#########
# Link-uri Livecam
#########

# URL of the public camera
url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_175k.mov"
url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_175k.mov"
url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_175k.mov"
#url = "http://192.168.18.37:8090/video"
#url = "https://cdn.skylinewebcams.com/_7960249963.webp"
#url = "https://cdn.skylinewebcams.com/_7960249963.webp"
url = "https://play.webcamromania.ro/b3p4l5g5v534o20313" # piata romana - nu da eroare, dar nici nu returneaza frameruri
#url = "blob:https://rtsp.me/1898e8d0-d146-40c2-af06-94acbdcb9c0d"
#url = "https://youtu.be/rs2be3mqryo"
#url = 'rtsp://85.186.25.54/axis-media/media.amp'
#url = 'rtsp://userid:password@192.168.1.108:8080/h264_ulaw.sdp'
#url = 'rtsp://userid:password@192.168.1.136:10554/udp/av0_0'
#########
# .mp4 local
#########

#url = r'E:\Hyperfy_test\v1.mp4'

import urllib.request
import cv2
from PIL import Image
from io import BytesIO


# Instantiez un obiuect din clasa VideoCapture
cap = cv2.VideoCapture(url)
print(cap.isOpened())

while True:
    # citire frame cu frame
    ret, frame = cap.read()
    print(ret, frame)

    # iese din while, daca nu se returneaza corect frame-ul
    if not ret:
        #print("Error frame (1)")
        #? poate n-ar fi bine sa dau chiar break. Vedem mai incolo. Poate dau continue
        break
        #continue

    # Sdisplay frame
    cv2.imshow("Public Camera", frame)

    # q => quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release VideoCapture
cap.release()
cv2.destroyAllWindows() # elibereaza memoria

'''











'''
print('1')
stream = VideoStream(url).start()
print('2')

while True:
    print('3')
    frame = stream.read()
    if frame is None:
        print(4)
        continue

    frame = imutils.resize(frame,width=1200)
    cv2.imshow('AsimCodeCam', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
stream.stop()
'''

