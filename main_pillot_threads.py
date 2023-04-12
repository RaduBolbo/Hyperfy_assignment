'''

Voi simula procesul.
a si b sunt listele de box-uri/ listele de label-uri

'''

######
# V2) Cu lock-uri
######

import threading
import time

from fasterrcnn import Fasterrcnn


a = 0
b = 0

# creez si niste lock-uri
a_lock = threading.Lock()
b_lock = threading.Lock()

# simuleaza reteaua cu interenta mai rapida
def thread1():
    # treb sa fie global ca sa fie viziboil
    global a
    time.sleep(0.09) # simulare workload
    with a_lock:
        a += 1

# simuleaza reteaua cu interenta mai lenta
def thread2():
    # treb sa fie global ca sa fie viziboil
    global b
    time.sleep(0.15) # simulare workload
    with b_lock:
        b += 1

'''
Aici undeva ar trebui sa vina INCARCAREA MODELELOR, INAINTEA incvarcarii filmului
'''

t1 = threading.Thread(target=thread1)
t2 = threading.Thread(target=thread2)
t1.start()
t2.start()

while True:

    
    if not t1.is_alive(): # se verifica daca th1 si-a termiant trreaba, ca sa pot trimite un alt frame spre inferenta
        t1 = threading.Thread(target=thread1)
        t1.start()
    if not t2.is_alive(): # se verifica daca th1 si-a termiant trreaba, ca sa pot trimite un alt frame spre inferenta
        t2 = threading.Thread(target=thread2)
        t2.start()

    # print-ul simuleaza afisarea imaginii
    with a_lock, b_lock:
        print("a =", a, "b =", b)
    time.sleep(0.01) # asta simuleaza workload-ul suplimentar



######
# V1) Fara lock-uri
######
"""
import threading
import time


a = 0
b = 0

# simuleaza reteaua cu interenta mai rapida
def thread1():
    # treb sa fie global ca sa fie viziboil
    global a
    time.sleep(0.09) # simulare workload
    a += 1

# simuleaza reteaua cu interenta mai lenta
def thread2():
    # treb sa fie global ca sa fie viziboil
    global b
    time.sleep(0.15) # simulare workload
    b += 1

'''
Aici undeva ar trebui sa vina INCARCAREA MODELELOR, INAINTEA incvarcarii filmului
'''

t1 = threading.Thread(target=thread1)
t2 = threading.Thread(target=thread2)
t1.start()
t2.start()

while True:

    
    if not t1.is_alive(): # se verifica daca th1 si-a termiant trreaba, ca sa pot trimite un alt frame spre inferenta
        t1 = threading.Thread(target=thread1)
        t1.start()
    if not t2.is_alive(): # se verifica daca th1 si-a termiant trreaba, ca sa pot trimite un alt frame spre inferenta
        t2 = threading.Thread(target=thread2)
        t2.start()

    # print-ul simuleaza afisarea imaginii
    print("a =", a, "b =", b)
    time.sleep(0.01) # asta simuleaza workload-ul suplimentar
"""























