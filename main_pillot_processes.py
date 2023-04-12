

import multiprocessing as mp
import time

def process1(a, lock):
    while True:
        with lock:
            a.value += 1
        time.sleep(0.09) # simulate workload

def process2(b, lock):
    while True:
        with lock:
            b.value += 1
        time.sleep(0.55) # simulate workload

if __name__ == '__main__':
    # create shared memory variables
    a = mp.Value('i', 0)
    b = mp.Value('i', 0)

    # create lock
    lock = mp.Lock()

    # create processes
    p1 = mp.Process(target=process1, args=(a, lock))
    p2 = mp.Process(target=process2, args=(b, lock))
    p1.start()
    p2.start()

    while True:
        # print the values of a and b
        with lock:
            print("a = ", a.value, "b = ", b.value)
        time.sleep(0.01)



'''
import multiprocessing as mp
import time

def process1(a):
    while True:
        a.value += 1
        time.sleep(0.09) # simulate workload

def process2(b):
    while True:
        b.value += 1
        time.sleep(0.55) # simulate workload

if __name__ == '__main__':
    # create shared memory variables
    a = mp.Value('i', 0)
    b = mp.Value('i', 0)

    # create processes
    p1 = mp.Process(target=process1, args=(a,))
    p2 = mp.Process(target=process2, args=(b,))
    p1.start()
    p2.start()

    while True:
        # print the values of a and b
        print("a = ", a.value, "b = ", b.value)
        time.sleep(0.01)
'''



'''
import multiprocessing as mp
import time

from fasterrcnn import Fasterrcnn

def process1(a):
    time.sleep(0.09) # simulate workload
    a.value += 1

def process2(b):
    time.sleep(0.55) # simulate workload
    b.value += 1

if __name__ == '__main__':
    # create shared memory variables
    a = mp.Value('i', 0)
    b = mp.Value('i', 0)

    # create processes
    p1 = mp.Process(target=process1, args=(a,))
    p2 = mp.Process(target=process2, args=(b,))
    p1.start()
    p2.start()

    while True:
        if not p1.is_alive():
            p1 = mp.Process(target=process1, args=(a,))
            p1.start()
        if not p2.is_alive():
            p2 = mp.Process(target=process2, args=(b,))
            p2.start()

        # print the values of a and b
        print("a = ", a.value, "b = ", b.value)
        time.sleep(0.01)
'''
