from inspect import trace
import threading
import time

# global variable x
i = -1
# x = [0]*100



def generator(n):
    for i in range(n):
        yield i

def increment():
    global i

    idx = i + 1
    i += 1

    return idx


def thread_task(lock, val, x):
    # global x


    lock.acquire()
    idx = increment()
    lock.release()

    while idx < 100-1:
    
        assert x[idx] == 0, 'wrong!'
        x[idx] = val
        print(idx, val)
        # time.sleep(0.01)

        lock.acquire()
        idx = increment()
        lock.release()
        

def main_task():
  
    global n

    # creating a lock
    lock = threading.Lock()
    x = [0]*100

    # creating threads
    t1 = threading.Thread(target=thread_task, args=(lock, 1, x))
    t2 = threading.Thread(target=thread_task, args=(lock, 2, x))
    t3 = threading.Thread(target=thread_task, args=(lock, 3, x))
    

    # start threads
    t1.start()
    t2.start()
    t3.start()
    

    # wait until threads finish their job
    t1.join()
    t2.join()
    t3.join()

    return x

if __name__ == "__main__":
    x = main_task()
    print(x)

