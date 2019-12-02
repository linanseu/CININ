import time
import socket
def rec_UDP():
    UDP_PORT = 8810
    UDP_IP = "0.0.0.0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    data = "".encode()
    counter = 0
    #while (data.decode())!="FINISH!":
    while (counter<100):
        data, addr = sock.recvfrom(1024)
        #print ("received message:", data.decode())
        counter = counter + 1
        #print(counter)
    return counter

def send_UDP():
    UDP_PORT = 8812
    UDP_IP = "18.218.33.187"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    MESSAGE = 'X'*1400
    for time in range(0,1000):
        sock.sendto(MESSAGE.encode(), (UDP_IP, UDP_PORT))
    #sock.sendto('FINISH!'.encode(), (UDP_IP, UDP_PORT))



from multiprocessing import TimeoutError
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=2)

#start_time = time.time()
async_result = pool.apply_async(rec_UDP, ()) # build a thread to wait for the results to come
#####
# do the normal convolutions
counter = 0
while (counter<10000000):
    counter = counter + 1;
####
pool.apply_async(send_UDP, ()) # build another thread to send the results
try:
    return_val = async_result.get(timeout = 5)  # try to wait for the results from the other CNNs to come
except TimeoutError:
    return_val = 'NOTHING!'
    #continue
print(return_val)



async_result = pool.apply_async(rec_UDP, ()) # we will reopen the receiver buffer to wait for the next around of packet to come
############
#do the conv operations
counter = 0
while (counter<10000000):
    counter = counter + 1;
############
pool.apply_async(send_UDP, ()) # build another thread to send the results
# do some other stuff in the main process
try:
    return_val = async_result.get(timeout = 5)  # try to wait for the results from the other CNNs to come
except TimeoutError:
    return_val = 'NOTHING!'
    #continue

print(return_val)

