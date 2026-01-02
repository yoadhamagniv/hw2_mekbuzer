#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#

import multiprocessing
import pickle

class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
                # One-direction pipe: recv only + send only
        self._recv_conn, self._send_conn = multiprocessing.Pipe(duplex=False)

        # Protect only the send side (many writers)
        self._send_lock = multiprocessing.Lock()

        # Separate lock for the counter (or use _count.get_lock())
        self._count = multiprocessing.Value('i', 0)

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)

        # IMPORTANT: lock only around send to avoid interleaving bytes
        with self._send_lock:
            self._send_conn.send_bytes(data)

        # Increment count ONLY after send completed
        with self._count.get_lock():
            self._count.value += 1

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        # IMPORTANT: do not hold any locks while blocking on recv
        data = self._recv_conn.recv_bytes()
        msg = pickle.loads(data)

        # Decrement count after receive
        with self._count.get_lock():
            self._count.value -= 1

        return msg
    
    def empty(self):
        '''Get whether the queue is currently empty
            
        Return
        ------
        A boolean value
        '''
        with self._count.get_lock():
            return self._count.value == 0