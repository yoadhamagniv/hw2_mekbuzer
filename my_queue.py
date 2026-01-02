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
        # Pipe(duplex=False) -> one end is read-only, one end is write-only
        self._recv_conn, self._send_conn = multiprocessing.Pipe(duplex=False)

        # Writers-many: protect sending so multiple writers won't interleave bytes
        self._lock = multiprocessing.Lock()

        # Message counter:
        # empty() is allowed to be "optimistic empty" while a put is in progress.
        # We increment only AFTER send_bytes finishes, so empty() becomes False only
        # when the message is fully sent and put() finished.
        self._count = multiprocessing.Value('i', 0)

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)

        with self._lock:
            self._send_conn.send_bytes(data)
            self._count.value += 1

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        data = self._recv_conn.recv_bytes()
        msg = pickle.loads(data)

        # Decrement count after successful receive
        with self._lock:
            self._count.value -= 1

        return msg
    
    def empty(self):
        '''Get whether the queue is currently empty
            
        Return
        ------
        A boolean value
        '''
        # Only reader calls empty(), but we still guard with lock because writers update _count.
        with self._lock:
            return self._count.value == 0