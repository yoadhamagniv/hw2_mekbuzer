#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
import multiprocessing
import multiprocessing
import numpy as np
import random
from scipy import ndimage

class Worker(multiprocessing.Process):
    
    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()

        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''
        self.jobs = jobs
        self.result = result

        self.train_x, self.train_y = training_data
        self.batch_size = batch_size

        random.seed()  
        np.random.seed()

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''
        img = image.reshape(28, 28)
        rot = ndimage.rotate(img, angle, reshape=False, order=1, mode='constant', cval=0.0)
        return rot.reshape(784)

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
        img = image.reshape(28, 28)
        shifted = ndimage.shift(img, shift=(-dy, -dx), order=1, mode='constant', cval=0.0)
        return shifted.reshape(784)
    
    @staticmethod
    def add_noise(image, noise):
        '''Add noise to the image
        for each pixel a value is selected uniformly from the 
        range [-noise, noise] and added to it. 

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        An numpy array of same shape
        '''
        n = np.random.uniform(-noise, noise, size=image.shape)
        out = image + n
        out = np.clip(out, 0.0, 1.0)
        return out

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        img = image.reshape(28, 28)
        center = (28 - 1) / 2.0  # 13.5

        # mapping output->input:
        # in_r = r
        # in_c = c - tilt*(r-center)
        matrix = np.array([[1.0, 0.0],
                        [-tilt, 1.0]])
        offset = np.array([0.0, tilt * center])

        skewed = ndimage.affine_transform(
            img, matrix=matrix, offset=offset,
            order=1, mode='constant', cval=0.0
        )
        return skewed.reshape(784)

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        angle = random.uniform(-15, 15)
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        noise = random.uniform(0.0, 0.15)
        tilt = random.uniform(-0.25, 0.25)

        out = image
        out = self.rotate(out, angle)
        out = self.shift(out, dx, dy)
        out = self.skew(out, tilt)
        out = self.add_noise(out, noise)

        out = np.clip(out, 0.0, 1.0)
        return out

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        while True:
            msg = self.jobs.get()

            if msg is None:
                try:
                    self.jobs.task_done()
                except Exception:
                    pass
                break

            job_id, indexes = msg

            batch_x = self.train_x[indexes]

            aug_x = np.empty_like(batch_x)
            for i in range(batch_x.shape[0]):
                aug_x[i] = self.process_image(batch_x[i])

            self.result.put((job_id, aug_x))

            try:
                self.jobs.task_done()
            except Exception:
                pass