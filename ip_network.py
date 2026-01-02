#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
import os
import multiprocessing
import random
from preprocessor import Worker
from network import *

class IPNeuralNetwork(NeuralNetwork):
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        # 1. Create Workers
		# (Call Worker() with self.mini_batch_size as the batch_size)
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        n_workers = max(1, cpus - 1)

		# 2. Set jobs
		self.jobs = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        self.workers = []

        for _ in range(n_workers):
            worker = Worker(self.jobs, self.results, training_data, self.mini_batch_size)
            worker.start()
            self.workers.append(worker)

        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)
        
        # 3. Stop Workers
        for _ in range(n_workers):
            self.jobs.put(None)  # Send stop signal to workers

        for worker in self.workers:
            worker.join()  # Wait for workers to finish
        
        
    
    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        batches = []

        # Store original batches by job_id
        orig_x = {}
        orig_y = {}

        # Submit jobs: "augment this batch" (represented by indexes)
        for job_id in range(self.number_of_batches):
            indexes = random.sample(range(data.shape[0]), batch_size)
            orig_x[job_id] = data[indexes]
            orig_y[job_id] = labels[indexes]

            # Worker will use indexes to fetch data locally and augment it
            self.jobs.put((job_id, indexes))

        # Collect results (may arrive out of order)
        aug_x = {}
        while len(aug_x) < self.number_of_batches:
            job_id, ax = self.results.get()
            aug_x[job_id] = ax

        # Build final batches: original + augmented
        for job_id in range(self.number_of_batches):
            x0 = orig_x[job_id]
            y0 = orig_y[job_id]
            xa = aug_x[job_id]

            # Concatenate original and augmented images
            x_out = np.concatenate([x0, xa], axis=0)

            # Labels duplicated (augmented image keeps same label)
            y_out = np.concatenate([y0, y0], axis=0)

            batches.append((x_out, y_out))

        return batches 
