import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import os

import tensorflow as tf

import random
random.seed(20)

class dataloader:
	"""
	Class to handle the dataloader for mini-batch training (for collocation points)
	It can be used also for mini-batch training with exact data if needed
	"""
	def __init__(self, datasets_class, batch_size, reshuffle_every_epoch):
		"""!
		Constructor

		@param datasets_class an object of type datasets_class that contains all the datasets we need
		@param batch_size dimension of a batch_size for collocation points
		@param reshuffle_every_epoch boolean that indicates if we want to reshuffle the points at every epoch
		"""

		## datasets_class object, we'll use its getter methods(get_coll_data(), get_num_collocation() etc)
		self.datasets_class = datasets_class
		## batch size of collocation points for minibatch training
		self.batch_size = batch_size
		## boolean for reshuffle_every_epoch or not
		self.reshuffle_each_iteration = reshuffle_every_epoch
	###############################################################################################

	def dataload_collocation(self):
		"""Return a dataloader for collocation points.
		Implemented using tf.data.Dataset.from_tensor_slices and batch"""
		# get the collocation data
		inputs,_,_ = self.datasets_class.get_coll_data()

		#load the data of collocation
		data = tf.data.Dataset.from_tensor_slices(inputs)

		#load data in batch of size = batch_size and shuffle
		data = data.shuffle(buffer_size=(self.datasets_class.get_num_collocation()+1),
							reshuffle_each_iteration=self.reshuffle_each_iteration)
		coll_loader = data.batch(batch_size=self.batch_size)

		print('len(collocation data) is', len(data))
		print('len(dataloader) is', len(coll_loader))
		if(self.reshuffle_each_iteration):
			print('Reshuffled at every epochs')

		return coll_loader,self.batch_size

	def dataload_exact(self, exact_batch_size):
		"""!
		Return a dataloader for exact points (only if needed.)
		You have to provide an additional parameter "exact_batch_size"
		Implemented using tf.data.Dataset.from_tensor_slices and batch

		@param exact_batch_size dimension of exact points batch"""
		# get the exact (noisy) data
		inputs,_,_ = self.datasets_class.get_exact_data_with_noise()

		#load the data of collocation
		data = tf.data.Dataset.from_tensor_slices(inputs)

		#load data in batch of size = batch_size and shuffle
		data = data.shuffle(buffer_size=(self.datasets_class.get_num_exact()+1),
							reshuffle_each_iteration=self.reshuffle_each_iteration)
		exact_loader = data.batch(batch_size=exact_batch_size)

		print('len(exact data) is', len(data))
		print('len(dataloader) is', len(train_loader))
		if(self.reshuffle_each_iteration):
			print('Reshuffled at every epochs')

		return exact_loader,exact_batch_size
