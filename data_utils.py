import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import tensorflow as tf
import time

def get_time():
	return time.strftime("%Y/%m/%d, %H:%M:%S", time.localtime())

def transform(x, L1, L2):
	return x*(L2-L1) + L1

def sqrt_l2_norm(x):
	return np.sqrt(np.sum(x**2, axis = 0, keepdims = True))

def l2_error(true, pred, relative = True):
	error = sqrt_l2_norm(true - pred)
	if relative:
		error /= sqrt_l2_norm(true)
	return error

def tensor_grid(x):
	"""build tensor grid for multiple parameters
	
	Arguments:
		x {tuple or list of np.array} -- parameters
	
	Returns:
		grid {np.ndarray} -- tensor grids

	Example:
		>>> tensor_grid(([1, 2], [3, 4], [5, 6, 7]))
		>>> np.array([[1, 3, 5],
					[1, 3, 6],
					[1, 3, 7],
					[1, 4, 5],
					[1, 4, 6],
					[1, 4, 7],
					[2, 3, 5],
					[2, 3, 6],
					[2, 3, 7],
					[2, 4, 5],
					[2, 4, 6],
					[2, 4, 7]])
	"""
	return np.vstack(np.meshgrid(*x, indexing = 'ij')).reshape((len(x), -1)).T

class batch_generator:
	def __init__(self, dataset, batch_size):
		self.dataset = dataset
		self.batch_size = batch_size
		self.index = np.arange(dataset[0].shape[0])
		np.random.shuffle(self.index)
		self.pointer = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self.pointer >= len(self.index):
			np.random.shuffle(self.index)
			self.pointer = 0
		self.pointer += self.batch_size
		return cast_to_tf_constant((d[self.pointer-self.batch_size:self.pointer, :] for d in self.dataset), tf.float32)

def cast_to_tf_constant(xlist, dtype = tf.float32):
	return list(map(lambda x: tf.constant(x, dtype = dtype), xlist))

