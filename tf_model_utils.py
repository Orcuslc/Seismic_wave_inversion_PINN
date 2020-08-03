import numpy as np
import matplotlib.pyplot as plt
from functools import wraps, reduce
import tensorflow as tf
import tensorflow.keras as keras
# import tensorflow_probability as tfp
import time
from .data_utils import *

# def cast_to_tf_constant(dtype, inclass = False):
# 	def decorate(func):
# 		@wraps(func)
# 		def cast(*args):
# 			newargs = []
# 			if inclass:
# 				newargs.append(args[0])
# 				args = args[1:]
# 			for arg in args:
# 				newargs.append(tf.constant(arg, dtype = dtype))
# 			return func(*newargs)
# 		return cast
# 	return decorate

class Fake_Model:
	def __init__(self, trainable_variables = None):
		self.trainable_variables = trainable_variables


# class LBFGS_compatible:

# 	def __init__(self, models, additional_variables = None):
# 		"""
# 			models, additional_variables: should be list
# 		"""
# 		self.idx = []
# 		self.part = []
# 		self.n_tensors = []
# 		self.shapes = []
# 		if additional_variables is None:
# 			self.models = models
# 		else:
# 			new_model = Fake_Model(additional_variables)
# 			self.models = models + [new_model]
# 		self._prepare(self.models)
# 		self.iter = tf.Variable(0, trainable = False)
		
# 	def _prepare(self, models):
# 		i = 0
# 		count = 0
# 		for model in models:
# 			shape = tf.shape_n(model.trainable_variables)
# 			self.shapes.extend(shape)
# 			self.n_tensors.append(len(shape))
# 			for shape_j in shape:
# 				n = np.product(shape_j)
# 				self.idx.append(tf.reshape(tf.range(count, count+n, dtype = tf.int32), shape_j))
# 				self.part.extend([i]*n)
# 				count += n
# 				i += 1

# 		self.part = tf.constant(self.part)
		
# 	@property
# 	def _variables(self):
# 		# return reduce(lambda x, y: x.trainable_variables+y.trainable_variables, [Fake_Model([])] + self.models)
# 		res = []
# 		for model in self.models:
# 			res += model.trainable_variables
# 		return res

# 	@tf.function
# 	def assign_new_model_parameters(self, params_1d):
# 		params = tf.dynamic_partition(params_1d, self.part, sum(self.n_tensors))
# 		model_index = 0
# 		for i, (shape, param) in enumerate(zip(self.shapes, params)):
# 			if i >= sum(self.n_tensors[:(model_index+1)]):
# 				model_index += 1
# 			self.models[model_index].trainable_variables[i-sum(self.n_tensors[:model_index])].assign(tf.reshape(param, shape))

	
# 	# need to be overloaded by subclasses
# 	@tf.function
# 	def loss_function(self, tape):
# 		tape.watch([t_collocation, t_data, u_data])
# 		u_c, c_c = forward(t_collocation)
# 		u_d, c_d = forward(t_data)
# 		loss_c = get_residue_loss(tape, t_collocation, u_c, c_c)
# 		loss_dr = get_residue_loss(tape, t_data, u_d, c_d)
# 		loss_dv = loss_func(u_d, u_data)
# 		loss = w_c*loss_c + w_dr*loss_dr + w_dv*loss_dv
# 		return loss, loss_c, loss_dr, loss_dv

# 	@tf.function
# 	def target(self, params_1d):
# 		tape = tf.GradientTape(persistent = True)
# 		tape.__enter__()
# 		self.assign_new_model_parameters(params_1d)
# 		loss, *losses = self.loss_function(tape)
# 		tape.__exit__(None, None, None)

# 		grads = tape.gradient(loss, self._variables)
# 		grads = tf.dynamic_stitch(self.idx, grads)
# 		self.iter.assign_add(1)
# 		del tape
# 		return loss, grads

# 	def optimize(self, save_path, model_names, loss_names, print_loss = True, **kwargs):
# 		init_params = tf.dynamic_stitch(self.idx, self._variables)
# 		results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function = self.target, 
# 									initial_position = init_params, **kwargs)
# 		tf.print("Convergence:", results.converged)
# 		self.assign_new_model_parameters(results.position)
	
# 		if print_loss:
# 			for model, name in zip(self.models, model_names):
# 				model.save("models/{}/{}".format(save_path, name))
# 			tape = tf.GradientTape(persistent = True)
# 			tape.__enter__()
# 			loss, *losses = self.loss_function(tape)
# 			tape.__exit__(None, None, None)
# 			del tape

# 			print("{}, Iter: {}, Loss: {:.4e}, ".format(get_time(), self.iter.numpy(), loss) + ": {:.4e}, ".join(loss_names + [" "]).format(*losses))

