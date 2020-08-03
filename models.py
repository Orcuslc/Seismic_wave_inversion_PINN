import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

def siren_model(layers, c, w0, lambda_1, device = "/device:gpu:0", adaptive = None):
	class scaled_dense(keras.layers.Layer):
		def __init__(self, units, input_dim, c, w0, lambda_1):
			super(scaled_dense, self).__init__()
			self.lambda_1 = lambda_1
			w_init = tf.random_uniform_initializer(-np.sqrt(c/input_dim), np.sqrt(c/input_dim))
			self.w = tf.Variable(initial_value = w_init(shape = (input_dim, units), dtype = "float32"), trainable = True)
			b_init = tf.zeros_initializer()
			self.b = tf.Variable(initial_value = b_init(shape=(units,), dtype="float32"), trainable=True)
			self.w0 = tf.Variable(w0, dtype = "float32", trainable = True)
			
		def call(self, inputs):
			self.add_loss(self.lambda_1*tf.reduce_sum(tf.abs(self.w)))
			return tf.sin(tf.matmul(inputs, self.w)*self.w0 + self.b)
	
	with tf.device(device):
		model = keras.models.Sequential()
		if adaptive == "first" or adaptive == "all":
			model.add(scaled_dense(layers[1], layers[0], c, w0, lambda_1))
		else:
			model.add(keras.layers.Dense(layers[1], input_shape = (layers[0], ), activation = K.sin,
										kernel_initializer = keras.initializers.RandomUniform(-w0*np.sqrt(c/layers[0]), w0*np.sqrt(c/layers[0])),
										kernel_regularizer = keras.regularizers.l1(lambda_1), use_bias = True))
		for i in range(1, len(layers)-2):
			if adaptive == "all":
				model.add(scaled_dense(layers[i+1], layers[i], c, 1.0, lambda_1))
			else:
				model.add(keras.layers.Dense(layers[i+1], input_shape = (layers[i], ), activation = K.sin,
									kernel_initializer = keras.initializers.RandomUniform(-np.sqrt(c/layers[i]), np.sqrt(c/layers[i])),
									kernel_regularizer = keras.regularizers.l1(lambda_1), use_bias = True))
		model.add(keras.layers.Dense(layers[-1], use_bias = True))
	return model

def tanh_model(layers, device = "/device:gpu:0", adaptive = None, bias = None):
	class adaptive_dense(keras.layers.Layer):
		def __init__(self, units, input_dim):
			super(adaptive_dense, self).__init__()
			w_init = keras.initializers.GlorotUniform()
			self.w = tf.Variable(initial_value = w_init(shape = (input_dim, units), dtype = "float32"), trainable = True)
			b_init = tf.zeros_initializer()
			self.b = tf.Variable(initial_value = b_init(shape=(units,), dtype="float32"), trainable=True)
			self.a = tf.Variable(1.0, dtype = "float32", trainable = True)
			
		def call(self, inputs):
			return tf.tanh(self.a*(tf.matmul(inputs, self.w) + self.b))
	
	with tf.device(device):
		model = keras.models.Sequential()
		if adaptive == "first" or adaptive == "all":
			model.add(adaptive_dense(layers[1], layers[0]))
		else:
			model.add(keras.layers.Dense(layers[1], input_shape = (layers[0], ), activation = "tanh"))
		for i in range(1, len(layers)-2):
			if adaptive == "all":
				model.add(adaptive_dense(layers[i+1], layers[i]))
			else:
				model.add(keras.layers.Dense(layers[i+1], input_shape = (layers[i], ), activation = "tanh"))
		if bias:
			model.add(keras.layers.Dense(layers[-1], bias_initializer = keras.initializers.Constant(bias)))
		else:
			model.add(keras.layers.Dense(layers[-1]))
	return model

def swish_model(layers, device = "/device:gpu:0", adaptive = None, bias = None):
	class adaptive_dense(keras.layers.Layer):
		def __init__(self, units, input_dim):
			super(adaptive_dense, self).__init__()
			w_init = keras.initializers.GlorotUniform()
			self.w = tf.Variable(initial_value = w_init(shape = (input_dim, units), dtype = "float32"), trainable = True)
			b_init = tf.zeros_initializer()
			self.b = tf.Variable(initial_value = b_init(shape=(units,), dtype="float32"), trainable=True)
			self.a = tf.Variable(1.0, dtype = "float32", trainable = True)
			
		def call(self, inputs):
			return inputs*tf.sigmoid(self.a*(tf.matmul(inputs, self.w) + self.b))
	
	with tf.device(device):
		model = keras.models.Sequential()
		if adaptive == "first" or adaptive == "all":
			model.add(adaptive_dense(layers[1], layers[0]))
		else:
			model.add(keras.layers.Dense(layers[1], input_shape = (layers[0], ), activation = "tanh"))
		for i in range(1, len(layers)-2):
			if adaptive == "all":
				model.add(adaptive_dense(layers[i+1], layers[i]))
			else:
				model.add(keras.layers.Dense(layers[i+1], input_shape = (layers[i], ), activation = "tanh"))
		if bias:
			model.add(keras.layers.Dense(layers[-1], bias_initializer = keras.initializers.Constant(bias)))
		else:
			model.add(keras.layers.Dense(layers[-1]))
	return model


class rectangular_constant_model(keras.Model):
	def __init__(self, boundary, values, trainable, device = "/device:gpu:0"):
		"""
		boundary: the interfaces of the middle rectangular interval; in ascending order; len(boundary) = 2
		value_init: initial values of each variable; len(value_init) = 2; here we suppose the value in the first and third interval are the same
		"""
		super(rectangular_constant_model, self).__init__()
		self.boundary = boundary
# 		self.values = [tf.Variable(vi, dtype = tf.float32, trainable = True) for vi in value]
		with tf.device(device):
			self.values = [tf.Variable(xi, dtype = tf.float32, trainable = ti) for (xi, ti) in zip(values, trainable)]
		
	def call(self, inputs): 
		"""
		inputs: (x, z)
		outputs: c'(z)
		"""
		return tf.where(tf.math.logical_and(tf.greater_equal(inputs, self.boundary[0]), 
											tf.less_equal(inputs, self.boundary[1])),
						self.values[1],
						self.values[0])


class step_constant_model(rectangular_constant_model):
	def call(self, inputs):
		return tf.where(tf.greater_equal(inputs, self.boundary),
						self.values[1],
						self.values[0])



class constant_model(keras.Model):
	def __init__(self, values, trainable, device = "/device:gpu:0"):
		super(constant_model, self).__init__()
		with tf.device(device):
			self.values = [tf.Variable(xi, dtype = tf.float32, trainable = ti) for (xi, ti) in zip(values, trainable)]

	def call(self, inputs):
		return self.values