import jax
import jax.numpy as jnp
from jax import random

def jacfwd_fn(model):
	def jac_(params, inputs):
		return jax.jit(jax.vmap(jax.jacfwd(model, 1), in_axes = (None, 0)))(params, inputs)
	return jac_

def jacrev_fn(model):
	def jac_(params, inputs):
		return jax.jit(jax.vmap(jax.jacrev(model, 1), in_axes = (None, 0)))(params, inputs)
	return jac_

def hessian_fn(model):
	def hes_(params, inputs):
		return jax.jit(jax.vmap(jax.hessian(model, 1), in_axes = (None, 0)))(params, inputs)
	return hes_

# def jacobian(model, in_dim, out_dim):
# 	if in_dim >= out_dim:
# 		jac_ = jacrev(model)
# 	else:
# 		jac_ = jacfwd(model)
	
# 	def jac_fn(params, inputs):
# 		jac_matrix = jac_(params, inputs)
# 		outputs = []


def siren_layer_params(key, scale, m, n, dtype = jnp.float32):
	w_key, b_key = random.split(key)
	return random.uniform(w_key, (m, n), dtype, minval = -scale, maxval = scale), jnp.zeros((n, ), dtype)

def tanh_layer_params(key, m, n, dtype = jnp.float32):
	w_key, b_key = random.split(key)
	w_init_fn = jax.nn.initializers.glorot_normal()
	return w_init_fn(w_key, (m, n), dtype), jnp.zeros((n, ), dtype)

def init_siren_params(key, layers, c0, w0, w1, dtype = jnp.float32):
	keys = random.split(key, len(layers))
	weights = [w0] + [1.0]*(len(layers)-3) + [w1]
	return [siren_layer_params(k, w*jnp.sqrt(c0/m), m, n) for k, w, m, n in zip(keys, weights, layers[:-1], layers[1:])]

def init_tanh_params(key, layers):
	keys = random.split(key, len(layers))
	return [tanh_layer_params(k, m, n) for (k, m, n) in zip(keys, layers[:-1], layers[1:])]

@jax.jit
def mse(pred, true):
	return jnp.mean(jnp.square(pred.reshape((-1, 1)) - true.reshape((-1, 1))))

@jax.jit
def mae(pred, true):
	return jnp.mean(jnp.abs(pred.reshape((-1, 1)) - true.reshape((-1, 1))))

@jax.jit
def l2_regularization(params, lambda_0):
	res = 0
	for p in params:
		res += jnp.sum(jnp.square(p[0]))
	return res*lambda_0

@jax.jit
def l1_regularization(params, lambda_0):
	"""
		params[i]: (w, b)
	"""
	res = 0
	for p in params:
		res += jnp.sum(jnp.abs(p[0]))
	return res*lambda_0	


class Batch_Generator:
	def __init__(self, key, dataset, batch_size):
		self.key = key
		self.dataset = dataset
		self.batch_size = batch_size
		self.index = jnp.arange(dataset[0].shape[0])
		self.pointer = 0
		self._shuffle()
		
	def _shuffle(self):
		key, subkey = random.split(self.key)
		self.index = random.permutation(subkey, jnp.arange(self.dataset[0].shape[0]))
		self.key = key
		
	def __iter__(self):
		return self
	
	def __next__(self):
		if self.pointer >= len(self.index):
			self._shuffle()
			self.pointer = 0
		self.pointer += self.batch_size
		index_ = self.index[self.pointer-self.batch_size:self.pointer]
		return [d[index_, :] for d in self.dataset]