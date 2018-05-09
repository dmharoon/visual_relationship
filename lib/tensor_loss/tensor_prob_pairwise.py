import caffe
import numpy as np 

class TensorProbPairwise(caffe.Layer):
	"""docstring for Layer"""

	def setup(self, bottom, top):
		self._R = 10
		if len(bottom) != 1:
		  	raise Exception("Need only one input")
		if bottom[0].count != (self._R * 2 * 70):
			raise Exception("Need {} dimensional input".format((self._R * 140)))

		self._pred1 = bottom[0].data[0, :(self._R * 70)].reshape((self._R, 70))
		self._pred2 = bottom[0].data[0, (self._R * 70):].reshape((self._R, 70))
		self._Z = self._compute_Z()

	def _compute_Z(self):
		_sum_p1=np.zeros(self._R)
		_sum_p2=np.zeros(self._R)
		_Z=0.0
		for i in range(self._R):
			_sum_p1[i] = np.sum(np.exp(self._pred1[i]))
			_sum_p2[i] = np.sum(np.exp(self._pred2[i]))
			
			_Z += (_sum_p1[i] * _sum_p2[i]) 
		print "partition function _Z: ",_Z
		return _Z

	def reshape(self,bottom,top):
		self.softmax = np.zeros((70,70),dtype=np.float32)
		top[0].reshape(70,70)

	def forward(self,bottom,top):
		self._probs = np.zeros((self._R, 70, 70), dtype = np.float32)
		for i in range(self._R):
			self._probs[i] = np.outer(np.exp(self._pred1[i]), np.exp(self._pred2[i]))
		self._softmax = np.sum(self._probs,axis=0) / self._Z
		top[0].data[...] = self._softmax
		
	def backward(self, top, propagate_down, bottom):
		pass	
