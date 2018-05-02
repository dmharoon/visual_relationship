import caffe
import numpy as np 

class TensorDecompLoss(caffe.Layer):
	"""docstring for Layer"""

	def setup(self, bottom, top):
		self._R = 10
		if len(bottom) != 2:
		  	raise Exception("Need two inputs")
		if bottom[0].count != (self._R * 2 * 70):
			raise Exception("Need {} dimensional input".format((self._R * 140)))
		if bottom[1].count != 2:
			print bottom[1].data[...]
			raise Exception("Need tuple of predicates as labels")

		self._pred1 = bottom[0].data[0, :(self._R * 70)].reshape((self._R, 70))
		self._pred2 = bottom[0].data[0, (self._R * 70):].reshape((self._R, 70))
		self._Z = self._compute_Z()
		self._label1, self._label2 = bottom[1].data[...]
		self._label1, self._label2 = int(self._label1), int(self._label2)

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
		top[0].reshape(1)

	def forward(self,bottom,top):
		self._probs = np.zeros((self._R, 70, 70), dtype = np.float32)
		for i in range(self._R):
			self._probs[i] = np.outer(np.exp(self._pred1[i]), np.exp(self._pred2[i]))
		self._softmax = np.sum(self._probs,axis=0) / self._Z
		print self._label1, self._label2
		self._Error = -np.log(self._softmax[self._label1, self._label2])
		top[0].data[...] = self._Error
		print "error is : ", self._Error
		
	def backward(self, top, propagate_down, bottom):
		print "backward ...."
		diff1 = np.zeros_like(self._pred1,dtype=np.float32)
		diff2 = np.zeros_like(self._pred2,dtype=np.float32)
		tmp1 = np.zeros_like(self._pred1,dtype=np.float32)
		tmp2 = np.zeros_like(self._pred2,dtype=np.float32)
		for i in range(self._R):
			tmp1[i,self._label1] = (self._probs[i, self._label1, self._label2]) / (self._softmax[self._label1, self._label2])
			tmp2[i,self._label2] = tmp1[i,self._label1]
			diff1[i] = (np.sum(self._probs[i],axis=(1)) - tmp1[i]) / self._Z
			diff2[i] = (np.sum(self._probs[i],axis=(0)) - tmp2[i]) / self._Z
		bottom[0].diff[...] = np.concatenate([diff1.reshape((self._R * 70)),diff2.reshape((self._R * 70))])



		