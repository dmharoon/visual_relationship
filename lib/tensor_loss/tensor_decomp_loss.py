import caffe
import numpy as np 

class TensorDecompLoss(caffe.Layer):
	"""docstring for Layer"""

	def setup(self, bottom, top):
		self._R = 10
		if len(bottom) != 2:
		  	raise Exception("Need two inputs")
		if bottom[0].count != (self._R * 270):
			raise Exception("Need {} dimensional input".format((self._R * 270)))
		
		self._sub = bottom[0].data[0,:(100*self._R)]
		self._obj = bottom[0].data[0,(100*self._R):(2*100*self._R)]
		self._pred = bottom[0].data[0,(2*100*self._R):]

		self.__sub = np.zeros((self._R,100))
		self.__obj = np.zeros((self._R,100))
		self.__pred = np.zeros((self._R,70))
		for i in range(self._R):
			self.__sub[i] = self._sub[(i*100):((i+1)*100)]
			self.__obj[i] = self._obj[(i*100):((i+1)*100)]
			self.__pred[i] = self._pred[(i*70):((i+1)*70)]
		self._label = np.zeros((100,100,70))
		self._label[np.unravel_index(int(bottom[1].data[...]),(100,100,70))]=1

	def _compute_Z(self):
		_sum_s=np.zeros(self._R)
		_sum_o=np.zeros(self._R)
		_sum_p=np.zeros(self._R)
		self._Z=0
		for i in range(self._R):
			_sum_s[i] = np.sum(np.exp(self.__sub[i]))
			_sum_o[i] = np.sum(np.exp(self.__obj[i]))
			_sum_p[i] = np.sum(np.exp(self.__pred[i]))
			self._Z += _sum_s[i] * _sum_o[i] * _sum_p[i] 
		return self._Z

	def reshape(self,bottom,top):
		self.probs = np.zeros((self._R,100,100,70),dtype=np.float32)
		self.softmax = np.zeros((100,100,70),dtype=np.float32)
		top[0].reshape(1)

	def forward(self,bottom,top):
		self._Z = self._compute_Z()
		for i in range(self._R):
			self.__sub[i] = np.exp(self._sub[(i*100):((i+1)*100)])
			self.__obj[i] = np.exp(self._obj[(i*100):((i+1)*100)])
			self.__pred[i] = np.exp(self._pred[(i*70):((i+1)*70)])
			self.probs[i] = np.outer(np.outer(self.__sub[i],self.__obj[i]), self.__pred[i]).reshape(100,100,70)
		self.probs_final = np.divide(np.sum(self.probs,axis=0),self._Z)
		self._Error = np.sum(self._label * np.log(self.probs_final))
		top[0].data[...] = self._Error
		print "error is : ", self._Error
		
	def backward(self, top, propagate_down, bottom):
		print "hi................"
		print "hi................"
		self._sub_diff = np.zeros(self._R * 100)
		self._obj_diff = np.zeros(self._R * 100)
		self._pred_diff = np.zeros(self._R * 70)
		self._Z = self._compute_Z()

		idx = np.unravel_index(int(bottom[1].data[...]),(100,100,70))
		_sum_s_diff = np.zeros_like(self.__sub,dtype=np.float32)
		_sum_o_diff = np.zeros_like(self.__obj,dtype=np.float32)
		_sum_p_diff = np.zeros_like(self.__pred,dtype=np.float32)
		for i in range(self._R):
			_sum_s_diff[i] = (np.sum(self.probs[i],axis=(1,2)) - (self.probs[i][idx] / self.probs_final[idx])) / self._Z 
			_sum_o_diff[i] = (np.sum(self.probs[i],axis=(0,2)) - (self.probs[i][idx] / self.probs_final[idx])) / self._Z 
			_sum_p_diff[i] = (np.sum(self.probs[i],axis=(0,1)) - (self.probs[i][idx] / self.probs_final[idx])) / self._Z

			self._sub_diff[(i*100):((i+1)*100)] = _sum_s_diff[i]
			self._obj_diff[(i*100):((i+1)*100)] = _sum_o_diff[i]
			self._pred_diff[(i*70):((i+1)*70)] = _sum_p_diff[i]

		self._diff = np.concatenate([self._sub_diff, self._obj_diff, self._pred_diff])
		print self._diff
		bottom[0].diff[...] = self._diff  



		