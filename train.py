import caffe
import argparse
import time, os, sys
import json
import cv2
import cPickle as cp
import numpy as np
import math
import os.path as osp
import sys


def _train():
	max_iter = 10000

	display = 100

	test_iter = 100

	test_interval = 500


	train_loss = np.zeros(int(math.ceil(max_iter * 1.0 / display)))
	test_loss = np.zeros(int(math.ceil(max_iter * 1.0 / test_interval)))
	test_acc = np.zeros(int(math.ceil(max_iter * 1.0 / test_interval)))


	solver.step(1)


	_train_loss = 0
	_test_loss = 0
	_accuracy = 0


	for it in range(max_iter):
	    
	    solver.step(1)
	    
	    _train_loss += solver.net.blobs['loss'].data
	    if it % display == 0:
	    
	        train_loss[int(it / display)] = _train_loss / display
	        _train_loss = 0

	    
	    if it % test_interval == 0:
	        for test_it in range(test_iter):
	    
	            solver.test_nets[0].forward()
	    
	            _test_loss += solver.test_nets[0].blobs['loss'].data
	    
	    
	    
	        test_loss[it / test_interval] = _test_loss / test_iter
	    
	   
	        print _test_loss
	        _test_loss = 0
	   

	   



def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.getcwd()
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

caffe.set_mode_cpu()
#caffe.set_device(0)
solver = caffe.get_solver('prototxts/solver.prototxt')
solver.net.copy_from('snapshots/VGG_ILSVRC_16_layers.caffemodel')
#solver.net.forward()  # fprop
#solver.net.backward()  # bprop
#solver.step(50)
#solver.solve()
_train()
