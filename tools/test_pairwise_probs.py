#!/usr/bin/env python

import _init_paths
import caffe
import argparse
import time, os, sys
import json
import cv2
import cPickle as cp
import numpy as np
import math

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser()
	# image_paths file format: [ path of image_i for image_i in images ]
	# the order of images in image_paths should be the same with obj_dets_file
	parser.add_argument('--image_paths', dest='image_paths', help='file containing test dataset',
						default='', type=str)
	parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
						default=0, type=int)
	parser.add_argument('--def', dest='prototxt',
						help='prototxt file defining the unary network',
						default=None, type=str)
	parser.add_argument('--net', dest='caffemodel',
						help='unary model to test',
						default=None, type=str)
	parser.add_argument('--def_2', dest='prototxt_pairwise',
						help='prototxt file defining the pairwise network',
						default=None, type=str)
	parser.add_argument('--net_2', dest='caffemodel_pairwise',
						help='pairwise model to test',
						default=None, type=str)
	parser.add_argument('--num_dets', dest='max_det',
						help='max number of detections per image',
						default=100, type=int)
	# obj_dets_file format: [ obj_dets of image_i for image_i in images ]
	# 	obj_dets: numpy.array of size: num_instance x 5
	# 		instance: [x1, y1, x2, y2, prob, label]
	parser.add_argument('--obj_dets_file', dest='obj_dets_file', 
						help='file containing object detections',
						default=None, type=str)
	# type 0: im only
	# type 1: pos only
	# type 2: im + pos
	# type 3: im + pos + qa + qb
	parser.add_argument('--input_type', dest='type',
						help='type of input sets',
						default=3, type=int)

	parser.add_argument('--ncls', dest='num_class', help='number of object classes', default=101, type=int)

	parser.add_argument('--out', dest='out', help='name of output file', default='', type=str)
	parser.add_argument('--out_u', dest='out_unary', help='name of output unary file', default='', type=str)
	parser.add_argument('--out_p', dest='out_pair', help='name of output pairwise file', default='', type=str)

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()
	return args


def getPred(pred, max_num_det):
	if pred.shape[0] == 0:
		return pred
	inds = np.argsort(pred[:, 4])
	inds = inds[::-1]
	if len(inds) > max_num_det:
		inds = inds[:max_num_det]			
	return pred[inds, :]


def getUnionBBox(aBB, bBB, ih, iw):
	margin = 10
	return [max(0, min(aBB[0], bBB[0]) - margin), \
		max(0, min(aBB[1], bBB[1]) - margin), \
		min(iw, max(aBB[2], bBB[2]) + margin), \
		min(ih, max(aBB[3], bBB[3]) + margin)]

def getAppr(im, bb):
	subim = im[bb[1] : bb[3], bb[0] : bb[2], :]
	subim = cv2.resize(subim, None, None, 224.0 / subim.shape[1], 224.0 / subim.shape[0], interpolation=cv2.INTER_LINEAR)
	pixel_means = np.array([[[103.939, 116.779, 123.68]]])
	subim -= pixel_means
	subim = subim.transpose((2, 0, 1))
	return subim	

def getDualMask(ih, iw, bb):
	rh = 32.0 / ih
	rw = 32.0 / iw	
	x1 = max(0, int(math.floor(bb[0] * rw)))
	x2 = min(32, int(math.ceil(bb[2] * rw)))
	y1 = max(0, int(math.floor(bb[1] * rh)))
	y2 = min(32, int(math.ceil(bb[3] * rh)))
	mask = np.zeros((32, 32))
	mask[y1 : y2, x1 : x2] = 1
	assert(mask.sum() == (y2 - y1) * (x2 - x1))
	return mask

def forward_batch(net, ims, poses, qas, qbs, args):
	forward_args = {}
	if args.type != 1:
		net.blobs["im"].reshape(*(ims.shape))
		forward_args["im"] = ims.astype(np.float32, copy=False)
	if args.type != 0:
		net.blobs["posdata"].reshape(*(poses.shape))
		forward_args["posdata"] = poses.astype(np.float32, copy=False)
	if args.type == 3:		
		net.blobs["qa"].reshape(*(qas.shape))
		forward_args["qa"] = qas.astype(np.float32, copy=False)
		net.blobs["qb"].reshape(*(qbs.shape))
		forward_args["qb"] = qbs.astype(np.float32, copy=False)	
	net_out = net.forward(**forward_args)
	itr_pred = net_out["pred"].copy()
	return itr_pred

def concat_mask(sub_mask,obj_mask):
	return np.concatenate((sub_mask,obj_mask),axis=1)

def concat_im(im1,im2):
	im1 = cv2.resize(im1, (0,0), fx=0.5, fy=1)
	im2 = cv2.resize(im2, (0,0), fx=0.5, fy=1)
	return np.concatenate((im1,im2),axis=1)

def test_net(net, net_2, image_paths, args):
	f = open(args.obj_dets_file, "r")
	all_dets = cp.load(f)
	f.close()
	num_img = len(image_paths)
	num_class = args.num_class
	thresh = 0.05
	max_num_det = args.max_det
	batch_size = 1
	pred = []
	pred_bboxes = []
	unary_indices = []
	pair_indices = []
	for i in xrange(num_img):
		im = cv2.imread(image_paths[i]).astype(np.float32, copy=False)
		ih = im.shape[0]
		iw = im.shape[1]
			
		dets = getPred(all_dets[i], max_num_det)
		num_dets = dets.shape[0]
		pred.append([])
		pred_bboxes.append([])
		
		rels = [] 
		relIdx = 0
		pair_indices.append([])
		unary_indices.append([])
		for subIdx in xrange(num_dets):
			ims = []
			poses = []
			qas = []
			qbs = []
			for objIdx in xrange(num_dets):
				if subIdx != objIdx: 
					sub = dets[subIdx, 0: 4]
					obj = dets[objIdx, 0: 4]
					rBB = getUnionBBox(sub, obj, ih, iw)
					rAppr = getAppr(im, rBB)	
					rMask = np.array([getDualMask(ih, iw, sub), getDualMask(ih, iw, obj)])
					ims.append(rAppr)
					poses.append(rMask)
					qa = np.zeros(num_class - 1)
					aLabel = dets[subIdx, 5] 
					qa[aLabel - 1] = 1
					qb = np.zeros(num_class - 1)
					bLabel = dets[objIdx, 5]
					qb[bLabel - 1] = 1
					qas.append(qa)
					qbs.append(qb)

					rels.append([aLabel, bLabel, sub, obj, rBB])
					unary_indices[i].append([relIdx])
					relIdx += 1
			if len(ims) == 0:
				break
			ims = np.array(ims)
			poses = np.array(poses)
			qas = np.array(qas)
			qbs = np.array(qbs)
			_cursor = 0
			itr_pred = None
			num_ins = ims.shape[0]
			while _cursor < num_ins:
				_end_batch = min(_cursor + batch_size, num_ins)
				itr_pred_batch = forward_batch(net, ims[_cursor : _end_batch] if ims.shape[0] > 0 else None, poses[_cursor : _end_batch] if poses.shape[0] > 0 else None, qas[_cursor : _end_batch] if qas.shape[0] > 0 else None, qbs[_cursor : _end_batch] if qbs.shape[0] > 0 else None, args)
				if itr_pred is None:
					itr_pred = itr_pred_batch
				else:
					itr_pred = np.vstack((itr_pred, itr_pred_batch))
				_cursor = _end_batch

			
			cur = 0
			for objIdx in xrange(num_dets):	
				if subIdx != objIdx:
					sub = dets[subIdx, 0: 4]
					obj = dets[objIdx, 0: 4]
					for j in xrange(itr_pred.shape[1]):
						if itr_pred[cur, j] < thresh: 
							continue
						pred[i].append([itr_pred[cur, j], dets[subIdx, 4], dets[objIdx, 4], dets[subIdx, 5], j, dets[objIdx, 5]])
						pred_bboxes[i].append([sub, obj])						
					cur += 1
			assert(cur == itr_pred.shape[0])
		pred[i] = np.array(pred[i])
		pred_bboxes[i] = np.array(pred_bboxes[i])
		unary_indices[i] = np.array(unary_indices[i])	
		qas = []
		qbs = []
		poses = []
		ims = []
		for p,rel1 in enumerate(rels):
			aLabel = rel1[0]
			bLabel = rel1[1]
			aBBox = rel1[2]
			bBBox = rel1[3]
			rBBox = rel1[4]
			for q,rel2 in enumerate(rels):
				aLabel_2 = rel2[0]
				bLabel_2 = rel2[1]
				aBBox_2 = rel2[2]
				bBBox_2 = rel2[3]
				rBBox_2 = rel2[4]
				 
				if (p != q) and ((aLabel == aLabel_2) or (bLabel == bLabel_2)):
					
					pair_indices[i].append([p,q])	
					qa = np.zeros(num_class -1 )
					qa[aLabel - 1] = 1
					qa_2 = np.zeros(num_class - 1)
					qa_2[aLabel_2 - 1] = 1
					qa = np.concatenate([qa, qa_2])
					qas.append(qa)
			
					qb = np.zeros(num_class -1 )
					qb[bLabel - 1] = 1
					qb_2 = np.zeros(num_class -1)
					qb_2[bLabel_2 - 1] = 1
					qb = np.concatenate([qb, qb_2])
					qbs.append(qb)

					###Appearance and mask features
					ims.append(concat_im(getAppr(im, rBBox), getAppr(im, rBBox_2)))
					mask1 = concat_mask(getDualMask(ih, iw, aBBox ), getDualMask(ih, iw, bBBox)) 
					mask2 = concat_mask(getDualMask(ih, iw, aBBox_2), getDualMask(ih, iw, bBBox_2))
					poses.append([mask1, mask2])

		qas =  np.array(qas)
		qbs = np.array(qbs)
		ims = np.array(ims)
		poses = np.array(poses)
		_cursor = 0
		itr_pred_pair = []
		num_ins = ims.shape[0]
		while _cursor < num_ins:
			_end_batch = min(_cursor + batch_size, num_ins)
			itr_pred_pair_batch = forward_batch(net_2, ims[_cursor : _end_batch] if ims.shape[0] > 0 else None, poses[_cursor : _end_batch] if poses.shape[0] > 0 else None, qas[_cursor : _end_batch] if qas.shape[0] > 0 else None, qbs[_cursor : _end_batch] if qbs.shape[0] > 0 else None, args)
			#if itr_pred_pair is None:
			itr_pred_pair.append(itr_pred_pair_batch)
			#else:
			#	itr_pred_pair = np.vstack((itr_pred_pair, itr_pred_pair_batch))
			_cursor = _end_batch
		
		pair_indices[i] = np.array(pair_indices[i])	
		itr_pred_pair = np.array(itr_pred_pair)
								
	
	print "writing file.."
	f = open(args.out, "wb")
	cp.dump([pred, pred_bboxes], f, cp.HIGHEST_PROTOCOL)			
	f.close()

	print "writing pairwise.."
	f = open(args.out_unary,"wb")
	cp.dump([unary_indices, itr_pred], f, cp.HIGHEST_PROTOCOL)
	f.close()


	print "writing pairwise.."
	f = open(args.out_pair,"wb")
	cp.dump([pair_indices, itr_pred_pair], f, cp.HIGHEST_PROTOCOL)
	f.close()

if __name__ == '__main__':
	args = parse_args()

	print('Called with args:')
	print(args)

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)
	net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
	net_2 = caffe.Net(args.prototxt_pairwise, args.caffemodel_pairwise, caffe.TEST)

	#test_image_paths = json.load(open(args.image_paths))
	test_image_paths = ["dataset/VRD/images/test/3845770407_1a8cd41230_b.jpg"]

	test_net(net, net_2, test_image_paths, args)
	
