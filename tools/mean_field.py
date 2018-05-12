#!/usr/bin/env python
import numpy as np
import sys
import argparse
import cPickle as cp
import math


#import piecewisecrf.tests.mean_field_test as test

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--unary', dest='unary', help='file containing unary potentials',
				default='unary',type=str)
    parser.add_argument('--pairwise', dest='pairwise', help='file containing pairwise potentials',
				default='pairwise',type=str)
    parser.add_argument('--marginals', dest='marginals', help='output file for marginals',
				default='marginals',type=str)
    parser.add_argument('--out', dest='out', help='output file for predictions',
				default='out',type=str)

    if len(sys.argv) == 1:
        parser.print_help()
	sys.exit(1)
    args = parser.parse_args()
    return args


def _exp_norm(marginals):
    '''

    Normalize marginals

    Parameters
    ----------
    marginals : numpy array with dimensions [height, number_of_classes]
        Input array

    Returns
    -------
    marginals : numpy array
        Normalized marginals

    '''
    # apply exponential function
    marginal_max = np.amax(marginals, axis=1)
    marginal_max = np.repeat(marginal_max, marginals.shape[1])
    marginal_max = np.reshape(marginal_max, marginals.shape)
    marginals = marginals - marginal_max
    marginals = np.exp(marginals)

    # normalize each marginal
    marginal_sum = np.sum(marginals, axis=1)
    marginal_sum = np.repeat(marginal_sum, marginals.shape[1])
    marginal_sum = np.reshape(marginal_sum, marginals.shape)

    marginals = marginals / marginal_sum

    return marginals


def mean_field(unary, pairwise, zipped_indices, number_of_iterations=3):
    '''

    Computes given number of mean field iterations

    Parameters
    ----------
    unary : numpy array with dimensions [height, number_of_classes]
        Unary scores (outputs from network). This method will negate them in order
        to get potentials

    pairwise: list of tuples (pairwise_scores, zipped_indices, decoding)
        Pairwise scores and appropriate methods to use them:
            pairwise_scores: pairwise net outputs (not yet potentials)
		numpy array with dimensions [number_of_zipped_indices, number_of_classes, number_of_classes]
            zipped_indices: (first_, secod_index) list
            decoding: mapping index -> (first_, second_class)

    number_of_iterations: int
        Number of mean field iterations

    Returns
    -------
    marginals : numpy array
        Normalized marginals


    '''

    print "unary shape:", unary.shape
    print "pairwise shape:", pairwise.shape
    number_of_classes = unary.shape[1]

    # multiple net outputs by -1 because they represent potentials
    unary = -1.0 * unary
    pairwise = -1.0 * pairwise

    # initialize marginals to unary potentials
    marginals = np.zeros(unary.shape)
    marginals = np.array( -1.0 * unary)
    marginals = marginals.astype(np.float)
    marginals = _exp_norm(marginals)

    print "marginals shape..",marginals.shape

    for it_num in range(number_of_iterations):
        # print("Mean-Field iteration #{}".format(it_num + 1))
        print "mean field iteration.. ", it_num
	tmp_marginals = np.zeros(marginals.shape)
        tmp_marginals = np.array( -1.0 * unary)
        tmp_marginals = tmp_marginals.astype(np.float)
        for i, (f, s) in enumerate(zipped_indices):
            #tmp_marginals[f, :] -= pairwise[i].dot(marginals[s, :])
            #tmp_marginals[s, :] -= marginals[f, :].dot(pairwise[i])
            tmp_marginals[f, :] -= np.sum(np.multiply(pairwise[i], marginals[s, :]), axis = 1)
            tmp_marginals[s, :] -= np.sum(np.multiply(marginals[f, :], pairwise[i].T), axis = 1)

        tmp_marginals = _exp_norm(tmp_marginals)
        marginals = np.array(1 * tmp_marginals)
        marginals = marginals.astype(np.float)

    return marginals

def threshold(marginals, rels, thresh):
	pred = []
	pred_bboxes = []
	print "marginals shape: ",marginals.shape
	for cur in xrange(marginals.shape[0]):
		rel = rels[cur]
		for rel_cls in xrange(marginals.shape[1]):
			if marginals[cur, rel_cls] < thresh: 
				continue
			pred.append([ rel[5], marginals[cur, rel_cls], rel[6], rel[0], rel_cls, rel[1] ])
			pred_bboxes.append([rel[2], rel[3]])
				
	pred = np.array(pred)
	pred_bboxes = np.array(pred_bboxes)
	return pred, pred_bboxes

if __name__ == '__main__':
	args = parse_args()
	print('Called with args:')
	print(args)
	
	f = open(args.unary, 'r')
	ind_u, pot_u = cp.load(f)
	f.close()
	f = open(args.pairwise, 'r')
	ind_p, pot_p = cp.load(f)
	f.close()
	
	mean_field_iters = 6
	marginals = []	
	for i in range(len(ind_u)):
		marginals.append([])
		pot_u[i] = np.squeeze(pot_u[i])
		marginals[i] = mean_field(pot_u[i], pot_p[i], ind_p[i], mean_field_iters)	
	
	#marginals = np.array(marginals)
	print "writing marginals to file.."
	f = open(args.marginals, "wb")
	cp.dump(marginals, f, cp.HIGHEST_PROTOCOL)			
	f.close()
	thresh = 0.005
	
	pred = []
	pred_bboxes = []
	for i in range(len(ind_u)):
		pred.append([])
		pred_bboxes.append([])
		pred[i], pred_bboxes[i] = threshold(marginals[i], ind_u[i], thresh)

	print "writing final predictions to file..."	
	f = open(args.out, "wb")
	cp.dump([pred, pred_bboxes], f, cp.HIGHEST_PROTOCOL)
	f.close()






