import os.path as osp
import mat4py as mp
import cv2
import numpy as np
import math
import json

class DataLoader:
	def __init__(self, datasetRoot, split):
		self._root = datasetRoot
		self._split = split
		self._loadLabels()
		self._loadAnnotation(split)

	def _loadLabels(self):
		mat = mp.loadmat(osp.join(self._root, "predicate.mat"))
		self._relList = mat["predicate"]
		self._numRelClass = len(self._relList)
		self._relMapping = {}
		for i in xrange(len(self._relList)):
			self._relMapping[self._relList[i]] = i	
		mat = mp.loadmat(osp.join(self._root, "objectListN.mat"))
		self._objList = mat["objectListN"]
		self._numObjClass = len(self._objList) + 1
		self._objMapping = {}
		self._objMapping["__BG"] = 0
		for i in xrange(len(self._objList)):
			self._objMapping[self._objList[i]] = i + 1	

	def _loadAnnotation(self, split):
		mat = mp.loadmat(osp.join(self._root, "annotation_" + split + ".mat"))
		self._annotations = mat["annotation_" + split]

	def _getNumImgs(self):
		return len(self._annotations)

	def _getImPath(self, idx):
		return osp.join(self._root, "images", self._split, self._annotations[idx]["filename"])
	
	def _getNumRel(self):
		numRels = 0
		n = self._getNumImgs()
		for i in xrange(n):
			rels = self._getRels(i)
			numRels += len(rels)	
		return numRels

	def _getRels(self, idx):
		if "relationship" in self._annotations[idx]:
			rels = self._annotations[idx]["relationship"] 
			if isinstance(rels, dict):
				rels = [rels]	
			return rels
		else:
			return []

	def _outputDB(self, type, data):
		json.dump(data, open(type + self._split + ".json", "w"))

	def _bboxTransform(self, bbox, ih, iw): #[x1, y1, x2, y2]
		return [max(bbox[2], 0), max(bbox[0], 0), min(bbox[3] + 1, iw), min(bbox[1] + 1, ih)]
	
	def _getRelLabel(self, predicate):
		if not (predicate in self._relMapping):
			return -1
		return self._relMapping[predicate]

	def _getObjLabel(self, predicate):
		if not (predicate in self._objMapping):
			return -1
		return self._objMapping[predicate]

	def _getUnionBBox(self, aBB, bBB, ih, iw, margin = 10):
		return [max(0, min(aBB[0], bBB[0]) - margin), \
			max(0, min(aBB[1], bBB[1]) - margin), \
			min(iw, max(aBB[2], bBB[2]) + margin), \
			min(ih, max(aBB[3], bBB[3]) + margin)]

	def _getRelSamplesSingle(self):
		n = self._getNumImgs()
		self._sampleIdx = 0
		samples = []
		for i in xrange(n):
			rels = self._getRels(i)
			if len(rels) == 0:
				continue
			path = self._getImPath(i)
			im = cv2.imread(path)
			ih = im.shape[0]	
			iw = im.shape[1]
			for p in range(len(rels)):
				phrase = rels[p]["phrase"]
				rLabel = self._getRelLabel(phrase[1])
				aLabel = self._getObjLabel(phrase[0])
				bLabel = self._getObjLabel(phrase[2])
				aBBox = self._bboxTransform(rels[p]["subBox"], ih, iw)
				bBBox = self._bboxTransform(rels[p]["objBox"], ih, iw)
				rBBox = self._getUnionBBox(aBBox, bBBox, ih, iw)

				for q in range(len(rels)):
					phrase_2 = rels[q]["phrase"]
					rLabel_2 = self._getRelLabel(phrase_2[1])
					rLabel_12 = rLabel * len(self._relList) + rLabel_2
					aLabel_2 = self._getObjLabel(phrase_2[0])
					bLabel_2 = self._getObjLabel(phrase_2[2])
					aBBox_2 = self._bboxTransform(rels[q]["subBox"], ih, iw)
					bBBox_2 = self._bboxTransform(rels[q]["objBox"], ih, iw)
					rBBox_2 = self._getUnionBBox(aBBox_2, bBBox_2, ih, iw)

					if (p != q) and ((aLabel == aLabel_2) or (bLabel == bLabel_2)):

						samples.append({"p":p, "imPath": path, "rLabel_12": rLabel_12, "aLabel": aLabel, "bLabel": bLabel, "rBBox": rBBox, "aBBox": aBBox, "bBBox": bBBox,
							"q":q, "aLabel_2": aLabel_2, "bLabel_2": bLabel_2, "rBBox_2": rBBox_2, "aBBox_2": aBBox_2, "bBBox_2": bBBox_2})
						self._sampleIdx += 1	
						if self._sampleIdx % 100 == 0:
							print self._sampleIdx

		self._outputDB("rel", samples)	
		
if __name__ == "__main__":
	loader = DataLoader("/home/haroon/sem2/Machine Learning/drnet_cvpr2017_obj2vec/dataset/VRD", "train")
	#loader = DataLoader("/home/haroon/sem2/Machine Learning/drnet_cvpr2017_obj2vec/dataset/VRD", "test")
	loader._getRelSamplesSingle()
2