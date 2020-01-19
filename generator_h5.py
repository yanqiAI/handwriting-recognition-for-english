# coding:utf-8

from pathlib import Path
from imutils import paths
import random
import numpy as np
import h5py
import cv2
import os
from keras.utils import Sequence, to_categorical
from IPython.core.debugger import Tracer


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self, batchsize):
        r_n=[]
        if(self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize

        return r_n
    

class ImageGenerator(Sequence):
	def __init__(self, h5_dir, batch_size=None, maxlabellength=None, image_h=None, image_w=None):
		self.h5_dir = h5_dir
		self.batch_size = batch_size
		self.maxlabellength = maxlabellength
		self.image_h = image_h
		self.image_w = image_w
		
		# load h5 file
		dataset = h5py.File(self.h5_dir, 'r')
		self.images = dataset['images']  # np.uint8
		self.labels = dataset['labels']  # np.int32   start label is 0
		
		self.image_num = len(self.images)
		
		if self.image_num == 0:
			raise ValueError("image dir '{}' does not include any image".format(image_dir))
	
	def __len__(self):
		return int(self.image_num // self.batch_size)

	def __getitem__(self, index):
		batch_size = self.batch_size
		maxlabellength = self.maxlabellength
		image_h = self.image_h
		image_w = self.image_w
		
		r_n = random_uniform_num(self.image_num)

		while True:
			batch_idx = r_n.get(batch_size)
			images = [self.images[p] for p in batch_idx]
			labels = [self.labels[p] for p in batch_idx]
			
			images = np.array(images, 'f') / 255.0 - 0.5
			labels = np.array(labels)
			
			labels_pos = (labels != 100) * 1
			factor = np.ones(labels.shape[1], dtype=np.int32)
			label_length = np.dot(labels_pos, factor)[:, np.newaxis]
			
			input_length = np.array([image_w // 8] * batch_size)[:, np.newaxis]  # 最后解码的序列长度
			
			inputs = {'the_input': images,
			          'the_labels': labels,
			          'input_length': input_length,
			          'label_length': label_length,
			          }
			outputs = {'ctc': np.zeros([batch_size])}
			return inputs, outputs

if __name__ == '__main__':
	# test code
	train_data = r'E:\datasets\ocr_dataset\words\train_words.h5'
	generator = ImageGenerator(train_data, batch_size=4, maxlabellength=50, image_h=64, image_w=400)
