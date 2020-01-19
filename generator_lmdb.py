
# coding:utf-8
import os
import numpy as np
import lmdb
import re
import six
from PIL import Image
from keras.utils import Sequence
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

class LmdbDataset(Sequence):
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.dict = {}
        for i, char in enumerate(list(self.args.character)):
            self.dict[char] = i

        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            self.filtered_index_list = [index+1 for index in range(self.nSamples)]
            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples


    def __getitem__(self, index):
        x = np.zeros((self.args.batch_size, self.args.image_h, self.args.image_w, 1), dtype=np.float32)
        labels = np.ones([self.args.batch_size, self.args.maxlabellength]) * 100
        input_length = np.zeros([self.args.batch_size, 1])
        label_length = np.zeros([self.args.batch_size, 1])

        r_n = random_uniform_num(self.nSamples)
        while True:
            batch_idx = r_n.get(self.args.batch_size)
            for i, j in enumerate(batch_idx):
                image, label = self._generate_X_y(j)

                x[i] = np.expand_dims(image, axis=2)
                label_id = [self.dict[c] for c in list(label)]
                labels[i, :len(label)] = [int(k) for k in label_id]
                if(len(label) <= 0):
                    print("len < 0", j)
                input_length[i] = self.args.image_w // 8
                label_length[i] = len(label)
                
            inputs = {
                'the_input': x,
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                }
            outputs = {'ctc': np.zeros([self.args.batch_size])}

            return inputs, outputs


    def _generate_X_y(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            if self.args.rgb:
                img = Image.open(buf).convert('RGB')  # for color image
            else:
                img = Image.open(buf).convert('L')

        return (np.array(img) / 255.0 - 0.5, label)
