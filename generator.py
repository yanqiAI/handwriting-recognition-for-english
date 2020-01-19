from pathlib import Path
from imutils import paths
import random
import numpy as np
import cv2
import os
from keras.utils import Sequence, to_categorical
from IPython.core.debugger import Tracer


class ImageGenerator(Sequence):
    def __init__(self, image_dir, label_file, batch_size=None, maxlabellength=None, image_h=None, image_w=None):
        #self.image_paths = sorted(list(paths.list_images(image_dir)))
        #self.image_num = len(self.image_paths)
        
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.maxlabellength = maxlabellength
        self.image_h = image_h
        self.image_w = image_w

        with open(label_file, 'r', encoding='utf-8') as file:
            #self.labels_dict = {f.strip().split(' ')[0] : f.strip().split(' ')[-1] for f in file.readlines()} #only support single letter
            items = [item.strip() for item in file.readlines()]
            self.labels_dict = {f.split(' ')[0] : f.split(' ')[1:] for f in items}
            
        self.image_num = len(self.labels_dict.keys())

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    def __len__(self):
        return int(self.image_num // self.batch_size)

    def __getitem__(self, index):
        batch_size = self.batch_size
        image_h = self.image_h
        image_w = self.image_w
        batch_size = self.batch_size
        maxlabellength = self.maxlabellength

        x = np.zeros((batch_size, image_h, image_w, 1), dtype=np.float32)
        labels = np.ones([batch_size, maxlabellength]) * 100
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        
        sample_id = 0
        while True:
            image_name = random.choice(list(self.labels_dict.keys()))
            image_path = os.path.join(self.image_dir, image_name)
            image = cv2.imread(image_path, 0)[:, :, np.newaxis]
            h, w, _ = image.shape
            
            if h > image_h and w > image_w:
                i = np.random.randint(h - image_h + 1)
                j = np.random.randint(w - image_w + 1)
                image_patch = image[i : i + image_h, j : j + image_w]
                
                x[sample_id] = image_patch / 255. - 0.5
            else:
                x[sample_id] = image / 255. - 0.5
                
            str = self.labels_dict[image_name]
            label_length[sample_id] = len(str)
            input_length[sample_id] = w // 8 #最后解码的序列长度
            
            # sparse 编码
            labels[sample_id, :len(str)] = [int(k) for k in str]  # label 类别标签从0开始
            
            sample_id += 1
            if sample_id == batch_size:
                inputs = {'the_input': x,
                          'the_labels': labels,
                          'input_length': input_length,
                          'label_length': label_length,
                          }
                outputs = {'ctc': np.zeros([batch_size])}
                
                return inputs, outputs


class ValGenerator(Sequence):
    def __init__(self, image_dir, label_file, maxlabellength=None, image_h=None, image_w=None):
        # image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        # image_paths  = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        # image_paths = sorted(list(paths.list_images(image_dir)))
        # self.image_num = len(image_paths)
        
        self.image_dir = image_dir
        self.maxlabellength = maxlabellength
        self.image_h = image_h
        self.image_w = image_w
        self.data = []

        with open(label_file, 'r', encoding='utf-8') as file:
            self.labels_dict = {f.strip().split(' ')[0] : f.strip().split(' ')[-1] for f in file.readlines()}
        images_name = list(self.labels_dict.keys())
        self.image_num = len(images_name)

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        for name in images_name:
            image = cv2.imread(os.path.join(self.image_dir, name), 0)[:, :, np.newaxis]
            h, w, _ = image.shape

            if h > self.image_h and w > self.image_w:
                i = np.random.randint(h - self.image_h + 1)
                j = np.random.randint(w - self.image_w + 1)
                image_patch = image[i: i + self.image_h, j: j + self.image_w]

                x = image_patch / 255. - 0.5
            else:
                x = image / 255. - 0.5
                
            str = self.labels_dict[name]
            input_length = w // 8
            label_length = len(str)
            #sparse编码
            labels = [int(k) for k in str]

            inputs = {'the_input': np.expand_dims(x, axis=0),
                      'the_labels': np.expand_dims(labels, axis=0),
                      'input_length': np.expand_dims(input_length, axis=0),
                      'label_length': np.expand_dims(label_length, axis=0),
                      }
            outputs = {'ctc': np.zeros([1])}

            #self.data.append([np.expand_dims(x, axis=0), np.expand_dims(labels, axis=0), np.expand_dims(input_length, axis=0), np.expand_dims(label_length, axis=0)])
            self.data.append([inputs, outputs])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    # test code
    image_dir = r'E:\datasets\ocr_dataset\emnist_letter\images'
    train_data = r'data_process/train.txt'
    generator = ImageGenerator(image_dir, train_data, batch_size=8, image_size=28)
