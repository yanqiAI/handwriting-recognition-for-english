import os
import random
import numpy as np
import cv2
from imutils import paths
import h5py
from IPython.core.debugger import Tracer

def read_files(file_lst):
    with open(file_lst, 'r', encoding='utf-8') as f:
        items = [p.strip() for p in f.readlines()]
    return items

def dataset(images_dir, lst):
    images = np.zeros((len(lst), 64, 400, 1), dtype=np.uint8)
    labels = np.ones((len(lst), 50), dtype=np.int32) * 100
    
    for sample_id, item in enumerate(lst):
        img_path = item.split(' ')[0]
        label = item.split(' ')[1:]
        img_data = cv2.imread(os.path.join(images_dir, img_path), 0)
        images[sample_id] = img_data[:, :, np.newaxis]
        labels[sample_id, :len(label)] = [int(k) - 1 for k in label]

        if sample_id % 1000 == 0:
            print('processed {} ...'.format(sample_id))
            
    return images, labels
    
def file_h5(images_dir, file_lst, save_dir):
    items = read_files(file_lst)
    items = items[:1000]
    random.seed(42)
    random.shuffle(items)
    items_num = len(items)
    print('data num: {}'.format(len(items)))

    if items_num > 100000:
        bolcks_num = items_num // 100000
        residual = items_num % 100000
        items = items[:-residual]
        # group for items
        items_block = [items[i:i+100000] for i in range(0, len(items), 100000)]
        # create h5 file
        f = h5py.File(save_dir + '/' + 'train_words.h5', 'w')
        x = f.create_dataset('images', shape=(100000, 64, 400, 1), maxshape=(None, 64, 400, 1), dtype=np.uint8)
        y = f.create_dataset('labels', shape=(100000, 50), maxshape=(None, 50), dtype=np.int32)
        
        for idx, item_block in enumerate(items_block):
            f = h5py.File(save_dir + '/' + 'train_words.h5', 'a') #add model
            
            x = f['images']
            y = f['labels']

            images = np.zeros((len(item_block), 64, 400, 1), dtype=np.uint8)
            labels = np.ones((len(item_block), 50), dtype=np.int32) * 100

            for sample_id, item in enumerate(item_block):
                img_path = item.split(' ')[0]
                label = item.split(' ')[1:]
                img_data = cv2.imread(os.path.join(images_dir, img_path), 0)
                # img_data = img_data / 255. - 0.5
                images[sample_id] = img_data[:, :, np.newaxis]
                labels[sample_id, :len(label)] = [int(k) - 1 for k in label]
    
                if sample_id % 1000 == 0:
                    print('processed {} ...'.format(sample_id))
            
            x.resize([idx * len(item_block) + len(item_block), 64, 400, 1])
            y.resize([idx * len(item_block) + len(item_block), 50])
            
            x[idx * len(item_block): (idx + 1) * len(item_block)] = images
            y[idx * len(item_block): (idx + 1) * len(item_block)] = labels
            
            del images, labels
            print('=' * 50)
            print('The {} block data is loaded and saved sucessfully!'.format(idx + 1))
        
        if residual != 0:
            print('start to save residual data...')
            res_item = items[-residual:]
            images, labels = dataset(images_dir, res_item)
    
            x.resize([items_num, 64, 400, 1])
            y.resize([items_num, 50])
            x[-residual:] = images
            y[-residual:] = labels
            
            del images, labels
            
        f.close()
        print('h5 file has been saved sucessfully!')
        print('==' * 50)
       
    else:
        images, labels = dataset(images_dir, items)
        print('==' * 50)
        print('All data have processed over and start to save h5 file...')

        # Create a h5 file
        f = h5py.File(save_dir + '/' + 'train_words.h5', 'w')
        f.create_dataset('images', data=images)
        f.create_dataset('labels', data=labels)
        f.close()
        
        print('h5 file has been saved sucessfully!')
        print('==' * 50)

def check_h5(h5_dir):
    dataset = h5py.File(os.path.join(h5_dir, 'train_words.h5'), 'r')
    # check h5 file key
    print(dataset.keys())
    
    images = dataset['images']
    labels = dataset['labels']

    print('images_shape', images.shape)
    print('labels_shape', labels.shape)

def load_h5(h5_dir):
    dataset = h5py.File(os.path.join(h5_dir, 'train_words.h5'), 'r')
    images = dataset['images']
    labels = dataset['labels']
    
    print('images_0', type(images[0]))
    print('labels_0', type(labels[0]))

if __name__ == '__main__':
    imgs_dir = r'E:\datasets\ocr_dataset\words'
    save_dir = r'E:\datasets\ocr_dataset\words'
    train_lst = r'E:\datasets\ocr_dataset\words\train_words.txt'
    file_h5(imgs_dir, train_lst, save_dir)
    check_h5(save_dir)
    load_h5(save_dir)
