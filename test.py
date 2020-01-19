# coding:utf-8
import os
import argparse
import numpy as np
from pathlib import Path
import cv2
import random, shutil
from PIL import Image
from densenet_blstm import *
#from efficientnet_b0 import *
import time
from IPython.core.debugger import Tracer


char_dic = {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e", "6": "f", "7": "g", "8": "h", "9": "i", "10": "j",
            "11": "k", "12": "l", "13": "m", "14": "n", "15": "o", "16": "p", "17": "q", "18": "r", "19": "s",
            "20": "t", "21": "u", "22": "v", "23": "w", "24": "x", "25": "y", "26": "z", "27": "\'", "28": "-","29":"&"}

characters = 'abcdefghijklmnopqrstuvwxyz\'-&'
nclass = len(characters) + 1
img_h = 64
img_w = 400


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_dir', type=str, default='test_data', help="test image dir")
    parser.add_argument('--test_path', type=str, default=r'E:\datasets\ocr_dataset\words', help="images path")
    parser.add_argument('--test_lst', type=str, default=r'data_process/file_lst/val_words_mix.txt', help="test image list file")
    parser.add_argument('--train_lst', type=str, default=r'E:\my_code\english_project\single_letter\data_process\train_words_mix.txt', help="train image list file")
    parser.add_argument('--weight_file', type=str, default='checkpoints_densecnn/weights.001-0.047-0.99625.hdf5', help="trained weight file")
    args = parser.parse_args()
    return args


def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        #if pred_text[i] != nclass - 1 and (not ((i > 0 and pred_text[i] == pred_text[i - 1]) or (i > 1 and pred_text[i] == pred_text[i - 2]) or (i > 2 and pred_text[i] == pred_text[i - 3]))):
        #if pred_text[i] != nclass - 1 and (not ((i > 0 and pred_text[i] == pred_text[i - 1]) or (i > 1 and pred_text[i] == pred_text[i - 2]))):
        if pred_text[i] != nclass - 1 and (not ((i > 0 and pred_text[i] == pred_text[i - 1]))):
            char_list.append(characters[pred_text[i]])            
    return u''.join(char_list).replace('&', ' ')


def normalization_h(img):
    '''
	高度归一化
	img shape (64, w)
	'''
    h, w = img.shape[:2]
    
    if h >= img_h and w >= img_w:
        img_ = cv2.resize(img, (img_w, img_h))
    elif h > img_h and w < img_w:
        img = cv2.resize(img, (w, img_h))
        pad_l = random.randint(0, img_w - w)
        img_ = cv2.copyMakeBorder(img, 0, 0, pad_l, img_w - w - pad_l, cv2.BORDER_CONSTANT, value=255)
        img_ = cv2.resize(img_, (img_w, img_h))
    elif h <= img_h and w <= img_w:
        pad_top = random.randint(0, img_h - h)
        pad_l = random.randint(0, img_w - w)
        img_ = cv2.copyMakeBorder(img, pad_top, img_h - h - pad_top, pad_l, img_w - w - pad_l, cv2.BORDER_CONSTANT, value=255)
        img_ = cv2.resize(img_, (img_w, img_h))
    elif h < img_h and w > img_w:
        img = cv2.resize(img, (img_w, h))
        pad_top = random.randint(0, img_h - h)
        img_ = cv2.copyMakeBorder(img, pad_top, img_h - h - pad_top, 0, 0, cv2.BORDER_CONSTANT, value=255)
        img_ = cv2.resize(img_, (img_w, img_h))
    return img_


def predict(model, img):
    height, width = img.shape
    if width < img_w:
        img = normalization_h(img)
        img = img.reshape([1, 64, img_w, 1])
    else:
        scale = height * 1.0 / 64
        width = int(width / scale)
        img = cv2.resize(img, (width, 64))
        img = img.reshape([1, 64, width, 1])
     
    X = img / 255.0 - 0.5
    y_pred = model.predict(X)  # (h, w, nclass)
    y_pred = y_pred[:, :, :]
    
    # 解码出对应的字符
    out = decode(y_pred)
    return out


if __name__ == '__main__':
    args = get_args()
    image_dir = args.image_dir
    test_path = args.test_path
    test_lst = args.test_lst
    train_lst = args.train_lst
    weight_file = args.weight_file
    
    input = Input(shape=(64, None, 1), name='the_input')
    y_pred = dense_cnn(input, num_class=nclass)
    #y_pred = efficientnet_B0(input, num_class=nclass)
    model = Model(inputs=input, outputs=y_pred)
    model.load_weights(weight_file)
    model.summary()


    # start testing
    flag = 0 # 0:open test images  1: open test filelist  2: find hard samples
    if flag == 0:
        imgs = os.listdir(image_dir)
        for img_dir in imgs:
            img = cv2.imread(os.path.join(image_dir, img_dir), 0)
            pred_char = predict(model, img)
            print(img_dir, pred_char)
            print('=' * 50)

    elif flag == 1:
        with open(test_lst, 'r', encoding='utf-8') as file:
            items = [f.strip() for f in file.readlines()]
        file.close()
        print('test num: {}'.format(len(items)))

        save_error_sample = r'error_samples'
        if not os.path.exists(save_error_sample):
            os.mkdir(save_error_sample)
        
        start_time = time.time()
        correct = 0
        for i, item in enumerate(items):
            img_dir = os.path.join(test_path, item.split(' ')[0])
            gt_char = [char_dic[p] for p in item.split(' ')[1:]]
            gt_char = u''.join(gt_char).replace('&', ' ')
            
            img = cv2.imread(img_dir, 0)
            pred_char = predict(model, img)
            
            if gt_char == pred_char:
                correct += 1
                #print('file_dir: {}, gt_char: {}, pred_char: {}'.format(img_dir, gt_char, pred_char))
            else:
                shutil.copy(img_dir, save_error_sample)
                print('file_dir: {}, gt_char: {}, pred_char: {}'.format(img_dir, gt_char, pred_char))
                
            if i > 999:
                break
        
        #precision = correct / len(items)
        precision = correct / 1000
        print('=' * 50)
        print('Test precision is {}, cost time {} s'.format(precision, time.time() - start_time))
        print('=' * 50)
    else:
        with open(test_lst, 'r', encoding='utf-8') as file:
            items = [f.strip() for f in file.readlines()]
        file.close()
        print('train num: {}'.format(len(items)))
        
        save_hard_sample = r'E:\datasets\ocr_dataset\words\hard_samples'
        if not os.path.exists(save_hard_sample):
            os.mkdir(save_hard_sample)
            
        for i, item in enumerate(items):
            #img_dir = os.path.join(test_path, item.split(' ')[0])
            img_dir = item.split(' ')[0]
            gt_char = [char_dic[p] for p in item.split(' ')[1:]]
            
            # if '&' in gt_char:
            #     gt_char = [c.replace('&', ' ') for c in gt_char]
            gt_char = u''.join(gt_char).replace('&', ' ')
            

            img = cv2.imread(img_dir, 0)
            pred_char = predict(model, img)
            
            if gt_char != pred_char:
                shutil.copy(img_dir, save_hard_sample)
                print('file_dir: {}, gt_char: {}, pred_char: {}'.format(img_dir, gt_char, pred_char))
                
            if i % 100 == 0:
                print('{} has processed over!'.format(i))
            
            if i > 999:
                break

        print('=' * 50)
        print('Hard samples have saved sucessfully!')
        print('=' * 50)
        
