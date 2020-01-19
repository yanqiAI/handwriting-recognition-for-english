import os
import cv2
import glob
import random
import numpy as np
from pathlib import Path
from imutils import paths
from IPython.core.debugger import Tracer

char_dict = {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e", "6": "f", "7": "g", "8": "h", "9": "i", "10": "j",
             "11": "k", "12": "l", "13": "m", "14": "n", "15": "o", "16": "p", "17": "q", "18": "r", "19": "s",
             "20": "t", "21": "u", "22": "v", "23": "w", "24": "x", "25": "y", "26": "z", "27": "A", "28": "B",
             "29": "C", "30": "D", "31": "E", "32": "F", "33": "G", "34": "H", "35": "I", "36": "J", "37": "K",
             "38": "L", "39": "M", "40": "N", "41": "O", "42": "P", "43": "Q", "44": "R", "45": "S", "46": "T",
             "47": "U", "48": "V", "49": "W", "50": "X", "51": "Y", "52": "Z"}

train_dict = {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e", "6": "f", "7": "g", "8": "h", "9": "i", "10": "j",
             "11": "k", "12": "l", "13": "m", "14": "n", "15": "o", "16": "p", "17": "q", "18": "r", "19": "s",
             "20": "t", "21": "u", "22": "v", "23": "w", "24": "x", "25": "y", "26": "z", "27": "\'", "28": "-","29":"&"}

img_h = 64
img_w = 400

def text_crop(img, threshold):
    '''
    切除图像空白边缘部分
    '''
    ret, image_mask = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY_INV)
    n = np.argwhere(image_mask == 1)
    rows = np.unique([n[i][0] for i in range(n.shape[0])])
    cols = np.unique([n[i][1] for i in range(n.shape[0])])
    min_row = np.min(rows)
    max_row = np.max(rows)
    min_col = np.min(cols)
    max_col = np.max(cols)

    image_crop = img[min_row: max_row, min_col: max_col]
    return image_crop


def compute_padding_value(img_gray):
	'''
	计算padding的值
	取图像累积直方图中大于0.8处的值
	'''
	hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
	cdf_img = np.cumsum(hist)
	cdf_hist = cdf_img / np.max(cdf_img)
	padding_value = np.min(np.where(cdf_hist > 0.8)[0])
	
	return padding_value
	
def normalization_h(img):
	'''
	高度归一化
	img shape (32, w)
	'''
	padding_value = compute_padding_value(img)
	
	h, w = img.shape[:2]

	if h >= img_h and w >= img_w:
		img_ = cv2.resize(img, (img_w, img_h))
	elif h > img_h and w < img_w:
		img = cv2.resize(img, (w, img_h))
		pad_l = random.randint(0, img_w - w)
		img_ = cv2.copyMakeBorder(img, 0, 0, pad_l, img_w - w - pad_l, cv2.BORDER_CONSTANT, value=int(padding_value))
		img_ = cv2.resize(img_, (img_w, img_h))
	elif h <= img_h and w <= img_w:
		pad_top = random.randint(0, img_h - h)
		pad_l = random.randint(0, img_w - w)
		img_ = cv2.copyMakeBorder(img, pad_top, img_h - h - pad_top, pad_l, img_w - w - pad_l, cv2.BORDER_CONSTANT, value=int(padding_value))
		img_ = cv2.resize(img_, (img_w, img_h))
	elif h < img_h and w > img_w:
		img = cv2.resize(img, (img_w, h))
		pad_top = random.randint(0, img_h - h)
		img_ = cv2.copyMakeBorder(img, pad_top, img_h - h - pad_top, 0, 0, cv2.BORDER_CONSTANT, value=int(padding_value))
		img_ = cv2.resize(img_, (img_w, img_h))
	return img_


# data augment functions
def data_augment(img, background_path):
	# if np.random.random() < 0.15:
	# 	img = blur(img)
	if np.random.random() < 0.25:
		img = add_noise(img)
	if np.random.random() < 0.95:
		img = add_background(img, background_path)
	return img


def resize_image(img):
	img_h, img_w = img.shape[:2]
	scale = np.random.uniform(0.8, 1.2, 1)
	h = int(img_h * scale)
	w = int(img_w * scale)
	img_resize = cv2.resize(img, (w, h))
	return img_resize


def blur(img):
	img = cv2.blur(img, (3, 3))
	return img


def add_noise(img):
	noise_value = np.random.randint(0, 50)
	temp_x = np.random.randint(0, img.shape[0])
	temp_y = np.random.randint(0, img.shape[1])
	img[temp_x][temp_y] = noise_value
	return img


def add_background(img, background_path=None):
	'''
	添加背景
	'''
	# file list
	bg_images = sorted(glob.glob(os.path.join(background_path, '*.JPEG')))
	bg_images += sorted(glob.glob(os.path.join(background_path, '*.jpg')))
	bg_images += sorted(glob.glob(os.path.join(background_path, '*.png')))
	
	# 二值化处理
	ret, image_gray_binary = cv2.threshold(img, 150, 1, cv2.THRESH_BINARY)
	
	# random choose one background image
	bg_img = ''.join(random.sample(bg_images, 1))
	bg_image_gray = cv2.imread(bg_img, 0)
	
	# processing blur image
	bg_image_gray_resize = cv2.resize(bg_image_gray, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
	background_image = cv2.multiply(image_gray_binary, bg_image_gray_resize)
	return background_image


if __name__ == '__main__':
	image_path = r'E:\datasets\ocr_dataset\words\train3-11'
	#background_path = r'E:\datasets\background1'
	save_path = os.path.join(r'E:\datasets\ocr_dataset\words\words_data_1')
	
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	
	img_list = sorted(list(paths.list_images(image_path)))
	
	file_index_lst = open(r'words_index_lst_1.txt', 'w', encoding='utf-8')
	file_chars_lst = open(r'words_chars_lst_1.txt', 'w', encoding='utf-8')
	

	for i, img_path in enumerate(img_list):
		label_words = []
		img = cv2.imread(img_path)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		norm = normalization_h(img_gray)
		#img_aug = data_augment(norm, background_path)
		label = img_path.split(os.path.sep)[-1].split('-')[1:]
		
		for w in label:
			if '.jpg' not in w:
				label_words.append(w)
			else:
				label_words.append(w[:-4])
				
		label_index = ' '.join(label_words)
		label_char = ' '.join([train_dict[p] for p in label_words])
		
		name = save_path + '/' + '%08d' % i + '.png'
		cv2.imwrite(name, norm)
		
		file_index_lst.write(name.split(os.path.sep)[-1] + ' ' + label_index + '\n')
		file_chars_lst.write(name.split(os.path.sep)[-1] + ' ' + label_char + '\n')
		
		if i % 100 == 0:
			print('{} has processed over!'.format(i))
			
	file_index_lst.close()
	file_chars_lst.close()
	
	print('=' * 50)
	print('All words samples have generated sucessfully!')
	