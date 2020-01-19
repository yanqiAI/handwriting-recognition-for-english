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

train_dict = {"0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G", "7": "H", "8": "I", "9": "J"}

img_h = 32
img_w = 120


def generate_single(img_path_list):
	img_path = random.choice(img_path_list)
	lable = str(int(img_path.split(os.path.sep)[-2]) - 26)
	img = cv2.imread(img_path, 0)
	norm_img = normalization_h(img)
	single_samples = []
	for j, norm in enumerate(norm_img):
		img_aug = data_augment(norm, background_path)
		single_samples.append(img_aug)
	return [lable] * 50, single_samples


def generate_multiple(img_path_list):
	num_choice = np.random.randint(2, 5)
	img_paths = random.sample(img_path_list, num_choice)
	
	lable_ = []
	imgs = []
	widths = []
	for img_path in img_paths:
		lable = str(int(img_path.split(os.path.sep)[-2]) - 26)
		img = cv2.imread(img_path, 0)
		pad_w = random.randint(0, 10)
		img = cv2.copyMakeBorder(img, 0, 0, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
		h, w = img.shape
		if h > img_h:
			img = cv2.resize(img, (w, img_h))
		else:
			pad_top = random.randint(0, img_h - h)
			img = cv2.copyMakeBorder(img, pad_top, img_h - h - pad_top, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
			img = cv2.resize(img, (w, img_h))
		imgs.append(img)
		widths.append(w)
		lable_.append(lable)
	widths = np.insert(np.cumsum(widths), 0, 0)
	
	img_ = np.ones((img_h, np.sum(widths)), dtype=np.uint8) * 255
	for i, im in enumerate(imgs):
		img_[:, widths[i]: widths[i + 1]] = im
	
	_, w_ = img_.shape
	if w_ > img_w:
		img_ = cv2.resize(img_, (img_w, img_h))
	else:
		pad_l = random.randint(0, img_w - w_)
		img_ = cv2.copyMakeBorder(img_, 0, 0, pad_l, img_w - w_ - pad_l, cv2.BORDER_CONSTANT, value=(255, 255, 255))
		img_ = cv2.resize(img_, (img_w, img_h))
		
	img_ = data_augment(img_, background_path)
	
	return lable_, img_
	
	
def normalization_h(img):
	'''
	高度归一化
	img shape (32, w)
	'''
	h, w = img.shape[:2]
	norm_img = []
	
	for i in range(50):
		# resize original image
		# if np.random.random() < 0.25:
		# 	img = resize_image(img)
		
		if h > img_h:
			img = cv2.resize(img, (w, img_h))  # h=32
			pad_l = random.randint(0, img_w - w)
			img_ = cv2.copyMakeBorder(img, 0, 0, pad_l, img_w - w - pad_l, cv2.BORDER_CONSTANT, value=(255, 255, 255))
			img_ = cv2.resize(img_, (img_w, img_h))
			norm_img.append(img_)
		else:
			pad_top = random.randint(0, img_h - h)
			pad_l = random.randint(0, img_w - w)
			img_ = cv2.copyMakeBorder(img, pad_top, img_h - h - pad_top, pad_l, img_w - w - pad_l, cv2.BORDER_CONSTANT, value=(255, 255, 255))
			img_ = cv2.resize(img_, (img_w, img_h))
			norm_img.append(img_)
	return norm_img


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
	image_path = r'E:\datasets\ocr_dataset\Alphabet\Alphabet'
	background_path = r'E:\datasets\background1'
	save_path_single = os.path.join(r'E:\datasets\ocr_dataset\eng_letter\images_single')
	save_path_multiple = os.path.join(r'E:\datasets\ocr_dataset\eng_letter\images_multiple')
	if not os.path.exists(save_path_single):
		os.mkdir(save_path_single)
	
	if not os.path.exists(save_path_multiple):
		os.mkdir(save_path_multiple)
		
	img_list = sorted(list(paths.list_images(image_path)))
	
	choice_list = []
	for img_path in img_list:
		char_id = int(img_path.split(os.path.sep)[-2])
		if char_id > 25 and char_id < 36:
			choice_list.append(img_path)
	
	lables = []
	samples = []
	flag = 0 # 0: single letter 1: multiple letter
	if flag == 0:
		for i in range(2000):
			label, sample = generate_single(choice_list)
			lables.append(label)
			samples.append(sample)
	
		lables_list = [item for sublist in lables for item in sublist]
		images_list = [item for sublist in samples for item in sublist]
		
		file_lst = open(r'single_lst.txt', 'w', encoding='utf-8')
		for j, sample in enumerate(list(zip(lables_list, images_list))):
			name = save_path_single + '/' + '%05d' % j + '.png'
			cv2.imwrite(name, sample[1])
			file_lst.write(name.split(os.path.sep)[-1] + ' ' + sample[0] + '\n')
			
			if j % 100 == 0:
				print('{} has processed over!'.format(j))
				
		file_lst.close()

		print('=' * 50)
		print('All single letter samples have generated sucessfully!')
		
	else:
		for i in range(20000):
			label, sample = generate_multiple(choice_list)
			lables.append(label)
			samples.append(sample)
		
		file_lst = open(r'multiple_lst.txt', 'w', encoding='utf-8')
		for j, sample in enumerate(list(zip(lables, samples))):
			name = save_path_multiple + '/' + '%05d' % j + '.png'
			cv2.imwrite(name, sample[1])
			gt_lable = ' '.join(sample[0])
			file_lst.write(name.split(os.path.sep)[-1] + ' ' + gt_lable + '\n')
			
			if j % 100 == 0:
				print('{} has processed over!'.format(j))
				
		file_lst.close()
		
		print('=' * 50)
		print('All single letter samples have generated sucessfully!')
		