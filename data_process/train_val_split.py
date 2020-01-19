import os
import random

with open(r'file_lst/ori_label.txt', 'r', encoding='utf-8') as multi:
	items_m = [f.strip() for f in multi.readlines()]
# with open(r'single_lst.txt', 'r', encoding='utf-8') as single:
# 	items_s = [f.strip() for f in single.readlines()]

#samples = items_m + items_s
samples = items_m

num = len(samples)
print('Num: {}'.format(num))

random.seed(42)
random.shuffle(samples)

rate = 0.1
pick_num = int(num * rate)
val_samples = random.sample(samples, pick_num)
train_samples = list(set(samples) - (set(val_samples)))

train_set = open(r'file_lst/train_sentence.txt', 'w', encoding='utf-8')
val_set = open(r'file_lst/val_sentence.txt', 'w', encoding='utf-8')

for sample in train_samples:
	train_set.write(sample + '\n')

for sample in val_samples:
	val_set.write(sample + '\n')

train_set.close()
val_set.close()
