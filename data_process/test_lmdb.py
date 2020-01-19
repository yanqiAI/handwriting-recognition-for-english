
# -*- coding: utf-8 -*-
import numpy as np
import lmdb
import cv2
 
with lmdb.open(r"E:\datasets\ocr_dataset\words\train_lmdb") as env:
    txn = env.begin()
    nSamples = int(txn.get('num-samples'.encode()))
    print('samples_num:', nSamples)

    for key, value in txn.cursor():
        key = str(key, encoding='utf-8') #bytes ==> str
        imageBuf = np.fromstring(value, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        label_key = 'label-%09d'.encode() % int(key.split('-')[-1])
        label = txn.get(label_key).decode('utf-8')
        print(key, label)
        if img is not None:
            cv2.imshow('image', img)
            cv2.waitKey()
        else:
            print('This is a label: {}'.format(value))