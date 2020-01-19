import argparse
import numpy as np
from pathlib import Path
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras.layers.core import Lambda
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
#from efficientnet_b0 import efficientnet_B0, simple_cnn
#from simple_densenet import dense_cnn
from densenet_blstm import dense_cnn
#from generator import ImageGenerator, ValGenerator
#from generator_h5 import ImageGenerator
from generator_lmdb import LmdbDataset
from margin_softmax import *
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy
from IPython.core.debugger import Tracer

class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr
    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.20:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.40:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


def get_args():
    parser = argparse.ArgumentParser(description="train am-softmax model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, default=r'E:\datasets\ocr_dataset\eng_letter\mixture',
                        help="train image dir")
    parser.add_argument("--train_data", type=str, default=r'E:\datasets\ocr_dataset\words\data_lmdb_val',
                        help="validation image dir")
    parser.add_argument("--val_data", type=str, default=r'E:\datasets\ocr_dataset\words\data_lmdb_val',
                        help="validation image dir")
    parser.add_argument('--image_h', type=int, default=64,
                        help='training image h')
    parser.add_argument('--image_w', type=int, default=400,
                        help='training image w')
    parser.add_argument('--maxlabellength', type=int, default=50,
                        help='training max length of label')
    parser.add_argument('--character', type=str, default='abcdefghigklmnopqrstuvwxyz\'-&',
                        help='character label')
    parser.add_argument('--rgb', action='store_true', 
                        help='use rgb input')
    parser.add_argument("--batch_size", type=int, default=2,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=60,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=5000,
                        help="steps per epoch")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints_densecnn",
                        help="checkpoint dir")
    args = parser.parse_args()
    return args

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def main():
    args = get_args()
    image_dir = args.image_dir
    train_data = args.train_data
    val_data = args.val_data
    #image_size = args.image_size
    image_h = args.image_h
    image_w = args.image_w
    maxlabellength = args.maxlabellength
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    steps = args.steps
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    
    input = Input(shape=(image_h, None, 1), name='the_input')
    y_pred = dense_cnn(input, num_class=len(args.character)+1)
    #model = efficientnet_B0(num_class=26)
    #model = simple_cnn(num_class=26)
    
    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    if args.weight is not None:
        basemodel.load_weights(args.weight)
    
    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    
    opt = Adam(lr=lr)
    callbacks = []
    
    # model.compile(optimizer=opt,
    #               loss=[sparse_amsoftmax_loss],
    #               metrics=[sparse_categorical_accuracy])

    model.compile(optimizer=opt, loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['accuracy'])

    # generator = ImageGenerator(train_data, batch_size=batch_size, maxlabellength=maxlabellength, image_h=image_h, image_w=image_w)
    # val_generator = ImageGenerator(val_data, batch_size=batch_size, maxlabellength=maxlabellength, image_h=image_h, image_w=image_w)
    #val_generator = ValGenerator(image_dir, val_data, maxlabellength=maxlabellength, image_h=image_h, image_w=image_w)

    generator = LmdbDataset(train_data, args)
    val_generator = LmdbDataset(val_data, args)

    output_path.mkdir(parents=True, exist_ok=True)

    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
    callbacks.append(ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_acc:.5f}.hdf5",
                                     monitor="val_acc",
                                     verbose=1,
                                     mode="max",
                                     save_best_only=True))
    
    hist = model.fit_generator(generator=generator,
                               steps_per_epoch=steps,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)
    
    model.save(str(output_path) + '/' + 'model_final.h5')


if __name__ == '__main__':
    main()
