# coding:utf-8
from keras.layers import Input, Dense, Flatten, Permute
from keras.layers import Bidirectional, LSTM
from keras.layers.wrappers import TimeDistributed
from keras_efficientnets import EfficientNetB0
from keras.models import Model


# EfficientNet network
def efficientnet_B0(input, num_class = None):
    #input_layer = Input(shape = (None, None, input_channel_num))
    dense = EfficientNetB0(include_top = False, weights = None, input_tensor = input, pooling = 'max')
    x = dense.get_layer('swish_16').output

    x = Permute((2, 1, 3), name='permute')(x) #(b, None, 8, 240)
    x = TimeDistributed(Flatten(), name='flatten')(x)

    # add blstm layers
    rnnunit = 256
    x = Bidirectional(LSTM(rnnunit, return_sequences=True, implementation=2), name='blstm1')(x)
    #x = Dense(rnnunit, name='blstm1_out', activation='linear')(x)
    x = Bidirectional(LSTM(rnnunit, return_sequences=True, implementation=2), name='blstm2')(x)
    
    # softmax output layer
    y_pred = Dense(num_class, name='out', activation='softmax')(x)

    return y_pred

if __name__ == '__main__':
    input = Input(shape=(64, None, 1), name='the_input')
    y_pred = efficientnet_B0(input, 30)
    model = Model(inputs=input, outputs=y_pred)
    model.summary()
