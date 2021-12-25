#
# The SELDnet architecture
#

from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate,AveragePooling2D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.models import load_model
import keras
from Network import CRblock, MDDC_block, Adptive_Asymmetric_CNN, Adptive_Asymmetric_Attention_CNN, conv_selfattention, TDNNF
keras.backend.set_image_data_format('channels_first')
from IPython import embed
import numpy as np


def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, f_pool_size, t_pool_size,
              rnn_size, fnn_size, weights, doa_objective, is_accdoa):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))

    spec_cnn1 = spec_start
    spec_cnn1 = MDDC_block(spec_cnn1)
    #    spec_cnn1 = D2block(spec_cnn1)
    #    spec_cnn1 = RA_block(spec_cnn1)
    #spec_cnn1 = CRblock(spec_cnn1)
    #spec_cnn1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", dilation_rate=(1, 1))(spec_cnn1)
    spec_cnn1 = Adptive_Asymmetric_Attention_CNN(spec_cnn1)
    #spec_cnn1 = Adptive_Asymmetric_CNN(spec_cnn1)
    spec_cnn1 = CRblock(spec_cnn1)
    #spec_cnn1 = BatchNormalization(axis=1)(spec_cnn1)
    #spec_cnn1 = Activation('relu')(spec_cnn1)
    spec_cnn1 = AveragePooling2D(pool_size=(5, 2))(spec_cnn1)
    spec_cnn1 = Dropout(dropout_rate)(spec_cnn1)

    spec_cnn2 = spec_cnn1
    #    spec_cnn2 = RA_block(spec_cnn2)
    #spec_cnn2 = CRblock(spec_cnn2)
    #spec_cnn2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", dilation_rate=(1, 1))(spec_cnn2)
    spec_cnn2 = Adptive_Asymmetric_Attention_CNN(spec_cnn2)
    #spec_cnn2 = Adptive_Asymmetric_CNN(spec_cnn2)
    spec_cnn2 = CRblock(spec_cnn2)
    #spec_cnn2 = BatchNormalization(axis=1)(spec_cnn2)
    #spec_cnn2 = Activation('relu')(spec_cnn2)
    spec_cnn2 = AveragePooling2D(pool_size=(1, 2))(spec_cnn2)
    spec_cnn2 = Dropout(dropout_rate)(spec_cnn2)

    spec_cnn3 = spec_cnn2
    #    spec_cnn3 = RA_block(spec_cnn3)
    #spec_cnn3 = CRblock(spec_cnn3)
    #spec_cnn3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", dilation_rate=(1, 1))(spec_cnn3)
    spec_cnn3 = Adptive_Asymmetric_Attention_CNN(spec_cnn3)
    #spec_cnn3 = Adptive_Asymmetric_CNN(spec_cnn3)
    spec_cnn3 = CRblock(spec_cnn3)
    #spec_cnn3 = BatchNormalization(axis=1)(spec_cnn3)
    #spec_cnn3 = Activation('relu')(spec_cnn3)
    spec_cnn3 = AveragePooling2D(pool_size=(1, 2))(spec_cnn3)
    spec_cnn3 = Dropout(dropout_rate)(spec_cnn3)

    spec_cnn4 = spec_cnn3
    #    spec_cnn4 = RA_block(spec_cnn4)
    #spec_cnn4 = CRblock(spec_cnn4)
    #spec_cnn4 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", dilation_rate=(1, 1))(spec_cnn4)
    spec_cnn4 = Adptive_Asymmetric_Attention_CNN(spec_cnn4)
    #spec_cnn4 = Adptive_Asymmetric_CNN(spec_cnn4)
    spec_cnn4 = CRblock(spec_cnn4)
    #spec_cnn4 = BatchNormalization(axis=1)(spec_cnn4)
    #spec_cnn4 = Activation('relu')(spec_cnn4)
    spec_cnn4 = AveragePooling2D(pool_size=(1, 2))(spec_cnn4)
    spec_cnn4 = Dropout(dropout_rate)(spec_cnn4)

    spec_cnn5 = spec_cnn4
    #    spec_cnn5 = RA_block(spec_cnn5)
    #spec_cnn5 = CRblock(spec_cnn5)
    #spec_cnn5 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", dilation_rate=(1, 1))(spec_cnn5)
    spec_cnn5 = Adptive_Asymmetric_Attention_CNN(spec_cnn5)
    #spec_cnn5 = Adptive_Asymmetric_CNN(spec_cnn5)
    spec_cnn5 = CRblock(spec_cnn5)
    #spec_cnn5 = BatchNormalization(axis=1)(spec_cnn5)
    #spec_cnn5 = Activation('relu')(spec_cnn5)
    spec_cnn5 = AveragePooling2D(pool_size=(1, 2))(spec_cnn5)
    spec_cnn5 = Dropout(dropout_rate)(spec_cnn5)



    # TDNNFCnt1 = TDNNF(spec_cnn5)
    # TDNNFCnt1 = Dropout(dropout_rate)(TDNNFCnt1)
    # TDNNFCnt2 = TDNNF(TDNNFCnt1)
    # TDNNFCnt2 = Dropout(dropout_rate)(TDNNFCnt2)
    # TDNNFCnt3 = TDNNF(TDNNFCnt2)
    # TDNNFCnt3 = Dropout(dropout_rate)(TDNNFCnt3)
    # TDNNFCnt4 = TDNNF(TDNNFCnt3)
    # TDNNFCnt4 = Dropout(dropout_rate)(TDNNFCnt4)
    # TDNNFCnt5 = TDNNF(TDNNFCnt4)
    # TDNNFCnt5 = Dropout(dropout_rate)(TDNNFCnt5)
    # TDNNFCnt6 = TDNNF(TDNNFCnt5)
    # TDNNFCnt6 = Dropout(dropout_rate)(TDNNFCnt6)
    # TDNNFCnt7 = TDNNF(TDNNFCnt6)
    # TDNNFCnt7 = Dropout(dropout_rate)(TDNNFCnt7)
    # TDNNFCnt8 = TDNNF(TDNNFCnt7)
    # TDNNFCnt8 = Dropout(dropout_rate)(TDNNFCnt8)
    # TDNNFCnt9 = TDNNF(TDNNFCnt8)
    # TDNNFCnt9 = Dropout(dropout_rate)(TDNNFCnt9)
    # TDNNFCnt10 = TDNNF(TDNNFCnt9)
    # TDNNFCnt10 = Dropout(dropout_rate)(TDNNFCnt10)


    spec_cnn = Permute((2, 1, 3))(spec_cnn5)




    # CNN
    # spec_cnn = spec_start
    # for i, convCnt in enumerate(f_pool_size):
    #     spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
    #     spec_cnn = BatchNormalization()(spec_cnn)
    #     spec_cnn = Activation('relu')(spec_cnn)
    #     spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
    #     spec_cnn = Dropout(dropout_rate)(spec_cnn)
    # spec_cnn = Permute((2, 1, 3))(spec_cnn)

    # spec_cnn = spec_start
    # for i, convCnt in enumerate(f_pool_size):
    #     spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
    #     spec_cnn = BatchNormalization()(spec_cnn)
    #     spec_cnn = Activation('relu')(spec_cnn)
    #     spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn)
    #     spec_cnn = Dropout(dropout_rate)(spec_cnn)
    # spec_cnn = Permute((2, 1, 3))(spec_cnn)


    spec_rnn = Reshape((data_out[-2] if is_accdoa else data_out[0][-2], -1))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='concat'
        )(spec_rnn)


    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[-1] if is_accdoa else data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    model = None
    if is_accdoa:
        model = Model(inputs=spec_start, outputs=doa)
        model.compile(optimizer=Adam(), loss='mse')
    else:
        # FC - SED
        sed = spec_rnn
        for nb_fnn_filt in fnn_size:
            sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
            sed = Dropout(dropout_rate)(sed)
        sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
        sed = Activation('sigmoid', name='sed_out')(sed)

        if doa_objective is 'mse':
            model = Model(inputs=spec_start, outputs=[sed, doa])
            model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
        elif doa_objective is 'masked_mse':
            doa_concat = Concatenate(axis=-1, name='doa_concat')([sed, doa])
            model = Model(inputs=spec_start, outputs=[sed, doa_concat])
            model.compile(optimizer=Adam(), loss=['binary_crossentropy', masked_mse], loss_weights=weights)
        else:
            print('ERROR: Unknown doa_objective: {}'.format(doa_objective))
            exit()
    model.summary()
    return model


def masked_mse(y_gt, model_out):
    nb_classes = 12 #TODO fix this hardcoded value of number of classes
    # SED mask: Use only the predicted DOAs when gt SED > 0.5
    sed_out = y_gt[:, :, :nb_classes] >= 0.5 
    sed_out = keras.backend.repeat_elements(sed_out, 3, -1)
    sed_out = keras.backend.cast(sed_out, 'float32')

    # Use the mask to computed mse now. Normalize with the mask weights 
    return keras.backend.sqrt(keras.backend.sum(keras.backend.square(y_gt[:, :, nb_classes:] - model_out[:, :, nb_classes:]) * sed_out))/keras.backend.sum(sed_out)


def load_seld_model(model_file, doa_objective):
    if doa_objective is 'mse':
        return load_model(model_file)
    elif doa_objective is 'masked_mse':
        return load_model(model_file, custom_objects={'masked_mse': masked_mse})
    else:
        print('ERROR: Unknown doa objective: {}'.format(doa_objective))
        exit()



