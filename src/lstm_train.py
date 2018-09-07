import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers.recurrent import GRU,LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse
from keras.utils import to_categorical

parser = argparse.ArgumentParser(description='up sampling')

parser.add_argument('--fea', default="C_sync", type=str, help = "Select C_sync, mfcc_sync, log-S_sync")

parser.add_argument('--blk_length', default=10, type=int, help = "lengh of lstm")

parser.add_argument('--dropout', default=0.1, type=float, help = "dropout_prob")

parser.add_argument('--epochs', default=600, type=int, help = "number of epoch")

parser.add_argument('--batchsize', default=20, type=int, help = "number of epoch")

parser.add_argument('--lr', default=0.001, type=float, help = "learning rate")
parser.add_argument('--decay', default=0.001, type=float, help = "wight decay")
parser.add_argument('--filters', default=40, type=int, help = "number of filters")
parser.add_argument('--l2', default=0.03, type=float, help = "L2 value")

args = parser.parse_args()



params = {
    "epochs": args.epochs,
    "batch_size": args.batchsize,
    "dropout_keep_prob": args.dropout,
    "numclass":10
}

import glob

from PIL import Image
from matplotlib import cm

length = 300*2
dup    = 2

train_files = glob.glob("../train/%s*npy" % (args.fea))
val_files   = glob.glob("../val/%s*npy"   % (args.fea))
test_files   = glob.glob("../test/%s*npy"   % (args.fea))

def normalize(arr,min_,max_):
    return (arr-min_)/(max_ - min_)

def read_npy(dir):
    labels = []
    arrs   = []
    for d in dir:
        labels.append(int(d.split("_")[-1].split(".")[0]))
        y = np.load(d)
        im = Image.fromarray(np.uint8(cm.seismic(y)*255))

        arr    = np.asarray(im.resize((length, 12*dup),Image.ANTIALIAS))[:,:,:3]/255.0
        block_length = length/(args.blk_length)
        fro_ = np.array(range(0,length,block_length//2)[:-1])
        to_  = fro_+ block_length
        arrs.append(np.array([arr[:,fro_[i]:to_[i],:]  for i in range(len(fro_))]))

    return np.array(arrs) , np.array(labels)

X_train, y_train = read_npy(train_files)        
X_val,   y_val   = read_npy(val_files)
X_test,  y_test   = read_npy(test_files)

print (np.unique(y_train))

y_train = to_categorical(y_train-1,num_classes=params['numclass'])

y_val  =  to_categorical(y_val-1,num_classes=params['numclass'])

y_test  = to_categorical(y_test-1,num_classes=params['numclass'])


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
from keras.callbacks import ModelCheckpoint

from keras.layers import Flatten

def rnn_lstm(input_shape):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""
    
    model = Sequential()   
    
    model.add(ConvLSTM2D(filters=args.filters, kernel_size=(5, 5),
                   input_shape=input_shape,
                   padding='same', return_sequences=True,kernel_regularizer=regularizers.l2(args.l2),activation='relu'))
    model.add(BatchNormalization())


    model.add(ConvLSTM2D(filters=args.filters, kernel_size=(5, 5),
                       padding='same', return_sequences=False,kernel_regularizer=regularizers.l2(args.l2),activation='relu'))
    model.add(BatchNormalization())


    model.add(Flatten())
    model.add(Dropout(args.dropout))
    model.add(Dense(units=params['numclass'],activation='softmax',kernel_regularizer=regularizers.l2(args.l2)) )

    adam = optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.99, epsilon=None, decay=args.decay, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model


input_shape = X_train.shape[-4:]

model = rnn_lstm(input_shape)
saved_model = "BI_LSTM_L2"

'''
from keras.models import load_model
try:
    df_his = pd.read_csv("history_%s.csv" %(saved_model),index_col=0)
    model = load_model("models/%s" % (saved_model)) 
except:
    print("re train")
    df_his = None
'''

df_his=None
import os
directory = "models"

if not os.path.exists(directory):
    os.makedirs(directory)

# Train RNN (LSTM) model with train set
history = model.fit(X_train, y_train,
          batch_size=params['batch_size'],
          epochs=params['epochs'],
          validation_data=[X_val,y_val],
          callbacks = [ModelCheckpoint(filepath="models/"+saved_model,monitor='val_acc', verbose=1, save_best_only=True, mode='max')]
          )


if df_his is None:
    df = pd.DataFrame(history.history)
    df.to_csv("history_%s.csv" %(saved_model),header=True)
else:
    df = pd.concat([df_his, pd.DataFrame(history.history)]).reset_index()
    df.to_csv("history_%s.csv" %(saved_model),header=True)
