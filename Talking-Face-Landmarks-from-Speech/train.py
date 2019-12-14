# Written by S. Emre Eskimez, in 2017 - University of Rochester
# Usage: python train.py -i path-to-hdf5-train-file/ -u number-of-hidden-units -d number-of-delay-frames -c number-of-context-frames -o output-folder-to-save-model-file

import tensorflow as tf
import librosa
import numpy as np
import os, shutil, subprocess
from keras import backend as K
from keras.layers import Input, LSTM, Dense, Reshape, Activation, Dropout, Flatten
from keras.models import Model,load_model
from tqdm import tqdm
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
import h5py
from keras.callbacks import TensorBoard
import argparse, fnmatch
import pickle
import random
import time, datetime
import os
#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
sess = tf.Session()
K.set_session(sess)
os.environ['PYTHONHASHSEED'] = '128'
np.random.seed(128)
random.seed(128)
tf.set_random_seed(128)
#-----------------------------------------#

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--in-file", type=str, help="Input file containing train data")
parser.add_argument("-u", "--hid-unit", type=int, help="hidden units")
parser.add_argument("-d", "--delay", type=int, help="Delay in terms of number of frames")
parser.add_argument("-c", "--ctx", type=int, help="context window size")
parser.add_argument("-o", "--out-fold", type=str, help="output folder")
parser.add_argument("-e", "--epochs", type=int, help="Num of Epochs")
args = parser.parse_args()

# output_path = args.out_fold+'_'+str(args.hid_unit)+'/'
output_path = os.path.join(args.out_fold,"exp_ATFL_hid_units={}_delay={}_ctx={}_epochs={}/".format(args.hid_unit,args.delay,args.ctx,args.epochs))
if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)

ctxWin = args.ctx
num_features_X = 128 * (ctxWin+1)# input feature size
num_features_Y = 136 # output feature size --> (68, 2)
num_frames = 75 # time-steps
batchsize = 128
h_dim = args.hid_unit
lr = 1e-3


drpRate = 0.2 # Dropout rate 
recDrpRate = 0.2 # Recurrent Dropout rate 

frameDelay = args.delay # Time delay

numEpochs = args.epochs
dset = h5py.File(args.in_file, 'r') # Input hdf5 file must contain two keys: 'flmark' and 'MelFeatures'. 
# 'flmark' contains the normalized face landmarks and shape must be (numberOfSamples, time-steps, 136)
# 'MelFeatures' contains the features, namely the delta and double delta mel-spectrogram. Shape = (numberOfSamples, time-steps, 128)

numIt = int(dset['flmark'].shape[0]//batchsize) + 1
metrics = ['MSE', 'MAE']

def addContext(melSpc, ctxWin):
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx

def writeParams():
    # Write parameters of the network and training configuration
    with open(os.path.join(output_path, "model_info.txt"), "w") as text_file:
        text_file.write("{:30} {}\n".format('', output_path))
        text_file.write("------------------------------------------------------------------\n")
        text_file.write("{:30} {}\n".format('batchsize:', batchsize))
        text_file.write("{:30} {}\n".format('num_frames:', num_frames))
        text_file.write("{:30} {}\n".format('num_features_X:', num_features_X))
        text_file.write("{:30} {}\n".format('num_features_Y:', num_features_Y))
        text_file.write("{:30} {}\n".format('drpRate:', drpRate))
        text_file.write("{:30} {}\n".format('recDrpRate:', recDrpRate))
        text_file.write("{:30} {}\n".format('learning-rate:', lr))
        text_file.write("{:30} {}\n".format('h_dim:', h_dim))
        text_file.write("{:30} {}\n".format('train filename:', args.in_file))
        text_file.write("{:30} {}\n".format('loss:', metrics[0]))
        text_file.write("{:30} {}\n".format('metrics:', metrics[1:]))
        text_file.write("{:30} {}\n".format('num_it:', numIt))
        text_file.write("{:30} {}\n".format('frameDelay:', frameDelay))
        text_file.write("------------------------------------------------------------------\n")
        model.summary(print_fn=lambda x: text_file.write(x + '\n'))

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def dataGenerator():
    X_batch = np.zeros((batchsize, num_frames, num_features_X))
    Y_batch = np.zeros((batchsize, num_frames, num_features_Y))

    idxList = range(dset['flmark'].shape[0])

    batch_cnt = 0    
    while True:
        random.shuffle(idxList)
        for i in idxList:
            cur_lmark = dset['flmark'][i, :, :]
            cur_mel = dset['MelFeatures'][i, :, :]

            if frameDelay > 0:
                filler = np.tile(cur_lmark[0:1, :], [frameDelay, 1])
                cur_lmark = np.insert(cur_lmark, 0, filler, axis=0)[:num_frames]
             
            X_batch[batch_cnt, :, :] = addContext(cur_mel, ctxWin)
            Y_batch[batch_cnt, :, :] = cur_lmark
            
            batch_cnt+=1

            if batch_cnt == batchsize:
                batch_cnt = 0
                yield X_batch, Y_batch

def build_model():
    net_in = Input(shape=(num_frames, num_features_X))
    h = LSTM(h_dim, 
            activation='sigmoid', 
            dropout=drpRate, 
            recurrent_dropout=recDrpRate,
            return_sequences=True)(net_in)
    h = LSTM(h_dim, 
            activation='sigmoid',  
            dropout=drpRate, 
            recurrent_dropout=recDrpRate,
            return_sequences=True)(h)
    h = LSTM(h_dim, 
            activation='sigmoid', 
            dropout=drpRate, 
            recurrent_dropout=recDrpRate,
            return_sequences=True)(h)
    h = LSTM(num_features_Y, 
            activation='sigmoid', 
            dropout=drpRate, 
            recurrent_dropout=recDrpRate,
            return_sequences=True)(h)
    model = Model(inputs=net_in, outputs=h)
    model.summary()

    opt = Adam(lr=lr)

    model.compile(opt, metrics[0], 
                metrics= metrics[1:])
    return model
def build_model2(filepath):

    model = load_model(filepath)
    model.summary()

    opt = Adam(lr=lr)

    model.compile(opt, metrics[0], 
                metrics= metrics[1:])
    return model

gen = dataGenerator()
print("Loading Weights from: {}".format('/home/Talking-Face-Landmarks-from-Speech/models/D40_C5.h5'))
model = build_model2('/home/Talking-Face-Landmarks-from-Speech/models/D40_C5.h5')

writeParams()

callback = TensorBoard(output_path)
callback.set_model(model)

k = 0
for epoch in tqdm(range(numEpochs)):
    for i in tqdm(range(numIt)):
        X_test, Y_test = next(gen)

        logs = model.train_on_batch(X_test, Y_test)
        if np.isnan(logs[0]):
            print ('NAN LOSS!')
            exit()

        write_log(callback, metrics, logs, k)
        k+=1

    model.save(output_path+'talkingFaceModel.h5')