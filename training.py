#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################################
## Predicting sequences with the trained model
##################################################
## {License_info}
##################################################
## Author: Vittorio Pepe
## Copyright: Copyright 2021, VirExplorer
## Credits: DeepVirFinder
## License: {license}
## Version: 1.0.0
## Mmaintainer: Vittorio Pepe
## Email: vpepe.ds@gmail.com
## Status: development
## Usage: VirExplore.py -i <input> -l <contig lenght> - n <max samples per file> -m <model dir> -o <output dir> -c <cutoff lenght> 
##################################################


import os
import sys
import random
import optparse
import time
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import scikitplot as skplt
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          outDir='./'):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.0%}".format(value) for value in cf.flatten()/(np.sum(cf)/2)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
        
    plt.savefig(os.path.join(outDir,'confusion_matrix.png'))

tf.config.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
# Ref: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]) # Notice here
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

start_time = time.time()

os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

#### Automated run
prog_base = os.path.split(sys.argv[0])[1]

parser = optparse.OptionParser()
parser.add_option("-l", "--len", action = "store", type = int, dest = "contigLength",
									help = "contig Length")
parser.add_option("-i", "--indir", action = "store", type = "string", dest = "inDir",
									default='./data_train', help = "input directory for training, validation and test data")
parser.add_option("-o", "--outdir", action = "store", type = "string", dest = "outDir",
									default='./data_train/models', help = "output directory for the models")
parser.add_option("-e", "--epochs", action = "store", type = int, dest = "epochs",
									default=15, help = "number of epochs")

(options, args) = parser.parse_args()
if (options.contigLength is None):
	sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
	#filelog.write(prog_base + ": ERROR: missing required command-line argument")
	parser.print_help()
	sys.exit(0)

contigLength = options.contigLength
inDirTr = options.inDir + '/tr/encode'
inDirVal = options.inDir + '/val/encode'
inDirTst = options.inDir + '/tst/encode'
outDir = options.outDir
if not os.path.exists(outDir):
    os.makedirs(outDir)   
plotDir  = options.inDir + '/plots'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)
epochs = options.epochs

#### Manual run
# inDirTr = './data_train/tr/encode' 
# inDirVal = './data_train/val/encode' 
# inDirTst = './data_train/tst/encode' 
# outDir = './data_train/models' 
# plotDir = './data_train/plots'
# if not os.path.exists(plotDir):
#     os.makedirs(plotDir)
# if not os.path.exists(outDir):
#     os.makedirs(outDir)
# epochs = 50 
# contigLength = 100 

######

contigLengthk = contigLength/1000
contigLengthk = str(contigLengthk)
rdseed = 42
random.seed(rdseed)

######## loading data for training, validation, testing ##########
print("...loading data...")

## virus data
print("...loading virus data...")
# training
filename_codetrfw = [ x for x in os.listdir(inDirTr) if 'codefw.npy' in x and 'virus' in x and contigLengthk + 'k' in x ]
list_temp = []
for f in filename_codetrfw:
    print("data for training " + f)
    el = np.load(os.path.join(inDirTr,f))
    list_temp.append(el)
phageRef_codetrfw = np.concatenate(list_temp)
del list_temp
# validation
filename_codevalfw = [ x for x in os.listdir(inDirVal) if 'codefw.npy' in x and 'virus' in x and contigLengthk + 'k' in x ]
list_temp = []
for f in filename_codevalfw:
    print("data for validation " + f)
    el = np.load(os.path.join(inDirVal,f))
    list_temp.append(el)
phageRef_codevalfw = np.concatenate(list_temp)
del list_temp
# test
filename_codetstfw = [ x for x in os.listdir(inDirTst) if 'codefw.npy' in x and 'virus' in x and contigLengthk + 'k' in x ]#[0]
list_temp = []
for f in filename_codetstfw:
    print("data for testing " + f)
    el = np.load(os.path.join(inDirTst,f))
    list_temp.append(el)
phageRef_codetstfw = np.concatenate(list_temp)
del list_temp

## host data
print("...loading host data...")
# training
filename_codetrfw = [ x for x in os.listdir(inDirTr) if 'codefw.npy' in x and 'host' in x and contigLengthk + 'k' in x ]#[0]
list_temp = []
for f in filename_codetrfw:
    print("data for training " + f)
    el = np.load(os.path.join(inDirTr,f))
    list_temp.append(el)
hostRef_codetrfw = np.concatenate(list_temp)
del list_temp
# validation
filename_codevalfw = [ x for x in os.listdir(inDirVal) if 'codefw.npy' in x and 'host' in x and contigLengthk + 'k' in x ]#[0]
list_temp = []
for f in filename_codevalfw:
    print("data for validation " + f)
    el = np.load(os.path.join(inDirVal,f))
    list_temp.append(el)
hostRef_codevalfw = np.concatenate(list_temp)
del list_temp

# test
filename_codetstfw = [ x for x in os.listdir(inDirTst) if 'codefw.npy' in x and 'host' in x and contigLengthk + 'k' in x ]#[0]
#print("data for test " + filename_codetstfw)
list_temp = []
for f in filename_codetstfw:
    print("data for testing " + f)
    el = np.load(os.path.join(inDirTst,f))
    list_temp.append(el)
hostRef_codetstfw = np.concatenate(list_temp)
del list_temp


######## combine virus and host data, shuffling training data ##########
print("...combining virus and host data...")
### training V+B
Y_tr = np.concatenate((np.repeat(0, hostRef_codetrfw.shape[0]), np.repeat(1, phageRef_codetrfw.shape[0])))
X_trfw = np.concatenate((hostRef_codetrfw, phageRef_codetrfw), axis=0)
print("...shuffling training data...")
index_trfw = list(range(0, X_trfw.shape[0]))
np.random.shuffle(index_trfw)
X_trfw_shuf = X_trfw[np.ix_(index_trfw, range(X_trfw.shape[1]), range(X_trfw.shape[2]))]
del X_trfw
Y_tr_shuf = Y_tr[index_trfw]

X_trfw_shuf.shape[0]
Y_tr_shuf.shape
print('...number of sequences for training:', X_trfw_shuf.shape[0])
### validation virus+host
Y_val = np.concatenate((np.repeat(0, hostRef_codevalfw.shape[0]), np.repeat(1, phageRef_codevalfw.shape[0])))
X_valfw = np.concatenate((hostRef_codevalfw, phageRef_codevalfw), axis=0)
del hostRef_codevalfw, phageRef_codevalfw
print('...number of sequences for validation:', X_valfw.shape[0])
### test V+B
Y_tst = np.concatenate((np.repeat(0, hostRef_codetstfw.shape[0]), np.repeat(1, phageRef_codetstfw.shape[0])))
X_tstfw = np.concatenate((hostRef_codetstfw, phageRef_codetstfw), axis=0)
del hostRef_codetstfw, phageRef_codetstfw
print('...number of sequences for testing:', X_tstfw.shape[0])

######### training model #############
# parameters
nb_filter1 = 150 
nb_filter2 = 100
nb_filter3 = 50
filter_len1 = 15 
filter_len2 = 15 
filter_len3 = 15 
nb_dense1 = 75  
pool_size1 = 2
stride_size1 = 2 
dropout_pool = 0.2 
dropout_dense = 0.2 
#learningrate = 0.001
batch_size = 1000 #5000
modPattern = 'model_CNN_5Layers_'+contigLengthk+'k_fl'+str(filter_len1)+'_fn'+str(nb_filter1)+'_dn'+str(nb_dense1)+'_e'+str(epochs)
modName = os.path.join( outDir, modPattern + '.h5')
checkpointer = ModelCheckpoint(filepath=modName, verbose=1,save_best_only=True)
earlystopper = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=5, verbose=1)

##### build model #####
def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output

print("...building model...")

model = Sequential()

model.add(Conv1D(nb_filter1,input_shape= (None, 5), kernel_size = filter_len1, activation='relu', padding ='same'))
model.add(Dropout(dropout_pool))
model.add(MaxPooling1D(pool_size=pool_size1, strides=stride_size1))

model.add(Conv1D(nb_filter2, kernel_size = filter_len2, activation='relu', padding ='same'))
model.add(Dropout(dropout_pool))
model.add(MaxPooling1D(pool_size=pool_size1, strides=stride_size1))

model.add(Conv1D(nb_filter2, kernel_size = filter_len2, activation='relu', padding ='same'))
model.add(Dropout(dropout_pool))
model.add(MaxPooling1D(pool_size=pool_size1, strides=stride_size1))

model.add(Conv1D(nb_filter3, kernel_size = filter_len3, activation='relu', padding ='same'))
model.add(Dropout(dropout_pool))
model.add(MaxPooling1D(pool_size=pool_size1, strides=stride_size1))

model.add(Conv1D(nb_filter3, kernel_size = filter_len3, activation='relu', padding ='same'))
model.add(Dropout(dropout_pool))
model.add(GlobalMaxPooling1D())

model.add(Dense(nb_dense1, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(dropout_dense))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics='acc')

model.summary()

X_trfw_shuf = tf.convert_to_tensor(X_trfw_shuf)
Y_tr_shuf = tf.convert_to_tensor(Y_tr_shuf)
X_valfw = tf.convert_to_tensor(X_valfw)
Y_val = tf.convert_to_tensor(Y_val)
X_tstfw = tf.convert_to_tensor(X_tstfw)
Y_tst = tf.convert_to_tensor(Y_tst)

history = model.fit(X_trfw_shuf, Y_tr_shuf, 
                    validation_data=(X_valfw, Y_val),
                    batch_size=batch_size,
                    callbacks=[checkpointer, earlystopper],
                    epochs=epochs,
                    verbose=0)
       
#### plotting model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(os.path.join(plotDir,'loss.png'))
plt.show()

#### plotting model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(os.path.join(plotDir,'accuracy.png'))
plt.show()

#### cleanup
tf.keras.backend.clear_session()
del model

## Final evaluation AUC ###
if os.path.isfile(modName):
    model = load_model(modName)
    print("...model exists...")
    print("...loading best model...")

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
## train data

type = 'tr'
X_fw = X_trfw_shuf
Y = Y_tr_shuf
startTime = time.time()  
print("...predicting "+type+"...\n")
Y_pred = model.predict(X_fw, batch_size=100)
endTime = time.time() - startTime         
print('Execution time prediction training: ', endTime/60)
auc_roc = sklearn.metrics.roc_auc_score(Y, Y_pred)
print('auc_'+type+'='+str(auc_roc)+'\n')
### ROC Curves
Y_pred_2d = np.concatenate((1-Y_pred,Y_pred),axis=1)
skplt.metrics.plot_roc(Y, Y_pred_2d)
plt.savefig(os.path.join(plotDir,'roc_tr.png'))
#plt.show()

np.savetxt(os.path.join(outDir, modPattern + '_' + type + 'fw_Y_pred.txt'), np.transpose(Y_pred))
np.savetxt(os.path.join(outDir, modPattern + '_' + type + 'fw_Y_true.txt'), np.transpose(Y))
del Y, X_fw

# val data
type = 'val'
X_fw = X_valfw
Y = Y_val
print("...predicting "+type+"...\n")
Y_pred = model.predict(X_fw, batch_size=100)
auc_roc = sklearn.metrics.roc_auc_score(Y, Y_pred)
print('auc_'+type+'='+str(auc_roc)+'\n')
np.savetxt(os.path.join(outDir, modPattern + '_' + type + 'fw_Y_pred.txt'), np.transpose(Y_pred))
np.savetxt(os.path.join(outDir, modPattern + '_' + type + 'fw_Y_true.txt'), np.transpose(Y))
### ROC Curves
Y_pred_2d = np.concatenate((1-Y_pred,Y_pred),axis=1)
skplt.metrics.plot_roc(Y, Y_pred_2d)
plt.savefig(os.path.join(plotDir,'roc_val.png'))
#plt.show()
del Y, X_fw

# test data
type = 'tst'
X_fw = X_tstfw
Y = Y_tst
startTime = time.time() 
print("...predicting "+type+"...\n")
Y_pred = model.predict(X_fw, batch_size=100)
endTime = time.time() - startTime         
print('Execution time prediction test: ', endTime/60)
auc_roc = sklearn.metrics.roc_auc_score(Y, Y_pred)
print('auc_'+type+'='+str(auc_roc)+'\n')

### ROC Curves & Confusion Matrix
Y_pred_2d = np.concatenate((1-Y_pred,Y_pred),axis=1)
skplt.metrics.plot_roc(Y, Y_pred_2d)
plt.savefig(os.path.join(plotDir,'auc_roc_tst.png'))
#plt.show()


Y_predicted = Y_pred_2d[:,1] > 0.5
conf_matrix = confusion_matrix(Y, Y_predicted)
print(conf_matrix)

labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Host', 'Virus']
make_confusion_matrix(conf_matrix, 
                      group_names=labels,
                      categories=categories, 
                      cmap='Blues',
                      outDir=plotDir)

TN = conf_matrix[0, 0]
TP = conf_matrix[1, 1]
FN = conf_matrix[1, 0]
FP = conf_matrix[0, 1]
recall  = TP/(TP+FN)
specificity  = TN/(TN+FP)
precision = TP/(TP+FP)
f1_score  = 2*precision*recall / (precision + recall)
pos_pred_val = TP/ (TP+FP)
neg_pred_val = TN/ (TN+FN)
TPR = TP/(TP+FN) # recall
FDR = FP/(TP+FP)
FNR = FN
print('TPR: ', recall)
print('TNR: ', specificity)
print('F1 score: ', f1_score)
print('pos_pred_val: ', pos_pred_val)
print('neg_pred_val: ', neg_pred_val)


### Brier Score
from sklearn.metrics import brier_score_loss
# calculate bier score
loss_brier = brier_score_loss(Y, Y_pred)
print('Brier score: ', loss_brier )
del Y, X_fw
