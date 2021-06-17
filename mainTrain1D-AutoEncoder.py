# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:55:37 2020

@author: Sergey
"""
#%%
import os
import tensorflow as tf
from CNNHelper.CNN1D import CNN1D 
from CNNHelper.DenseDNA import MLPDNA
from CNNHelper.AutoEncoder1D import AutoEncoder1D


import Core.Misc as Misc
import json 
from datetime import date
from tensorflow import keras
from Core.DataHandler import DataLoader
import numpy as np
import sys
import zipfile
import pickle
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
savedir = Misc.GetModelSavePath(ROOT_DIR,str(date.today()))
DataSaveDir = os.path.join(ROOT_DIR, "Data")

log = {}
log['Test-Loss'] = []
log['Test-acc'] = []
log['Val-Loss'] = []
log['Val-acc'] = []

log = json.dumps(log)
f = open(os.path.join(savedir ,'logs.json'),"w")
f.write(log)
f.close()
    


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2028)])
  except RuntimeError as e:
    print(e)




def GetArray(X_Data):
    X_DataPast = np.reshape(X_Data[:,0:256,0],[X_Data.shape[0],256,1])
    X_DataFuture = np.reshape(X_Data[:,256:256+32,0],[X_Data.shape[0],32,1])
    
    X_DataPast = np.reshape(X_DataPast,[X_DataPast.shape[0],4,64,1])

    return X_DataPast, X_DataFuture


class ValDataEval(tf.keras.callbacks.Callback):
    def __init__(self, x_v,y_v,num_iter,logpath):
        self.ValDataX = x_v
        self.ValDataY = y_v
        self.NumIter = num_iter
        self.CurrentLoss = 0
        self.CurrentAcc = 0
        self.LogPath = logpath
        self.TrainLoss = []
        self.ValLoss = []
        self.TrainAcc = []
        self.ValAcc = []

    def on_batch_end(self, batch, logs):     

        self.TrainLoss.append(logs['loss'])
        self.TrainAcc.append(logs['mae'])
    def on_epoch_end(self,epoch,logs):
        self.ValLoss.append(logs['val_loss'])
        self.ValAcc.append(logs['val_mae'])
        # if (batch % 5000 == 0  and batch>=5000) or batch==0:
        #     f = open(self.LogPath)
        #     log = json.load(f)
        #     f.close()
        #     log['Test-Loss'].append(logs['loss'])
        #     log['Test-acc'].append(logs['accuracy'])
        #     results = self.model.evaluate(self.ValDataX,self.ValDataY,batch_size=256,verbose=0)
        #     ls  = results[0]
        #     log['Val-Loss'].append(ls)
        #     log['Val-acc'].append(results[1])
        #     self.CurrentLoss = ls
        #     self.CurrentAcc = results[1]
        #     log = json.dumps(log)

        #     f = open(self.LogPath,"w")
        #     f.write(log)
        #     f.close()
                        
            
        # sys.stdout.write("\r"+ str(batch) + ' out of ' + str(self.NumIter) + ' || '  + ' Training loss: ' + format(logs['loss'], '.4f') + ', Training acc: ' +  format(logs['accuracy'], '.4f') + ' || ' + ' Validation loss: ' + format( self.CurrentLoss, '.4f') + ', Validation acc: ' + format( self.CurrentAcc, '.4f') )
        # sys.stdout.flush()

    # def on_epoch_begin(self,epoch,logs):
    #     print('\n'+ 'Epoch: '+ str(epoch)+'\n')

        
        
            
            

            
        
       



# model = LSTMAutoEncoder1D();
# model = CNN1D(1)
model = AutoEncoder1D()

model = model.Autoencoder

opt = keras.optimizers.Adam(learning_rate=0.001)

# model.compile(optimizer =opt, loss="bce", metrics='accuracy')
model.compile(optimizer =opt, loss="mae", metrics='mae')


model.summary()
mcp_save_bestAcc = keras.callbacks.ModelCheckpoint(os.path.join(savedir,'modelBestAcc.hdf5'), save_best_only=True, monitor='val_accuracy', mode='max')
mcp_save_bestLoss = keras.callbacks.ModelCheckpoint(os.path.join(savedir,'modelBestLoss.hdf5'), save_best_only=True, monitor='val_loss', mode='min')


# model.load_weights(r'D:\Sergey\FluorocodeMain\FluorocodeMain\StoredModels\2021-02-21\Training_3\modelBestLoss.hdf5' )
# model.load_weights(r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\StoredModels\2021-06-01\Training_1\modelBestAcc.hdf5' )

dt = DataLoader()
pathTraining = os.path.join(    DataSaveDir, "AutoencoderTrain")
pathValidation = os.path.join(    DataSaveDir, "AutoencoderVal")
zip_file = zipfile.ZipFile(os.path.join(savedir,'TrainingParams.zip'), "w")
zip_file.write(os.path.join(pathTraining,'ParamsAutoencoderTrain.csv'),arcname = 'ParamsTraining.csv')
zip_file.write(os.path.join(pathValidation,'ParamsAutoencoderVal.csv'),arcname = 'ParamsValidation.csv')
zip_file.close()


X_Data ,Y_Data,Label_Data, pos  = dt.BatchLoadTrainingData(os.path.join(   pathTraining))
x_v ,y_v   ,   Label_DataV, posV    = dt.BatchLoadTrainingData(os.path.join(   pathValidation))



Label_Data = np.squeeze(Label_Data[:,0,:])
Label_DataV = np.squeeze(Label_DataV[:,0,:])



X_Data = X_Data/100
x_v = x_v/100
# X_Data = X_Data[:,92:92+256,:]
# x_v =x_v[:,92:92+256,:]
plt.figure()
plt.plot(X_Data[np.where(Label_Data==1)[0][0],:,:],color='b')
plt.figure()
plt.plot(x_v[np.where(Label_Data==1)[0][0],:,:],color='r')
#%%


model_json = model.to_json()
with open(os.path.join(savedir ,"model-Architecture.json"), "w") as json_file:
    json_file.write(model_json)
    
btchsz = 64
num_iter = int( np.round( X_Data.shape[0]/btchsz))
CallBackEval =  ValDataEval(x_v, x_v ,num_iter,os.path.join(savedir,'logs.json'))



history=model.fit(X_Data,X_Data , batch_size = btchsz, epochs=10, verbose = 1, validation_data = (x_v ,x_v), callbacks=[CallBackEval,mcp_save_bestAcc,mcp_save_bestLoss])
#%%
with open(os.path.join(savedir,'history.pkl'), 'wb') as file:

    pickle.dump(history.history, file)

print(os.path.join(savedir,'history.pkl'))

# %%
