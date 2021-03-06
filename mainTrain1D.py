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
from CNNHelper.ResNET import ResNet50
from keras_flops import get_flops

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
        self.TrainAcc.append(logs['accuracy'])
    def on_epoch_end(self,epoch,logs):
        self.ValLoss.append(logs['val_loss'])
        self.ValAcc.append(logs['val_accuracy'])
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
model = CNN1D(51)
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")

# model = MLPDNA(1)
# model = ResNet50(7,412)
# model = model.Autoencoder

opt = keras.optimizers.Adam(learning_rate=0.0001)
# opt = keras.optimizers.SGD(learning_rate=0.001,momentum=0.01)

# model.compile(optimizer =opt, loss="bce", metrics='accuracy')
model.compile(optimizer =opt, loss='categorical_crossentropy', metrics='accuracy')


model.summary()
mcp_save_bestAcc = keras.callbacks.ModelCheckpoint(os.path.join(savedir,'modelBestAcc.hdf5'), save_best_only=True, monitor='val_accuracy', mode='max')
mcp_save_bestLoss = keras.callbacks.ModelCheckpoint(os.path.join(savedir,'modelBestLoss.hdf5'), save_best_only=True, monitor='val_loss', mode='min')


# model.load_weights(r'D:\Sergey\FluorocodeMain\FluorocodeMain\StoredModels\2021-02-21\Training_3\modelBestLoss.hdf5' )
# model.load_weights(r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\StoredModels\2021-06-13\Training_1\modelBestAcc.hdf5' )

dt = DataLoader()
pathTraining = os.path.join(    DataSaveDir, "Training")
pathValidation = os.path.join(    DataSaveDir, "Validation")
zip_file = zipfile.ZipFile(os.path.join(savedir,'TrainingParams.zip'), "w")
zip_file.write(os.path.join(pathTraining,'ParamsTraining.csv'),arcname = 'ParamsTraining.csv')
zip_file.write(os.path.join(pathValidation,'ParamsValidation.csv'),arcname = 'ParamsValidation.csv')
zip_file.close()


X_Data ,Y_Data,Label_Data, pos  = dt.BatchLoadTrainingData(os.path.join(   pathTraining))
x_v ,y_v   ,   Label_DataV, posV    = dt.BatchLoadTrainingData(os.path.join(   pathValidation))

# Label_Data = np.squeeze(Label_Data[:,0,:])
# Label_DataV = np.squeeze(Label_DataV[:,0,:])

Label_Data = np.squeeze(Label_Data)
Label_DataV = np.squeeze(Label_DataV)

# X_Data =X_Data[0:58000,:,:]
# Label_Data  =Label_Data[0:58000,:]

# X_Data = X_Data[:,92:92+256,:]
# x_v =x_v[:,92:92+256,:]
#%%
opt = keras.optimizers.Adam(learning_rate=0.0001)
# opt = keras.optimizers.SGD(learning_rate=0.001,momentum=0.01)

# model.compile(optimizer =opt, loss="bce", metrics='accuracy')
model.compile(optimizer =opt, loss='categorical_crossentropy', metrics='accuracy')
plt.figure()
# plt.plot(X_Data[np.where(Label_Data==1)[0][0],:,:],color='b')
plt.plot(X_Data[0,:,:],color='b')

plt.figure()
plt.plot(X_Data[1,:,:],color='r')

plt.figure()
plt.plot(X_Data[2,:,:],color='g')



model_json = model.to_json()
with open(os.path.join(savedir ,"model-Architecture.json"), "w") as json_file:
    json_file.write(model_json)
    
btchsz = 256
num_iter = int( np.round( X_Data.shape[0]/btchsz))
CallBackEval =  ValDataEval(x_v, Label_DataV ,num_iter,os.path.join(savedir,'logs.json'))



history=model.fit(X_Data, Label_Data , batch_size = btchsz, epochs=10, verbose = 1, validation_data = (x_v ,Label_DataV ), callbacks=[CallBackEval,mcp_save_bestAcc,mcp_save_bestLoss])
#%%
with open(os.path.join(savedir,'history.pkl'), 'wb') as file:

    pickle.dump(history.history, file)

print(os.path.join(savedir,'history.pkl'))

# %%
