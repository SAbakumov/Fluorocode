#%%
import json,os,sys,pickle,tensorflow as tf
sys.path.insert(1, os.path.join(os.path.dirname(__file__),"Core"))

import matplotlib.pyplot as plt
import numpy as np

import Misc
from RealTraces import RealTraces
from DataHandler import DataLoader

plt.rcParams.update({'font.size': 22})
class EvalPerformance():
    def __init__(self,ModelPath,ModelName,ModelWeightsPath):
        self.ModelPath = ModelPath
        self.ModelName = ModelName
        self.ModelWeightsPath = ModelWeightsPath

    def LoadModel(self):
        with open(os.path.join(self.ModelPath, self.ModelName), 'r') as json_file:
            json_savedModel= json_file.read()
            self.model = tf.keras.models.model_from_json(json_savedModel)
        return self.model  

    def LoadWeights(self):
        self.model.load_weights(self.ModelWeightsPath)

    def LoadTrainingCurves(self):
        objectRep = open(os.path.join(self.ModelPath, 'history.pkl'), "rb")
        self.modelhistory = pickle.load(objectRep)
        return self.modelhistory

    def PlotModelHistory(self):
        plt.figure(figsize=(10,7))
        plt.plot(np.arange(0,len(self.modelhistory["loss"])), self.modelhistory["loss"],color = 'b',linewidth=2)
        plt.plot(np.arange(0,len(self.modelhistory["val_loss"])), self.modelhistory["val_loss"],color = 'r',linewidth=2)
        plt.title("Model loss")
        plt.legend(["Training","Validation"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale('log')

    def LoadSimulatedData(self,data_folder):
        dt = DataLoader()
        self.X_Data ,self.Y_Data,self.Label_Data, self.pos  = dt.BatchLoadTrainingData(data_folder)
        print(self.X_Data.shape)
# AUTOENCODER 

    def EvalNData(self,indeces):
        self.X_Data = self.X_Data/100
        Gauss      = 7.5*Misc.GetGauss1d(240 , Misc.FWHMtoSigma(1500),62.6)

        self.X_Data[indeces[0]] =self.X_Data[indeces[0]] + np.reshape(Gauss,[240,1]) 
        # self.X_Data[np.where(self.X_Data>4.5)]=4.5

        output = self.model.predict(self.X_Data[indeces])
        print(output.shape)
        plt.figure(figsize=(12,5))
        plt.plot(Gauss,linewidth=2,linestyle='--',color='g')
        plt.plot(self.X_Data[indeces[0]],linestyle = '--',Color='b')
        plt.plot(output[0],Color='r')
        plt.legend(["Blob","Input trace","Reconstructed Trace"])
    
    def EvalRealData(self,tested_traces, index):
        tested_traces[index] = ( tested_traces[index]-np.min(tested_traces[index]))/np.std(tested_traces[index])
        tested_traces[index] = np.reshape(tested_traces[index],[1,len(tested_traces[index]),1])
        output = self.model.predict(tested_traces[index] )
        plt.figure(figsize=(12,5))        
        plt.plot((tested_traces[index]).flatten(),linestyle = '--',Color='b')
        plt.plot(output[0],Color='r')

    def ClassifyData(self,data): 
        output = self.model.predict(data)
        return output

#%%
if __name__ == '__main__':
    Eval = EvalPerformance(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\StoredModels\2021-05-30\Training_1","model-Architecture.json",r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\StoredModels\2021-05-30\Training_1\modelBestLoss.hdf5" )
    Eval.LoadModel()
    Eval.LoadWeights()
    history = Eval.LoadTrainingCurves()
    Eval.PlotModelHistory()
    Eval.LoadSimulatedData(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\Validation")
    Eval.EvalNData([5,6,7])

    traces = RealTraces()
    traces.set_tracepath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiDry\segmentation-results.hdf5")
    traces.set_alignmentpath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiDry\FILE1-all-reference-species-results.csv")
    traces.load_tested_traces()

    Eval.EvalRealData(traces.tested_traces,10)
    Eval.ClassifyData(traces.tested_traces)