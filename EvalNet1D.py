#%%
import sys,os,tensorflow as tf, numpy as np
sys.path.insert(1, os.path.join(os.path.dirname(__file__),"Core"))
# tf.config.set_visible_devices([], 'GPU')
import RealTraces
import Metrics
import sklearn.metrics
import matplotlib.pyplot as plt
import copy
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")
xcorrvals = [0.7,0.65,0.6]
colors = plt.cm.jet(np.linspace(0,1,3))
#%%
traces = RealTraces.RealTraces()
traces.set_tracepath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\Pectobacterium\Pectobacterium_3\segmentation-results.hdf5")
# traces.set_tracepath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\Vibrio\segmentation-results.hdf5")
# traces.set_tracepath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiEthGly\segmentation-results.hdf5")

traces.set_alignmentpath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiEthGly\logs_xcorr\FILE1-all-reference-species-results.csv")
# traces.set_alignmentpath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\Pectobacterium\Pectobacterium_2\logs_xcorr\FILE1-all-reference-species-results.csv")
traces.set_alignmentparams(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\Pectobacterium\Pectobacterium_2\logs_xcorr\FILE0-classification-parameters.csv")

traces.load_tested_traces()
# traces.load_alignment_file()


# traces.get_cut_length(400)
# traces.get_cut_xcorr(0.75,'NC_000913.3')
traces.normalize_traces_local(10000,True)
traces.pick_traces_subset('NC_000913.3','NoAlignment')
traces.prepare_input_size(412)
traces.plot_traces(20)

#%%
# folder = r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\StoredModels\2021-06-02\Training_1'
folder = r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\StoredModels\2021-06-13\Training_1'

Eval = Metrics.EvalPerformance(folder,"model-Architecture.json",os.path.join(folder, "modelBestAcc.hdf5") )
Eval.LoadModel()
Eval.LoadWeights()
# history = Eval.LoadTrainingCurves()
# Eval.PlotModelHistory()
t1,output_default = Eval.ClassifyData(traces.default_input_traces)
t2,output_flipped = Eval.ClassifyData(traces.flipped_input_traces)
output = np.transpose(np.array([output_default.flatten(), output_flipped.flatten()]))
output = np.max(output,axis=1)
# output[np.where(output<0.98)]=0
# output[np.where(output>0.98)]=1
# print(len(np.where(output==1)[0])/len(output))
print("done")
x_vFP =copy.deepcopy( output)
# %%
plt.figure(figsize=(10,10))

for xcorr in xcorrvals:
    # traces.set_tracepath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\Pectobacterium\Pectobacterium_2\segmentation-results.hdf5")
    traces.set_tracepath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiEthGly\segmentation-results.hdf5")
    traces.load_tested_traces()
    
    traces.load_alignment_file()
# CP034237.1
    traces.get_cut_xcorr(xcorr,'NC_000913.3')
    traces.normalize_traces_local(10000,True)
    traces.pick_traces_subset('NC_000913.3','Alignment')
    traces.prepare_input_size(412)
    # Eval.PlotModelHistory()
    t1,output_default = Eval.ClassifyData(traces.default_input_traces)
    t2,output_flipped = Eval.ClassifyData(traces.flipped_input_traces)
    output = np.transpose(np.array([output_default.flatten(), output_flipped.flatten()]))
    output = np.max(output,axis=1)
    print((t1+t2)/len(output))

    # output[np.where(output<0.98)]=0
    # output[np.where(output>0.98)]=1
    x_vTP = output




    y_true = np.ones(x_vTP.shape)
    y_false = np.zeros(x_vFP.shape)
    y_true = np.concatenate([y_true,y_false])
    y_pred = np.concatenate([x_vTP,x_vFP])
    fpr,tpr,ths = sklearn.metrics.roc_curve(y_true,y_pred)

    plt.plot(fpr,tpr,linewidth=1.2,color=colors[xcorrvals.index(xcorr)])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim([0,1])
    plt.ylim([0,1])
plt.legend(["X-Corr cutoff:"+str(x) for x in xcorrvals])
plt.title("Pectobacterim Classifier- TP: P.Carot., FP: E.Coli ")
# %%
