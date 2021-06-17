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
# xcorrvals = [0.7,0.65,0.6]
xcorrvals = [0.6]

colors = plt.cm.jet(np.linspace(0,1,3))
#%%
traces = RealTraces.RealTraces()


# # folder = r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\StoredModels\2021-06-02\Training_1'
folder = r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\StoredModels\2021-06-16\Training_1'

Eval = Metrics.EvalPerformance(folder,"model-Architecture.json",os.path.join(folder, "modelBestAcc.hdf5") )
Eval.LoadModel()
Eval.LoadWeights()

datasets = {}
alignments = {}
pathdst = r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TestingMulticlass'
dst = os.listdir(pathdst )
dst.append(dst[1])
dst.pop(1)

# Classes = ['Random','NC_000913.3','CP034237.1','NC_007795.1','NC_003197.2','NC_000964.3','NZ_CP045605.1','LR215978.1','NC_004567',' NC_004307','CR626927.1']
# for genome in Classes:
#     datasets[genome] = os.path.join(pathdst,dst[Classes.index(genome)])
#     alignments[genome] = []
datasets['NC_000913.3']= r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiEthGly\segmentation-results.hdf5"
datasets['CP034237.1']= r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\Pectobacterium\Pectobacterium_2\segmentation-results.hdf5"
datasets['Other'] = r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\Staphylococcus\segmentation-results.hdf5"

alignments = {}
alignments['NC_000913.3'] = r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiEthGly\logs_xcorr\FILE1-all-reference-species-results.csv"
alignments['CP034237.1'] = r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\Pectobacterium\Pectobacterium_2\logs_xcorr\FILE1-all-reference-species-results.csv"
alignments['Other'] = []
# alignments['CP034237.1'] = []
# alignments['NC_000913.3'] = []
# %%
Classes = ['Other','NC_000913.3','CP034237.1']
plt.figure(figsize=(10,10))

# for xcorr in xcorrvals:
xcorr = 0.6
AlignmentMatrix = np.zeros([len(Classes),len(Classes)])
AlignmentMatrix = np.zeros([11,11])

for genome in Classes:
    traces.set_tracepath(datasets[genome])
    traces.set_alignmentpath(alignments[genome])
    traces.set_alignmentparams(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\Pectobacterium\Pectobacterium_2\logs_xcorr\FILE0-classification-parameters.csv")

 
    traces.load_tested_traces()
    if len(alignments[genome])>0:
        traces.load_alignment_file()
        traces.get_cut_xcorr(xcorr,genome)
        traces.normalize_traces_local(10000,True)
        traces.pick_traces_subset(genome,'Alignment')
        traces.prepare_input_size(412)
    else:
        traces.normalize_traces_local(10000,True)
        traces.pick_traces_subset(genome,'NoAlignment')
        traces.prepare_input_size(412)

    t1,output_default = Eval.ClassifyData(traces.default_input_traces)
    t2,output_flipped = Eval.ClassifyData(traces.flipped_input_traces)

    print((t1+t2)/len(output_flipped))
    output = np.stack((output_flipped,output_default), axis=2)
    output = np.max(output,axis=2)
    
    output[output>0.7]=1
    output[output<0.7]=0

    for j in range(0,11):
        AlignmentMatrix[Classes.index(genome),j]  = len(np.where(output[:,j]==1)[0])

    AlignmentMatrix[Classes.index(genome),:] = AlignmentMatrix[Classes.index(genome),:]/len( output)
# %%
# fig, ax = plt.figure(figsize=(10,10))

fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(AlignmentMatrix, cmap=plt.cm.Blues)

for i in range(0,len(AlignmentMatrix)):
    for j in range(0,len(AlignmentMatrix)):
        c = AlignmentMatrix[j,i]
        ax.text(i, j, str(np.round(c,2)), va='center', ha='center')
plt.xticks(np.arange(0,len(Classes)),labels =Classes ,rotation='vertical')
plt.yticks(np.arange(0,len(Classes)),labels =Classes )

# %%
