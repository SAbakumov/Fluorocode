#%%
import sys,os,csv,tensorflow as tf, numpy as np
sys.path.insert(1, os.path.join(os.path.dirname(__file__),"Core"))
# tf.config.set_visible_devices([], 'GPU')
import RealTraces
import Metrics
import sklearn.metrics
import matplotlib.pyplot as plt
import copy
import time
import imageio
plt.rcParams.update({'font.size': 12})
def padpower2(A):
    return np.pad(A, (0, int(2**np.ceil(np.log2(len(A)))) - len(A)), 'constant')

#%%
folder = r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\StoredModels\2021-06-04\Training_1'
Eval = Metrics.EvalPerformance(folder,"model-Architecture.json",os.path.join(folder, "modelBestLoss.hdf5") )
Eval.LoadModel()
Eval.LoadWeights()

traces = RealTraces.RealTraces()
traces.set_tracepath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiEthGly\segmentation-results.hdf5")
traces.load_tested_traces()
plt.figure()
mses = []
indeces = []
filenames = []
slidingwindow = 20
traces_to_write = []
for trace_ind in range(0, len(traces.tested_traces)):
    trace = traces.tested_traces[trace_ind]


    trace = (trace-np.min(trace))/np.std(trace)
    orlen = len(trace)
    trace = padpower2(trace)
    t1,output = Eval.ClassifyData(np.reshape(trace,[1,len(trace)]))
    trace_mse = []
    trace = trace[0:orlen]
    output = output.flatten()
    output = output[0:orlen]
    for mseind in range(0,len(output)-slidingwindow):
        trace_mse.append(np.mean(np.power(trace[mseind:mseind+slidingwindow]- output[mseind:mseind+slidingwindow],2)))
    trace_mse = np.array(trace_mse)
    trace_mse = trace_mse*trace[0:len(trace)-slidingwindow]
    ind = np.where(trace_mse>0.15)[0]
    if len(ind)>0:
        plt.figure()
        plt.subplots(ncols=1, nrows=2)
        ax1 = plt.subplot(211)
    
        ax1.plot(trace)
        ax1.plot(output)
        ax2 = plt.subplot(212, sharex=ax1)

        ax2.plot(trace_mse)
        indeces.append(trace_ind)
        plt.savefig(os.path.join(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\gifs",str(trace_ind)+'.png'))
        plt.close()
        filenames.append(str(trace_ind)+'.png')
    # else:
        traces_to_write.append(traces.tested_traces[trace_ind])

with open(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TestAutoencoder\segmented_traces_filtered.csv", "w", newline="") as outfile:

    writer = csv.writer(outfile)

    for trace in traces_to_write:

        writer.writerow(trace)

    # mses.append(np.mean(np.power(output.flatten()[0:orlen ]-trace[0:orlen ],2)))
    # if mses[-1]<0.008:
    #     indeces.append(trace_ind)
    #     plt.figure()
    #     plt.plot(output.flatten()[0:orlen ])
    #     plt.plot(trace[0:orlen ])
    #     plt.savefig(os.path.join(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\gifs",str(trace_ind)+'.png'))
    #     filenames.append(str(trace_ind)+'.png')
print("Done")

#%%
with imageio.get_writer(r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\gifs\mygif.gif', mode='I',fps=2) as writer:
    for filename in filenames:
        image = imageio.imread(os.path.join(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\gifs",filename))
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(os.path.join(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\gifs",filename))
    # plt.plot(output.flatten()[0:orlen ])
    # plt.plot(trace[0:orlen ])
    # break


# t1,output_default = Eval.ClassifyData(traces.default_input_traces)



4#%%
# %%
plt.figure(figsize=(10,10))

for xcorr in xcorrvals:
    traces.set_tracepath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiDry\segmentation-results.hdf5")
    traces.load_tested_traces()
    traces.load_alignment_file()

    # traces.get_cut_xcorr(xcorr,'NC_000913.3')
    traces.normalize_traces_local(10000,True)
    traces.pick_traces_subset('NC_000913.3','NoAlignment')
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
# plt.legend(["X-Corr cutoff:"+str(x) for x in xcorrvals])
plt.title("E-Coli Classifier- TP: E-Coli, FP: Vibrio ")
# %%
