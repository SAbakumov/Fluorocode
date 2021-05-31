#%%
import sys,os,numpy as np
sys.path.insert(1, os.path.join(os.path.dirname(__file__),"Core"))

import RealTraces
import Metrics


traces = RealTraces.RealTraces()
traces.set_tracepath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiDry\segmentation-results.hdf5")
traces.set_alignmentpath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiDry\FILE1-all-reference-species-results.csv")
traces.set_alignmentparams(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiEthGly\logs_xcorr\FILE0-classification-parameters.csv")

traces.load_tested_traces()
traces.load_alignment_file()


traces.get_cut_length(256)
traces.get_cut_xcorr(0.7,'NC_000913.3')
traces.normalize_traces_local(10000)
traces.pick_traces_subset('NC_000913.3')
traces.prepare_input_size(256)

folder = r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\StoredModels\2021-05-31\Training_1"
Eval = Metrics.EvalPerformance(folder,"model-Architecture.json",os.path.join(folder, "modelBestLoss.hdf5") )
Eval.LoadModel()
Eval.LoadWeights()
history = Eval.LoadTrainingCurves()
Eval.PlotModelHistory()
output_default = ( np.round(Eval.ClassifyData(traces.default_input_traces),decimals=2)).flatten()
output_flipped = (np.round(Eval.ClassifyData(traces.flipped_input_traces),decimals=2)).flatten()
output = np.transpose(np.array([output_default, output_flipped]))
print(np.max(output,axis=1))
