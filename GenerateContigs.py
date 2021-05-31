#%%
import csv,random,os ,numpy as np
from Core import Misc, SIMTraces
from datetime import date
import matplotlib.pyplot as plt

Params = {"Wavelength" : 576,
        "NA" : 1.4,
        "FragmentSize" :800,
        "PixelSize" : 32.25*2,
        "ResEnhancement":1,
        "FromLags" :True,
        "ShuffleData":False,
        "Lags":[],
        "Enzyme" : 'TaqI',
        "NumTransformations"  :[200,200],
        "StretchingFactor" :[1.75],
        "LowerBoundEffLabelingRate" : 0.7,
        "UpperBoundEffLabelingRate" : 0.9,
        "amplitude_variation":[8.55696606597531,	3.23996722003733],
        "step" :3,
        "PixelShift": 0.8,
        "NoiseAmp": [5],
        "GenerateFullReference" :True,
        "LocalNormWindow":10000,
        "ZNorm": False,
        "Norm":  False,
        "Date" : str(date.today()),
        "Type" : "Training",
        "Genomes" : ['NC_000913.3'],
        "Classes": [0,1],
        "FPR": 0.5, #per kb 0.5
        "FPR2": 0.1, #per kb 0.2
        "Random-min": 52,
        "Random-max": 210,
        "SaveFormatAsCSV": False,
        "ConcatToCsv": False}    

FullGenome = []
with open('D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\DataForContigs\GEN-NC_000913.3.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        if len(row)>0:
            FullGenome = [float(x) for x in row]



contig_bounds = np.sort( random.choices(FullGenome,k=5))
FullGenome = np.array(FullGenome)

dist_between_contigs = 20000
end_frag_size = 30000

for cnt_num in range(0,len(contig_bounds)-1):
    cnt_begin = np.where(FullGenome>contig_bounds[cnt_num]+dist_between_contigs)[0][0]
    cnt_end = np.where(FullGenome<contig_bounds[cnt_num+1]-dist_between_contigs)[0][-1]

    contig = FullGenome[cnt_begin:cnt_end]

    cnt_leftbound= (contig[0:np.where(contig>contig[0]+end_frag_size)[0][0]]).astype(np.int64)
    cnt_rightbound= (contig[np.where(contig<contig[-1]-end_frag_size)[0][-1]:-1]).astype(np.int64) 
    plt.barh(0,np.max(cnt_rightbound)-np.min(cnt_leftbound),left = np.min(cnt_leftbound) )
    plt.barh(0,np.max(cnt_leftbound)-np.min(cnt_leftbound),left = np.min(cnt_leftbound) ,color='r')
    plt.barh(0,np.max(cnt_rightbound)-np.min(cnt_rightbound),left = np.min(cnt_rightbound),color='r' )
    plt.ylim([-0.5, 2])
    cnt_leftbound = cnt_leftbound-np.min(cnt_leftbound)
    cnt_rightbound = cnt_rightbound-np.min(cnt_rightbound)
    with open(os.path.join(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\DataForContigs","contig"+str(cnt_num)+'left.csv'), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(cnt_leftbound)
    with open(os.path.join(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\DataForContigs","contig"+str(cnt_num)+'right.csv'), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(cnt_rightbound)



# %%
