#%%
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# TOTUNIQUE_DUAL_CHANNEL = []
TOTUNIQUE_SINGLE_CHANNEL = []
SPECIFICITY_SINGLE_CHANNEL = []
# single_xcorrscores_good = []
#%%
############### USER INPUT ###############
lengthpx = 32
pvalThs = 0.001
reference_genomes =  ['NC_000913.3.csv']
path =  r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\EColi\logs2'
name = 'FILE1-all-reference-species-results.csv'
############## USER INPUT END ############
#%%

allrows = []
with open(os.path.join(path,name)) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
            # row = [float(x) for x in row]
            allrows.append(row)


# %% GET REFERENCE SPECIES AND INDECES
IndItems = []
Genomes = []
for item in allrows[0]:
    if "reference species" in item:
        try:
            if len(IndItems)<20:
                IndItems.append(allrows[0].index(item))
                Genomes.append(allrows[1][allrows[0].index(item)])
        except:
            continue
allrows.pop(0)




Alignments = {}
for genome in Genomes:
    Alignments[genome] = {}
    Alignments[genome]["Species-matches"]=0
    Alignments[genome]["Unique-matches"]=0
    Alignments[genome]["XCorrscore"]=[]

# %% GET EACH NUMBER OF MATCHES

for trace in allrows:
    unique_match = 0
    for match in IndItems:
        pval = float( trace[match+5])
        xcorrscore = float( trace[match+1])
        if pval<pvalThs:
            Alignments[trace[match]]["Species-matches"] = Alignments[trace[match]]["Species-matches"]+1
            Alignments[trace[match]]["XCorrscore"].append(xcorrscore) 

            unique_match+=1
            if unique_match==1:
                unique_genome = trace[match]
    if unique_match==1:
        Alignments[unique_genome]["Unique-matches"] = Alignments[unique_genome]["Unique-matches"]+1
        Alignments[unique_genome]["XCorrscore"].append(xcorrscore) 
    

# %% GET BARS FOR TOTAL SPECIES
bars = []
ref_bars = []
ref_genomes = []
all_genomes = []
for genome in Genomes:
    if genome in reference_genomes:
        ref_bars.append(Alignments[genome]["Species-matches"])
        ref_genomes.append(genome)
    else:
        bars.append(Alignments[genome]["Species-matches"])
        all_genomes.append(genome)
# %% GET BARS FOR TOTAL SPECIES
bars_unique = []
ref_unique_bars = []
for genome in Genomes:
    if genome in reference_genomes:
        ref_unique_bars.append(Alignments[genome]["Unique-matches"])
    else:
        bars_unique.append(Alignments[genome]["Unique-matches"])
Genomes = ref_genomes+all_genomes
# %% PLOT BARS FOR TOTAL SPECIES
plt.figure(figsize = (18,8))
x = np.arange(0,len(ref_bars))
plt.bar(x,np.array(ref_bars)/1000,width=0.5,color="red")
x_rest = np.arange(np.max(x)+1,np.max(x)+1+len(bars))
plt.bar(x_rest,np.array(bars)/1000,width=0.5,color="blue")
x = np.concatenate([x,x_rest])
ax = plt.gca()
ax.set_xticklabels(Genomes)
ax.set_xticks(np.arange(len(Genomes)))
plt.setp(ax.get_xticklabels(), rotation=90)
plt.ylabel("Number of matches")
plt.title("Number of total matches [%]")
plt.legend(["Ground truth","Control"])

# %% PLOT BARS FOR UNIQUE SPECIES
plt.figure(figsize = (18,8))
x = np.arange(0,len(ref_unique_bars))
plt.bar(x,np.array(ref_unique_bars)/1000,width=0.5,color="red")
x_rest = np.arange(np.max(x)+1,np.max(x)+1+len(bars))
plt.bar(x_rest,np.array(bars_unique)/1000,width=0.5,color="blue")
x = np.concatenate([x,x_rest])
ax = plt.gca()
ax.set_xticklabels(Genomes)
ax.set_xticks(np.arange(len(Genomes)))
plt.setp(ax.get_xticklabels(), rotation=90)
plt.ylabel("Number of matches")
plt.title("Number of unique matches [%]")
plt.legend(["Ground truth","Control"])
# bars_tot = bars_unique+ref_unique_bars
# bars_unique =np.sum( np.array(bars_unique))
# bars_tot  = np.sum( np.array(bars_tot))
# %%
# TOTUNIQUE_SINGLE_CHANNEL.append(ref_unique_bars[0]/len(allrows)*100)
# SPECIFICITY_SINGLE_CHANNEL.append(100 - bars_unique /bars_tot*100)
# %%
frag_lengths = np.array([356,256,156,75,32])
frag_lengths = frag_lengths*64.5/1.72/0.34/1000
# # %%
# SPECIFICITY_SINGLE_CHANNEL.pop(-1)
# TOTUNIQUE_SINGLE_CHANNEL.pop(-1)

# plt.figure(figsize=(12,8))
# plt.plot(frag_lengths,TOTUNIQUE_SINGLE_CHANNEL,'-o' ,color='b',ms=20)
# plt.plot(frag_lengths,TOTUNIQUE_DUAL_CHANNEL,'-o',color='r',ms=20)
# plt.legend(["Single color", "Dual color"])
# plt.xlabel("Fragment size [kb]")
# plt.ylabel("Unique TP matches [%]")
# #%%
# plt.figure(figsize=(12,8))
# plt.plot(frag_lengths,SPECIFICITY_SINGLE_CHANNEL,'-o' ,color='b',ms=20)
# plt.plot(frag_lengths,SPECIFICITY_DUAL_CHANNEL,'-o',color='r',ms=20)
# plt.legend(["Single color", "Dual color"])
# plt.xlabel("Fragment size [kb]")
# plt.ylabel("Specificity [%]")
# %%
# plt.figure(figsize = (8,6))

# xcorr_scores = []
# xcorr_scores_g= []
# for xcorr in  single_xcorrscores:
#     xcorr_scores.append(np.mean(np.array(xcorr)))
# for xcorr in single_xcorrscores_good:
#     xcorr_scores_g.append(np.mean(np.array(xcorr)))
# plt.plot(frag_lengths, xcorr_scores,'-o',ms=10,color='blue')
# plt.plot(frag_lengths, xcorr_scores_g,'-o',ms=10,color='red')

# plt.xlabel("Fragment size [kb]")
# plt.ylabel("Average TP x-corr score")
# plt.legend(["Bad quality","Good quality"])

# %%
