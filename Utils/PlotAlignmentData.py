#%%
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

############### USER INPUT ###############
pvalThs = 0.01
reference_genomes =  ["NC_001501.1"]
path =  r'D:\Elizabete\Fluorocode\Fluorocode\Data\MLV_Pabi\Results'
name = 'FILE1-all-reference-species-results.csv'
############## USER INPUT END ############


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
        IndItems.append(allrows[0].index(item))
        Genomes.append(allrows[1][allrows[0].index(item)])
allrows.pop(0)

Alignments = {}
for genome in Genomes:
    Alignments[genome] = {}
    Alignments[genome]["Species-matches"]=0
    Alignments[genome]["Unique-matches"]=0

# %% GET EACH NUMBER OF MATCHES

for trace in allrows:
    unique_match = 0
    for match in IndItems:
        pval = float( trace[match+5])
        if pval<pvalThs:
            Alignments[trace[match]]["Species-matches"] = Alignments[trace[match]]["Species-matches"]+1
            unique_match+=1
            if unique_match==1:
                unique_genome = trace[match]
    if unique_match==1:
        Alignments[unique_genome]["Unique-matches"] = Alignments[unique_genome]["Unique-matches"]+1
        

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
plt.bar(x,ref_bars,width=0.5,color="red")
x_rest = np.arange(np.max(x)+1,np.max(x)+1+len(bars))
plt.bar(x_rest,bars,width=0.5,color="blue")
x = np.concatenate([x,x_rest])
ax = plt.gca()
ax.set_xticklabels(Genomes)
ax.set_xticks(np.arange(len(Genomes)))
plt.setp(ax.get_xticklabels(), rotation=90)
plt.ylabel("Number of matches")
plt.title("Number of total matches")
plt.legend(["Ground truth","Control"])

# %% PLOT BARS FOR UNIQUE SPECIES
plt.figure(figsize = (18,8))
x = np.arange(0,len(ref_unique_bars))
plt.bar(x,ref_unique_bars,width=0.5,color="red")
x_rest = np.arange(np.max(x)+1,np.max(x)+1+len(bars))
plt.bar(x_rest,bars_unique,width=0.5,color="blue")
x = np.concatenate([x,x_rest])
ax = plt.gca()
ax.set_xticklabels(Genomes)
ax.set_xticks(np.arange(len(Genomes)))
plt.setp(ax.get_xticklabels(), rotation=90)
plt.ylabel("Number of matches")
plt.title("Number of unique matches")
plt.legend(["Ground truth","Control"])
# %%
