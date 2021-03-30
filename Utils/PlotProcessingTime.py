
#%%
import csv
import numpy as np 
import matplotlib.pyplot as plt
#%%
path =  r'D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\EColi\logs2'
name = 'FILE4-processing-time.csv'

allrows = []
with open(os.path.join(path,name)) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
            # row = [float(x) for x in row]
            allrows.append(row)
# %%
allrows.pop(0)
for row in allrows:
    newrow = [float(x) for x in row]
    allrows[allrows.index(row)] = newrow

# %%
allrows = np.array(allrows)
reshufflingtime = np.mean( allrows[:,4])
alignmenttime = np.mean( allrows[:,3])
plt.figure(figsize=(8,6))
plt.bar([0,1],[alignmenttime,reshufflingtime],width=0.5)
plt.ylim([0,30])
plt.xticks([0,1])
ax = plt.gca()
ax.set_xticklabels(["Cross-correlation time","Reshuffling time"])
plt.ylabel("Time [s]")
plt.title("Total processing time per trace")
# %%
