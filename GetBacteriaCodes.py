import numpy as np
import csv
import random
import json
allrows = []
with open('D:\Sergey\FluorocodeMain\AllBact-csv.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        if not any(['Escherichia' in x for x in row]):
            allrows.append(row[0])
        else:
            print(row)
allbactonly = [x for x in allrows if x is not '']
bactdata =random.sample(allbactonly,k=50)
with open('D:\Sergey\FluorocodeMain\BactDatabase.json', 'w') as json_file:
    json.dump(allrows, json_file)

