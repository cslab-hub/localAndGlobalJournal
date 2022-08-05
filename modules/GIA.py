import numpy as np
from collections import defaultdict
from modules import helper
from modules import GCR
from modules import LASA
import math
from scipy.interpolate import interp1d

def abstractDataMixGTM(data, rMG, reductionInt, thresholdSet, interpolate = True, useEmbed=False, doFidelity=False):
    limit = 1000
    skipCounter = 0
    
    reduceStrings = ['max','max+','average','average+','median','median+']
    reduceString = reduceStrings[reductionInt]
    reductionWith = reduceString
    
    newX = []
    reduction = []
    
    dataN = np.array(data).squeeze()
    

    
    for index in range(len(dataN)):        
            if(useEmbed):
                X_sax = dataN[index].split(" ")
                vocab = helper.getMapValues(len(set(X_sax)))
                X_ori = [helper.trans(valx, vocab) for valx in np.array(data).squeeze()[index].split(" ")]
            else:
                X_sax = dataN[index]
                X_ori = X_sax 
  
            maxScores = dict()
    
            for lable in rMG.keys():
                maxScores[lable] =  np.sum(np.max(list(rMG[lable][reduceString].values()), axis=0))
                            
                if maxScores[lable] == 0:
                    maxScores[lable]  = -1
            
            _, _, targetLabel, _, _ = GCR.classifyGTM(X_sax, rMG, reduceString, maxScores, 0)
            if(targetLabel == np.nan):
                targetLabel = 0
            borderHeat = np.sum(np.array(list(rMG[targetLabel][reductionWith].values())))
            borderHeat = (borderHeat / np.sum([[a > 0 for a in list(rMG[targetLabel][reductionWith].values())]])) /thresholdSet[0]
            borderHeat2 = borderHeat/thresholdSet[1]

            if doFidelity:
                buffer = borderHeat
                borderHeat = borderHeat2
                borderHeat2 = buffer


            if interpolate:
                fitleredSet = []
                indexSet = []
                avgSet = []
                for h in range(len(X_ori)):
                    score = rMG[targetLabel][reductionWith][float(X_sax[h])][h]
                    if LASA.validataHeat(score, borderHeat, doFidelity):
                        if len(avgSet) != 0:
                            fitleredSet.append(np.median(avgSet))
                            indexSet.append(h - math.ceil(len(avgSet)/2))
                            avgSet = []
                        fitleredSet.append(X_ori[h])
                        indexSet.append(h)
                    elif LASA.validataHeat(score, borderHeat2, doFidelity):
                        avgSet.append(X_ori[h])
                        #avgSet = []
                    elif len(avgSet) != 0:
                        fitleredSet.append(np.median(avgSet))
                        indexSet.append(h - math.ceil(len(avgSet)/2))
                        avgSet = []

                if len(avgSet) != 0:
                    fitleredSet.append(np.median(avgSet))
                    indexSet.append(len(X_ori) - math.ceil(len(avgSet)/2))

                if(len(fitleredSet) < 2):
                    print('doSkip')
                    fitleredSet = []
                    indexSet = []
                    fitleredSet.append(0)
                    fitleredSet.append(0)
                    indexSet.append(0)
                    indexSet.append(len(X_ori))
                    reduction.append(0)
                    skipCounter += 1
                else:
                    reduction.append(1 - len(fitleredSet)/len(X_ori))
                    
                newXTemp = interp1d(indexSet, fitleredSet, bounds_error = False, fill_value = -2)
                newX.append([[x] for x in newXTemp(range(len(X_ori)))])
            
            
            else: 
                print('error')
                #TODO


    newX = np.array(newX, dtype=np.float32)
    print(np.array(newX).shape)
    print(data.shape)
    return newX, reduction, skipCounter