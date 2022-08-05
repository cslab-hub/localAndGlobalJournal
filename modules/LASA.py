import numpy as np
import pandas as pd
from modules import helper
import math
from scipy.interpolate import interp1d
import time
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sktime.datasets import load_osuleaf
from sktime.transformations.panel.shapelets import ContractedShapeletTransform


def validataHeat(value, heat, doFidelity):
    if doFidelity:
        return value <= heat
    else:
        return value > heat

#interpolation with customized combinations
def abstractDataS(data, earlyPredictorZ, order, step1, step2, step3, doMax, thresholdSet, useEmbed = False, takeAvg = True, heatLayer = 0, interpolate = True, doFidelity=False):
    limit = 500
    attentionQ0 = []
    attentionQ1 = []
    attentionQ2 = []

    for bor in range(int(math.ceil(len(data)/limit))):
        attOut = earlyPredictorZ.predict([data[bor*limit:(bor+1)*limit]])
        attentionQ0.extend(attOut[0]) 
        attentionQ1.extend(attOut[1])

        if len(attentionQ2) == 0:
            attentionQ2 = attOut[2]
        else:
            print(np.array(attentionQ2).shape)
            for k in range(len(attentionQ2)):
                
                attentionQ2[k] = np.append(attentionQ2[k], attOut[2][k], 0)
            print(np.array(attentionQ2).shape)
    
    attentionFQ = [np.array(attentionQ0), np.array(attentionQ1), np.array(attentionQ2)]
    
    if(order == 'lh'):
        axis1 = 0
        axis2 = 1 
    elif(order == 'hl'):
        axis1 = 2
        axis2 = 0
    else:
        axis1 = 2
        axis2 = 0  

    print('*********************')
    print(attentionFQ[2].shape)
    attentionFQ[1] = helper.doCombiStep(step1, attentionFQ[2], axis1)
    print(attentionFQ[1].shape)
    attentionFQ[1] = helper.doCombiStep(step2, attentionFQ[1], axis2) 
    print(attentionFQ[1].shape)

    print(attentionFQ[1].shape)
    newX = []
    reduction = []
    skipCounter = 0 
    for index in range(len(attentionFQ[1])):
                   
            if(useEmbed):
                X_sax = np.array(data).squeeze()[index].split(" ")
                vocab = helper.getMapValues(len(set(X_sax)))
                X_ori = [helper.trans(valx, vocab) for valx in np.array(data).squeeze()[index].split(" ")]
            else:
                X_sax = np.array(data).squeeze()[index]
                X_ori = X_sax 
  
            heat = helper.doCombiStep(step3, attentionFQ[1][index], 0) 
    
            #max mit 2 und 3, median mit 1 und 1.5
            #TODO set thresholds!
            if doMax:
                maxHeat = np.max(heat)
                borderHeat = maxHeat/thresholdSet[0]   #/2
                borderHeat2 = maxHeat/thresholdSet[1]#/3
            else:
                maxHeat = np.average(heat)
                borderHeat = maxHeat/thresholdSet[0]
                borderHeat2 = maxHeat/thresholdSet[1]#/1.2

            strongerInterpolation = True
            mediumSkips = 0

            if(doFidelity):  
                bufferHeat = borderHeat
                borderHeat = borderHeat2
                borderHeat2 = bufferHeat
            
            if interpolate:
                fitleredSet = []
                indexSet = []
                avgSet = []
                for h in range(len(heat)):
                    if validataHeat(heat[h], borderHeat, doFidelity):
                        if len(avgSet) > mediumSkips:
                            fitleredSet.append(np.median(avgSet))
                            indexSet.append(h - math.ceil(len(avgSet)/2))
                            avgSet = []
                        fitleredSet.append(X_ori[h])
                        indexSet.append(h)
                    elif validataHeat(heat[h], borderHeat2, doFidelity):
                        #fitleredSet.append([-1e9])
                        avgSet.append(X_ori[h])
                        #avgSet = []
                    elif len(avgSet) > mediumSkips and strongerInterpolation:
                        #fitleredSet.append([-1e9])
                        fitleredSet.append(np.median(avgSet))
                        indexSet.append(h - math.ceil(len(avgSet)/2))

                        avgSet = []
                    #else:
                        #fitleredSet.append([-1e9])

                if len(avgSet) > mediumSkips:
                    fitleredSet.append(np.median(avgSet))
                    indexSet.append(len(heat) - math.ceil(len(avgSet)/2))

                reduction.append(1 - len(fitleredSet)/len(heat))

                if(len(fitleredSet) == 0):
                    skipCounter += 1
                    fitleredSet.append(0)
                    indexSet.append(0)
                    fitleredSet.append(0)
                    indexSet.append(len(heat))
                elif(len(fitleredSet) < 2):
                    skipCounter += 1
                    fitleredSet.append(0)
                    indexSet.append(len(heat))
                newXTemp = interp1d(indexSet, fitleredSet, bounds_error = False, fill_value = -2)
                newX.append([[x] for x in newXTemp(range(len(heat)))])
            
            
            else: 
                fitleredSet = []
                indexSet = []
                avgSet = []
                for h in range(len(heat)):
                    if validataHeat(heat[h], borderHeat, doFidelity):
                        if len(avgSet) != 0:
                            fitleredSet[h - math.ceil(len(avgSet)/2)] = np.median(avgSet)
                            avgSet = []
                        fitleredSet.append(X_ori[h])
                    elif validataHeat(heat[h], borderHeat2, doFidelity):
                        fitleredSet.append(-2)
                        avgSet.append(X_ori[h])
                    elif len(avgSet) != 0:
                        fitleredSet.append(-2)
                        fitleredSet[h - math.ceil(len(avgSet)/2)] = np.median(avgSet)

                        avgSet = []
                    else:
                        fitleredSet.append(-2)
                if len(avgSet) != 0:
                    fitleredSet[len(heat) - math.ceil(len(avgSet)/2)] = np.median(avgSet)
                reduction.append(1 - len([x for x in fitleredSet if x != -2])/len(heat))
                newX.append([[x] for x in fitleredSet])

    newX = np.array(newX, dtype=np.float32)
    print(np.array(newX).shape)
    print(data.shape)

    return newX, reduction, skipCounter



def evaluateShapelets(pipeline, X_test, y_testy, basedata, outOri, outSax):
    results = dict()

    x_test = X_test
    x_test = np.array(x_test).squeeze()

    dfArray = []
    for x in x_test:
        ps = pd.Series(x)
        dfArray.append(ps)
    dftest = pd.DataFrame()
    dftest['dim_0'] = dfArray

    test_x = dftest
    test_y = y_testy

    preds = pipeline.predict(test_x)
    baselinePreds = pipeline.predict(basedata)

    print('Shapelet Results:')
    print('Acc:')
    
    results['accuracy'] = metrics.accuracy_score(y_testy , preds)
    results['precision'] = metrics.precision_score(y_testy, preds, average='macro')
    results['recall'] = metrics.recall_score(y_testy, preds, average='macro')
    results['f1 score'] = metrics.f1_score(y_testy, preds, average='macro')
    results['train predictions'] = baselinePreds
    results['test predictions'] = preds
    correct = sum(preds == test_y)
    results['correct'] = correct
    results['maxCorrect'] = len(test_y)

    
    if len(outOri) > 0:
        oriFidelityTrain = helper.modelFidelity(outOri[2][0], baselinePreds)
        oriFidelityTest = helper.modelFidelity(outOri[3], preds)
    else:
        oriFidelityTrain = -1
        oriFidelityTest = -1
    if len(outSax) > 0:
        saxFidelityTrain = helper.modelFidelity(outSax[2][0], baselinePreds)
        saxFidelityTest = helper.modelFidelity(outSax[3], preds)
    else:
        saxFidelityTrain = -1
        saxFidelityTest = -1

    results['Train Model Fidelity (Ori)'] = oriFidelityTrain
    results['Train Model Fidelity (Sax)'] = saxFidelityTrain
    results['Test Model Fidelity (Ori)'] = oriFidelityTest
    results['Test Model Fidelity (Sax)'] = saxFidelityTest

    print(str(correct) + "/" + str(len(test_y)))
    print(str(results['accuracy']))

    results['count'] = len(pipeline['st'].shapelets)

    print('len:')
    print(len(pipeline['st'].shapelets))
    infoGain = [s.info_gain for s in pipeline['st'].shapelets]
    infoGain = np.array(infoGain)
    results['Avg info gain'] = np.mean(infoGain)
    results['Avg info gain top 5'] = np.mean(infoGain[:5])
    print('avg info gain')
    print(results['Avg info gain'])
    print('avg info gain top 5')
    print(results['Avg info gain top 5'])
    
    lenGain = [s.length for s in pipeline['st'].shapelets]
    lenGain = np.array(lenGain)
    results['Avg len'] = np.mean(lenGain)
    results['Avg len top 5'] = np.mean(lenGain[:5])
    print('avg len')
    print(results['Avg len'])
    print('avg len top 5')
    print(results['Avg len top 5'])

    complexGain = []
    for s in pipeline['st'].shapelets:
        complexGain.append(helper.ce(basedata.iloc[s.series_id, 0][s.start_pos : s.start_pos + s.length].array))
    results['Avg CE'] = np.mean(complexGain)
    results['Avg CE top 5'] = np.mean(complexGain[:5])
    print('avg CE')
    print(results['Avg CE'])
    print('avg CE top 5')
    print(results['Avg CE top 5'])


    return results

def trainShapelets(x_train, y_train, time_contract_in_mins =  2, initial_num_shapelets_per_case = 5, verbose = 2, min_shapelet_length = 3, reduceTrainy = True):
    x_train = np.array(x_train).squeeze()
    if reduceTrainy:
        predictions = np.argmax(y_train,axis=1) +1 
    else:
        predictions = y_train


    dfArray = []
    for x in x_train:
        ps = pd.Series(x)
        dfArray.append(ps)
    dftrain = pd.DataFrame()
    dftrain['dim_0'] = dfArray


    train_x = dftrain
    train_y = predictions


    # How long (in minutes) to extract shapelets for.
    # This is a simple lower-bound initially;
    # once time is up, no further shapelets will be assessed
    #time_contract_in_mins = 2

    # The initial number of shapelet candidates to assess per training series.
    # If all series are visited and time remains on the contract then another
    # pass of the data will occur
    #initial_num_shapelets_per_case = 10

    # Whether or not to print on-going information about shapelet extraction.
    # Useful for demo/debugging
    #verbose = 2

    # example pipeline with 1 minute time limit
    pipeline = Pipeline(
        [
            (
                "st",
                ContractedShapeletTransform(
                    time_contract_in_mins=time_contract_in_mins,
                    num_candidates_to_sample_per_case=initial_num_shapelets_per_case,
    #                max_shapelets_to_store_per_class=1,
                    min_shapelet_length=min_shapelet_length,
                    verbose=False,
                ),
            ),
            ("rf", RandomForestClassifier(n_estimators=100)),
        ]
    )

    start = time.time()
    pipeline.fit(train_x, train_y)
    end_build = time.time()

    return pipeline, train_x