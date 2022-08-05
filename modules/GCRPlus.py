import numpy as np
from collections import defaultdict
from modules import helper
from modules import mainHelper
from sklearn import metrics
from datetime import datetime
from joblib import Parallel, delayed, parallel_backend
import multiprocessing as mp
import operator


#nestest dict for saves
def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

def getddList():
    return defaultdict(list)

def getddlv2():
    return defaultdict(getddList)

def nested_dict_static():
    return defaultdict(getddlv2)


#create groundwork for all types of GCRs
def makeAttention(attentionQ, x_train1, y_train1, order, step1, step2, num_of_classes, valuesA, doThreshold=False, doFidelity = False, doMax=False, doPenalty=False, threshold = -1, penaltyMode = 'entropy', reducePredictions=True, addOne=True):

    methodStart = datetime.now()

    #predicted lables
    #attentionQ = outSax[9]
    
    #print('stats:')
    #print(num_of_classes)
    #print(len(attentionQ[2]))
    #print(len(attentionQ[2][0]))
    #print(len(attentionQ[2][0][0]))
    #print(len(attentionQ[2][0][0][0]))
    #print(len(attentionQ[2][0][0][0][0]))

    if(order == 'lh'):
        axis1 = 0
        axis2 = 1
    elif(order == 'hl'):
        axis1 = 2
        axis2 = 0
    else:
        axis1 = 0
        axis2 = 1
    

    print("starting combiSteps"+ str(datetime.now() - methodStart))
    attentionQ[1] = helper.doCombiStep(step1, attentionQ[2], axis1)
    attentionQ[1] = helper.doCombiStep(step2, attentionQ[1], axis2) 
    print("finished combiSteps" + str(datetime.now() - methodStart))    

    #true lables
    if reducePredictions:
        if addOne:
            predictions = np.argmax(y_train1,axis=1) +1
        else:
            predictions = np.argmax(y_train1,axis=1) 
    else:
        predictions = y_train1

    #position counter
    rM = nested_dict_static()
    #attention sum at each point
    rMS = nested_dict_static()
    #relative average at each point + more side combinations
    rMA = nested_dict_static()
    #penalty buffer
    rMP = nested_dict_static()
    
    labelSet = set(predictions)
    laCount = dict()
    entropyDic = dict()
    entropyDicRelativeMult = dict()
    entropyDicRelativeDiv = dict()
    countingConstant = (1 * (num_of_classes + 1))
    if doPenalty:
        for la in labelSet:
            laCount[la] = predictions.tolist().count(la)
            part = laCount[la] / len(predictions)
            entropy = -(part * np.log(part))
            entropyDic[la] = entropy
            entropyDicRelativeMult[la] = entropy * num_of_classes
            entropyDicRelativeDiv[la] = (1+num_of_classes)/entropy
            #entropyDicRelativeMult[la] = entropy# * num_of_classes
            #entropyDicRelativeDiv[la] = (1+num_of_classes)/entropy
            #entropyDicRelativeMult[la] = entropy#/ num_of_classes
            #entropyDicRelativeDiv[la] = (1+num_of_classes) * entropy
    
    data_att = attentionQ[1][0]

    print("starting default dict" + str(datetime.now() - methodStart))

    print(data_att.shape)
    if addOne:
        endPoint = num_of_classes+1
    else:
        endPoint = num_of_classes
    for lable in range(1,endPoint):
        

            for toL in valuesA:

                for fromL in valuesA:
                    if(len(rM[lable][fromL][toL]) is 0):
                        rM[lable][fromL][toL] = np.zeros((len(data_att), len(data_att[0])))
                        rMS[lable][fromL][toL] = np.zeros((len(data_att), len(data_att[0])))
                        rMP[str(lable)+"pen"][fromL][toL] = np.zeros((len(data_att), len(data_att[0])))
                        rMA[lable][fromL][toL] = np.zeros((len(data_att), len(data_att[0])))

                if len(rMA[lable]['x'][toL]) is 0:
                    rMA[lable]['x'][toL] = np.zeros((len(data_att), len(data_att[0])))
                    rMP[str(lable)+"pen"]['x'][toL] = np.zeros((len(data_att), len(data_att[0])))

                if len(rMA[lable]['xAvg'][toL]) is 0:
                    rMA[lable]['xAvg'][toL] = np.zeros((len(data_att), len(data_att[0])))

    #put together all train attention to from symbol x to symbol y representation
    z = 0
    x_train1s = np.array(x_train1).squeeze()

    if doThreshold:
        if doMax:
            maxHeat = np.max(data_att)
            borderHeat = maxHeat / threshold
        else:
            maxHeat = np.average(data_att)
            borderHeat = maxHeat/threshold# maxHeat/1.6
        # if predictions[index] == 1:
        #     borderHeat = maxHeat/1.7
        # if predictions[index] == 2:
        #     borderHeat = 0 
        # borderHeat2 = maxHeat/1.65
    else:
        borderHeat = 0

    #alternative divide score part1
    #avgScore = dict()
    #for la in labelSet:
    #    avgScore[la] = []
    
    print("starting attention adding" + str(datetime.now() - methodStart))
    print(attentionQ[1].shape)
    print(x_train1s.shape)
    print(predictions.shape)
    print(borderHeat)
    print(data_att.shape)
    print('##############################################')

    #not parallel
    #for index in range(len(attentionQ[1])):
    #    sumAttention(index, attentionQ[1][index], x_train1s[index], borderHeat, doFidelity, rM, rMP, rMA, rMS, predictions[index], doPenalty, penaltyMode)

    Parallel(n_jobs=14, require='sharedmem', prefer="threads")(delayed(sumAttention)(index, attentionQ[1][index], x_train1s[index], borderHeat, doFidelity, rM, rMP, rMA, rMS, predictions[index], doPenalty, penaltyMode, avgScore) for index in range(len(attentionQ[1]))) #
    
    #not parallel
    #for index in range(len(attentionQ[1])):
    #    sumAttention(index, attentionQ, x_train1s, borderHeat, doFidelity, rM, rMP, rMA, rMS, predictions, doPenalty, penaltyMode)

    print("starting making relative attention" + str(datetime.now() - methodStart))

    #not parallel
    #for lable in rMA.keys():
    #    relativeAttentionMaking(lable, valuesA, data_att, rM, rMS, rMA, rMP, doPenalty, penaltyMode, entropyDicRelativeDiv, entropyDicRelativeMult, laCount, countingConstant)
    
    Parallel(n_jobs=14, require='sharedmem', prefer="threads")(delayed(relativeAttentionMaking)(lable, valuesA, data_att, rM, rMS, rMA, rMP, doPenalty, penaltyMode, entropyDicRelativeDiv, entropyDicRelativeMult, laCount, countingConstant) for lable in rMA.keys())

    #alternative divide score part2
    #for la in labelSet:
    #    avgScore[la] = np.median(avgScore[la])
    #print('done' + str(datetime.now() - methodStart))

    return rMA, rMS, rM#, avgScore

# relativate the summed scores
def relativeAttentionMaking(lable, valuesA, data_att, rM, rMS, rMA, rMP, doPenalty, penaltyMode, entropyDicRelativeDiv, entropyDicRelativeMult, laCount, countingConstant):
    #minV, maxV = getMinMax(rMS, lable)
    for toL in valuesA:

        for fromL in valuesA:
            #rMS[lable][fromL][toL] = (rMS[lable][fromL][toL] - minV) / (maxV - minV)
            #X_scaled = X_std * (1 - 0) + 0
            for i in range(len(data_att)):
                for j in range(len(data_att[i])): 
                    #FCAM r. average
                    if rM[lable][fromL][toL][i][j] > 0:
                        aAdder = 0
                        if doPenalty:
                            if(penaltyMode == "entropy"):
                                entropyRelativeDiv = entropyDicRelativeDiv[lable]
                                divver = rMP[str(lable)+"pen"][fromL][toL][i][j] / float(rM[lable][fromL][toL][i][j])
                                aAdder = entropyRelativeDiv * divver
                                adder = entropyRelativeDiv * rMP[str(lable)+"pen"][fromL][toL][i][j]

                                entropyRelativeMult = entropyDicRelativeMult[lable]
                                aSubber = divver / entropyRelativeMult # divver * entropyRelativeMult when mult is only : entropy
                                subber = rMP[str(lable)+"pen"][fromL][toL][i][j] / entropyRelativeMult

                                
                                rMS[lable][fromL][toL][i][j] += adder
                                rMA[lable]['x'][toL][i][j] += adder
                                for innerLable in rMA.keys():
                                    rMS[innerLable][fromL][toL][i][j] -= subber
                                    rMA[innerLable][fromL][toL][i][j] -= aSubber
                                    rMA[innerLable]['x'][toL][i][j] -= subber
                                    rMA[innerLable]['xAvg'][toL][i][j] -= aSubber
                                    
                            else:
                                divPart = rMP[str(lable)+"pen"][fromL][toL][i][j]/ laCount[lable]
                                adder =  countingConstant * divPart
                                aAdder = adder/float(rM[lable][fromL][toL][i][j])

                                aSubber = divPart / float(rM[lable][fromL][toL][i][j])
                                subber = divPart

                                rMS[lable][fromL][toL][i][j] += adder
                                rMA[lable]['x'][toL][i][j] += adder
                                for innerLable in rMA.keys():
                                    rMS[innerLable][fromL][toL][i][j] -= subber
                                    rMA[innerLable][fromL][toL][i][j] -= aSubber
                                    rMA[innerLable]['x'][toL][i][j] -= subber
                                    rMA[innerLable]['xAvg'][toL][i][j] -= aSubber
                        else:
                            aAdder = rMS[lable][fromL][toL][i][j] / float(rM[lable][fromL][toL][i][j])
                        
                        rMA[lable][fromL][toL][i][j] += aAdder
                        rMA[lable]['xAvg'][toL][i][j] += aAdder

                    

                        

        #GTM max of sum                
        rMA[lable]['max'][toL] = np.max(rMA[lable]['x'][toL], axis=0) 
        #GTM average of sum         
        rMA[lable]['average'][toL] = np.mean(rMA[lable]['x'][toL], axis=0) 
        #GTM median of sum         
        rMA[lable]['median'][toL] = np.median(rMA[lable]['x'][toL], axis=0) 
        #GTM max of r.average          
        rMA[lable]['max+'][toL] = np.max(rMA[lable]['xAvg'][toL], axis=0)  
        #GTM average of r.average          
        rMA[lable]['average+'][toL] = np.mean(rMA[lable]['xAvg'][toL], axis=0)
        #GTM median of r.average         
        rMA[lable]['median+'][toL] = np.median(rMA[lable]['xAvg'][toL], axis=0) 

# sum attention values together into gcr format
def sumAttention(index, data_att, data_word, borderHeat, doFidelity, rM, rMP, rMA, rMS, predictionsI, doPenalty, penaltyMode, avgScore, retry = 0):

    def validataHeat(value, heat, doFidelity):
        if doFidelity:
            return value <= heat
        else:
            return value > heat


    X_ori = data_word

    avgScore[predictionsI].append(np.sum(data_att))
    for i in range(len(data_att)):
        for j in range(len(data_att[i])):
            if data_att[i][j] != 0 and validataHeat(data_att[i][j], borderHeat, doFidelity):
                label = predictionsI
                rM[label][X_ori[i]][X_ori[j]][i][j] += 1 
                
                if doPenalty:

                    if(penaltyMode == 'entropy'):
                        rMP[str(label)+"pen"][X_ori[i]][X_ori[j]][i][j] += data_att[i][j]

                    else:
                        rMP[str(label)+"pen"][X_ori[i]][X_ori[j]][i][j] += data_att[i][j]

                else:
                    rMS[label][X_ori[i]][X_ori[j]][i][j] += data_att[i][j]
                    rMA[label]['x'][X_ori[j]][i][j] += data_att[i][j]
            #if data_att[i][j] > borderHeat2:
            #    #sum FCAM
            #    rMS[predictions[index]][X_ori[i]][X_ori[j]][i][j] += data_att[i][j] / 1.5
            #    #CRCAM Sum
            #    rMA[predictions[index]]['x'][X_ori[j]][i][j] += data_att[i][j] / 1.5
            
        

# sum class scores
def sumLabelScores(trial,label, indList, rMG, rM):
    lableScore = 0
    indListDel = list(range(len(trial)))
    l1 = rMG[label]

    for fromVi in indList:
        fromV = float(trial[fromVi])
        if rM[label][fromV][fromV][fromVi][fromVi] == 0:
            indListDel.remove(fromVi)

    for fromVi in indListDel:
        fromV = float(trial[fromVi])
        l2 = l1[fromV]
        for toVi in indListDel:
            toV = float(trial[toVi])
            value = l2[toV][fromVi][toVi]
            lableScore += value
    return (lableScore, label)

# sum class scores for penalty GCRs
def sumLabelScoresPenalty(trial, label, indList, rMG):
    lableScore = 0
    l1 = rMG[label]

    for fromVi in indList:
        fromV = float(trial[fromVi])
        l2 = l1[fromV]
        for toVi in indList:
            toV = float(trial[toVi])
            value = l2[toV][fromVi][toVi]
            lableScore += value
    return (lableScore, label)

#classify using a GCR
def doFullCLassify(trial, ylabel, rMG, maxScores, rM, doPenalty=False):
    lableScores = dict()
    indList = range(len(trial))
    if doPenalty:
        # not parallel
        #answers = []    
        #for lable in rMG.keys():
        #    answers.append(sumLabelScoresPenalty(trial,lable, indList, rMG))

        answers = Parallel(n_jobs=7, prefer="threads")(delayed(sumLabelScoresPenalty)(trial,lable, indList, rMG) for lable in rMG.keys())

    else:
        answers = Parallel(n_jobs=7, prefer="threads")(delayed(sumLabelScores)(trial,lable, indList, rMG, rM) for lable in rMG.keys())
    
    for lableScore, label in answers:
        lableScores[label] = lableScore

    #get final score
    for lable in rMG.keys():
        if maxScores[lable] > 0:
            lableScores[lable] = lableScores[lable]/maxScores[lable]

    biggestLable = list(lableScores.keys())[np.argmax(list(lableScores.values()))]
    biggestValue = lableScores[biggestLable]
    boolResult = biggestLable == ylabel

    return lableScores, boolResult, biggestLable, biggestValue, ylabel


def calcFCAMMaxScore(rMG, inputKeys):
    #sum maximum score
    methodStart = datetime.now()
    maxScores = dict()
    print("starting maxing FCAM" + str(datetime.now() - methodStart))
    for lable in rMG.keys():
        maxis = []
        for fromV in inputKeys:
            #print('starting maxing ' + str(lable) + ' progress: ' + str(fromV) + '/' + str(len(inputKeys)))
            maxis.append(np.max(list(rMG[lable][float(fromV)].values()), axis=0))
        if len(maxis) is not 0:
            maxScore =  np.sum(np.max(maxis, axis=0))
            if maxScore == 0:
                maxScores[lable]  = -1
            else:
                maxScores[lable]  = maxScore
        else:
            maxScores[lable] = -1

    print('done summing max FCAM' + str(datetime.now() - methodStart))
    return maxScores

#validate the full coherence matrices 
def classFullAtt(rMG, x_test, y_testy, maxScores, rM, doPenalty=False):
    methodStart = datetime.now()
    results = []
    predictResults = []
    biggestScores = []
    allLableScores = []

    print("starting classify FCAM" + str(datetime.now() - methodStart))


    #sum normal score
    # create a list to keep all processes
    processes = []
    
    # create a list to keep connections
    parent_connections = []
    #tupels = list(zip(range(len(x_test[0])), range(len(x_test[0]))))

    print('processes start FCAM' + str(datetime.now() - methodStart))

    answers = []
    #with parallel_backend("loky", inner_max_num_threads=2):
    #answers = Parallel(n_jobs=14, prefer="threads")(delayed(doFullCLassify)(x_test[ti], y_testy[ti], rMG, maxScores) for ti in range(len(x_test)))
    for ti in range(len(x_test)):
        answers.append(doFullCLassify(x_test[ti], y_testy[ti], rMG, maxScores, rM, doPenalty=doPenalty))

    #for ti in range(len(x_test)):
    #    trial = x_test[ti]
    #    #print('starting trial ' + str(ti) + '/' + str(len(x_test)))
    #    parent_conn, child_conn = mp.Pipe()
    #    parent_connections.append(parent_conn)
        
    #    process = mp.Process(target=doFullCLassify, args=(trial, y_testy[ti], tupels, rMG, maxScores, child_conn))
    #    processes.append(process)

    print('processes end FCAM' + str(datetime.now() - methodStart))
        
    # start all processes
    #for process in processes:
    #    process.start()
        
    # make sure that all processes have finished
    #for process in processes:
    #    process.join()
        
    asynLabels = []
    for ans in answers:
        #ans = parent_connection.recv()
        results.append(ans[1])
        predictResults.append(ans[2])
        biggestScores.append(ans[3])
        allLableScores.append(ans[0])
        asynLabels.append(ans[4])

    print('start results FCAM' + str(datetime.now() - methodStart))

    print("FCAM results :" + str(sum(results)/len(results)))
    acc = metrics.accuracy_score(predictResults, asynLabels)
    predicsion = metrics.precision_score(predictResults, asynLabels, average='macro')
    recall = metrics.recall_score(predictResults, asynLabels, average='macro')
    f1= metrics.f1_score(predictResults, asynLabels, average='macro')

    confidenceAcc = helper.confidenceGCR(biggestScores, results)

    print('done results FCAM' + str(datetime.now() - methodStart))

    return [acc, predicsion, recall, f1], [allLableScores, biggestScores], results, predictResults, confidenceAcc

def doXCLassify(trial, ylabel, rMG, maxScores, key, rM):
    lableScores = dict()
    for lable in rMG.keys():
        lableScores[lable] = 0
        l1 = rMG[lable][key]
    #for fromVi in range(len(trial)):
    ##    for toVi in range(len(trial)):
    #        toV = trial[toVi]
    #        for lable in rMG.keys():
    #            lableScores[lable] += rMG[lable][key][float(toV)][fromVi][toVi]

        #OLD CRCAM!!!
        #for toVi in range(len(trial)):
        #    toV = trial[toVi]
        #    lableScores[lable] += np.sum(l1[float(toV)], axis=0)[toVi]
        #NEW SMART CRCAM
        for toVi in range(len(trial)):
            toV = trial[toVi]
            lableScores[lable] += l1[float(toV)][toVi] 

    #get final score
    for lable in rMG.keys():
        #if maxScores[lable] > 0 and lableScores[lable] > 0:
        lableScores[lable] = lableScores[lable]/maxScores[lable]

    biggestLable = list(lableScores.keys())[np.argmax(list(lableScores.values()))]
    biggestValue = lableScores[biggestLable]
    boolResult = biggestLable == ylabel

    return lableScores, boolResult, biggestLable, biggestValue, ylabel
    #conn.send((lableScores, boolResult, biggestLable, biggestValue, ylabel))
    #conn.close() 

def calcCRCAMMaxScore(rMA, key):
    #sum max score
    methodStart = datetime.now()
    print('start summing max CRCAM' + str(datetime.now() - methodStart))  
    maxScores = dict()
    for lable in rMA.keys():
        maxScore =  np.sum(np.max(list(rMA[lable][key].values()), axis=0))
        if maxScore == 0:
            maxScores[lable]  = -1
        else:
            maxScores[lable]  = maxScore
    print('done summing max CRCAM' + str(datetime.now() - methodStart))
    return maxScores

def calcCRCAMMaxScoreNew(rMA, key):
    #sum max score
    methodStart = datetime.now()
    print('start summing max CRCAM' + str(datetime.now() - methodStart))  
    maxScores = dict()
    for lable in rMA.keys():
        maxScore =  np.sum(np.max(np.sum(list(rMA[lable][key].values()), axis=1), axis=0))
        if maxScore == 0:
            maxScores[lable]  = -1
        else:
            maxScores[lable]  = maxScore
    print('done summing max CRCAM' + str(datetime.now() - methodStart))
    return maxScores

#validate the column reduced coherence matrices 
def xAttentionMatch(rMA, x_test, y_testy, key, maxScores, rM):    
    methodStart = datetime.now()    
    results = []
    predictResults = []
    biggestScores = []
    allLableScores = []

    print('processes started CRCAM' + str(datetime.now() - methodStart))
    
    answers = []
    for ti in range(len(x_test)):
        answers.append(doXCLassify(x_test[ti], y_testy[ti], rMA, maxScores, key, rM))

    print('processes ended CRCAM' + str(datetime.now() - methodStart))
        
    asynLabels = []
    for ans in answers:
        results.append(ans[1])
        predictResults.append(ans[2])
        biggestScores.append(ans[3])
        allLableScores.append(ans[0])
        asynLabels.append(ans[4])

    print('start results CRCAM' + str(datetime.now() - methodStart))

    print('CRCAM results: ' + str(sum(results)/len(results)))
    acc = metrics.accuracy_score(predictResults, asynLabels)
    predicsion = metrics.precision_score(predictResults, asynLabels, average='macro')
    recall = metrics.recall_score(predictResults, asynLabels, average='macro')
    f1= metrics.f1_score(predictResults, asynLabels, average='macro')

    confidenceAcc = helper.confidenceGCR(biggestScores, results)

    return [acc, predicsion, recall, f1], [allLableScores, biggestScores], results, predictResults, confidenceAcc

#validate the GTM
def calcFullAbstractAttention(rMA, reductionInt, x_test, y_testy, divScore = None):
    methodStart = datetime.now() 
    results = []
    predictResults = []
    biggestScores = []
    allLableScores = []
    
    #all possible implemented reductions
    reduceStrings = ['max','max+','average','average+','median','median+']
    reduceString = reduceStrings[reductionInt]
    print("starting classify GTM " + reduceString  + " " +str(datetime.now() - methodStart))

    if divScore == None:
        maxScores = dict()
        
        for lable in rMA.keys():
            maxScores[lable] =  np.sum(np.max(list(rMA[lable][reduceString].values()), axis=0))
                        
            if maxScores[lable] == 0:
                maxScores[lable]  = -1
    else:
        maxScores = divScore

    print('processes start GTM' + str(datetime.now() - methodStart))
    answers = []
    for ti in range(len(x_test)):
        answers.append(classifyGTM(x_test[ti], y_testy[ti], rMA, maxScores, reduceString))

    print('processes end GTM' + str(datetime.now() - methodStart))
        
    asynLabels = []
    for ans in answers:
        results.append(ans[1])
        predictResults.append(ans[2])
        biggestScores.append(ans[3])
        allLableScores.append(ans[0])
        asynLabels.append(ans[4])

    print('results GTM' + str(datetime.now() - methodStart))
    print("GTM " + reduceString + " results: " + str(sum(results)/len(results)))
    acc = metrics.accuracy_score(predictResults, asynLabels)
    predicsion = metrics.precision_score(predictResults, asynLabels, average='macro')
    recall = metrics.recall_score(predictResults, asynLabels, average='macro')
    f1= metrics.f1_score(predictResults, asynLabels, average='macro')

    confidenceAcc = helper.confidenceGCR(biggestScores, results)

    return [acc, predicsion, recall, f1], [allLableScores, biggestScores], results, predictResults, confidenceAcc


def classifyGTM(trial, ylabel, rMG, maxScores, reduceString):
    methodStart = datetime.now() 
    lableScores = dict()

    for lable in rMG.keys():
        lableScores[lable] = 0
    
    for toVi in range(len(trial)):
        toV = trial[toVi]

        for lable in rMG.keys():
            lableScores[lable] += rMG[lable][reduceString][float(toV)][toVi] 

    #get final score
    for lable in rMG.keys():
        lableScores[lable] = lableScores[lable]/maxScores[lable]

    #classification
    biggestLable = list(lableScores.keys())[np.argmax(list(lableScores.values()))]
    biggestValue = lableScores[biggestLable]
    boolResult = biggestLable == ylabel

    return lableScores, boolResult, biggestLable, biggestValue, ylabel

def validataHeat(value, heat, doFidelity):
    if doFidelity:
        return value <= heat
    else:
        return value > heat