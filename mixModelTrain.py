from sacred import Experiment
import numpy as np
import seml
import os
import random
import warnings

from modules import LASA
from modules import helper
from modules import mainHelper
from modules import GCRPlus
from modules import dataset_selecter as ds
from modules import modelCreator

from datetime import datetime


import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    #init before the experiment!
    @ex.capture(prefix="init")
    def baseInit(self, nrFolds: int, patience: int, seed_value: int, gtmAbstractions, symbolCount: int):
        self.seed_value = seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)# 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
        tf.random.set_seed(seed_value)
        np.random.RandomState(seed_value)

        np.random.seed(seed_value)

        context.set_global_seed(seed_value)
        ops.get_default_graph().seed = seed_value

        #pip install tensorflow-determinism needed
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        np.random.seed(seed_value)

        #save some variables for later
        self.symbolCount = symbolCount

        self.valuesA = helper.getMapValues(symbolCount)
        self.kf = StratifiedKFold(nrFolds, shuffle=True, random_state=seed_value)
        self.earlystop = EarlyStopping(monitor= 'val_loss', min_delta=0 , patience=patience, verbose=0, mode='auto')
        self.fold = 0
        self.nrFolds = nrFolds
        self.seed_value = seed_value        
        self.gtmAbstractions = gtmAbstractions

        #init gpu
        physical_devices = tf.config.list_physical_devices('GPU') 
        for gpu_instance in physical_devices: 
            tf.config.experimental.set_memory_growth(gpu_instance, True)


    # Load the dataset
    @ex.capture(prefix="data")
    def init_dataset(self, dataset: str, number: int, takename: bool):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
        self.number = number
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_trainy, self.y_testy, self.seqSize, self.dataName, self.num_of_classes = ds.datasetSelector(dataset, self.seed_value, number, takeName=takename)


    #all inits
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.baseInit()
        self.init_dataset()

    #methods to save the results into the fullResults dict

    def fillOutDicWithNNOutFull(self, abstractionString, configString, fullResults, inputDict):
        if abstractionString not in fullResults.keys():
            fullResults[abstractionString] = dict()
        if configString not in fullResults[abstractionString].keys():
            fullResults[abstractionString][configString] = []
        fullResults[abstractionString][configString].append(inputDict)

    def fillOutDicWithNNOut(self, abstractionString, configString, fullResults, outData, outOri, outSax):
        inputDict = mainHelper.fillOutDicWithNNOutSmall(outData, outOri, outSax)
        self.fillOutDicWithNNOutFull(abstractionString, configString, fullResults, inputDict)
        
        #DONE model Fidelity berechnen! -> ggf später sobald ich alle daten habe? Weil dann müsste alles da sein!!!
        #TODO LAAM consistency of one trial in two different models?!?
        #DONE Remove uncertain classification for the GCR and see how well it does perform (on train and test data)fillOutDicWithGCRSmall
        # - wann ist es uncertain? wenn kein großer unterschied zu anderen classen existiert? Oder wenn der overall wert niedrig ist?
    
    def fillOutDicWithGCRFull(self, abstractionString, configString, fullResults, inputDict):
        if abstractionString not in fullResults.keys():
            fullResults[abstractionString] = dict()
        if configString not in fullResults[abstractionString].keys():
            fullResults[abstractionString][configString] = []
        fullResults[abstractionString][configString].append(inputDict)
        #if abstractionString not in fullResults.keys():
        #    fullResults[abstractionString] = []
        #if len(fullResults[abstractionString]) < fold:
        #fullResults[abstractionString].append(inputDict)
        #else:
        #    fullResults[abstractionString][fold-1] = {**fullResults[abstractionString][fold-1], **inputDict}    

    def fillOutDicWithGCR(self, abstractionString, configString, fullResults, outData, subContext, oriPred, saxPred):
        inputDict = mainHelper.fillOutDicWithGCRSmall(outData, subContext, oriPred, saxPred)

        self.fillOutDicWithGCRFull(abstractionString, configString, fullResults, inputDict)
        #TODO fixen parameter mapping done?

    def printTime(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)


    # one experiment run with a certain config set. MOST OF THE IMPORTANT STUFF IS DONE HERE!!!!!!!
    @ex.capture(prefix="model")
    def trainExperiment(self, numEpochs: int, batchSize: int, useEmbed: bool, calcOri: bool, doSymbolify: bool, useSaves: bool, calcComplexity: bool, multiVariant: bool, skipDebugSaves: bool, 
        dropeOutRate: float, takeAvg: bool, heatLayer: int, doLASA: bool, doBaseLasa: bool, doShapeletsBase: bool, doShapeletsLASA: bool, doGCR: bool, doOriginalGCR: bool, doThresholdGCR: bool, doPenaltyGCR: bool, doGIA: bool, dmodel: int, dff: int, localMaxThresholds, 
        localAvgThresholds, globalThresholds, giathreshold, penalityModes, shapeletLenghts, initial_num_shapelets_per_case: int, time_contract_in_mins: int, header: int, numOfAttentionLayers: int, steps, layerCombis, limit: int, doFidelity: bool): #, foldModel: int):

        print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
        print('Dataname:')
        print(self.dataName)
        self.printTime()
        warnings.filterwarnings('ignore')   

        fullResults = dict()
        consistancyDict = dict()

        
        #wname = modelCreator.getWeightName(self.dataName, self.nrFolds, self.symbolCount, numOfAttentionLayers, "results", header, learning = False, results = True, usedp=True, doHeaders=False, resultsPath = 'results')
        wname = modelCreator.getWeightName(self.dataName, self.nrFolds, self.symbolCount, numOfAttentionLayers, "results", header, learning = False, results = True, resultsPath = 'ppresults')

        #wname = wname.replace(":", "\uf03a")

        #if you know which configs already finished you can activate this
        if False: # not os.path.isfile(wname + '.pkl'):
            fullResults["Error"] = "dataset " + self.dataName + " not included: " + str(self.seqSize) + "; name: " + wname
            print('Not included ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " not included: " + str(self.seqSize) + "; name: " + wname)

            return "dataset " + self.dataName + " not included " + str(self.seqSize)  + "; name: " + wname #fullResults
        
        #don't recalculate already finished experiments
        wname = modelCreator.getWeightName(self.dataName, self.nrFolds, self.symbolCount, numOfAttentionLayers, "results", header, learning = False, results = True)
        if os.path.isfile(wname + '.pkl'):
            fullResults["Error"] = "dataset " + self.dataName + " already done: " + str(self.seqSize) + "; name: " + wname
            print('Already Done ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " already done: " + str(self.seqSize) + "; name: " + wname)
        
            return "dataset " + self.dataName + "already done: " + str(self.seqSize)  + "; name: " + wname #fullResults

        #limit the lenght of the data
        if self.seqSize > limit:
            fullResults["Error"] = "dataset " + self.dataName + " to big: " + str(self.seqSize)
            print('TO LONGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " to big: " + str(self.seqSize))
            return "dataset " + self.dataName + " to big: " + str(self.seqSize) #fullResults

        vocab = set(self.y_testy)

        consistancyIndicies = []
        consistancyLabels = []
        

        #safe some indecies for the consistency calculation
        for v in vocab:
            poi = np.where(self.y_testy == v)[0]
            times = [3, len(poi)]
            samples = random.sample(list(poi), np.min(times))
            for i in range(len(samples)):
                consistancyLabels.append(v)
            consistancyIndicies = consistancyIndicies + samples

        #consistancyIndicies = random.sample(range(1, len(self.X_test)), 10)

        # k fold train loop
        for train, test in self.kf.split(self.X_train, self.y_trainy):
            self.fold+=1

            fullResults["val indices fold: " + str(self.fold)] = test
            
            #preprocess data
            x_train1 = self.X_train[train]
            x_val = self.X_train[test]
            y_train1 = self.y_train[train]
            y_trainy2 = self.y_trainy[train]
            y_val = self.y_train[test]
            
            x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy, y_testy = modelCreator.preprocessData(x_train1, x_val, self.X_test, y_train1, y_val, self.y_test, y_trainy2, self.y_testy, self.fold, self.symbolCount, self.dataName, useEmbed = useEmbed, useSaves = useSaves, doSymbolify = doSymbolify, multiVariant=multiVariant)

            # calc original model
            abstractionString = "Original"
            if(calcOri):
                
                outOri = modelCreator.doAbstractedTraining(X_train_ori, X_val_ori, X_test_ori, y_train1, y_val, y_testy, batchSize, self.seed_value, self.num_of_classes, self.dataName, self.fold, self.symbolCount, numEpochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves=skipDebugSaves, rMA = None, useEmbed = useEmbed, earlystop = self.earlystop, useSaves=useSaves, 
                abstractionType=abstractionString, takeAvg = takeAvg, heatLayer=heatLayer, rate=dropeOutRate, calcComplexity=calcComplexity, abstraction = 0, reductionInt = -1, thresholdSet=None, order=None, step1=None, step2=None, step3=None, doMax=False, earlyPredictorZ = None)
            else:
                #better do not use this, else later calculations can crash!
                outOri = []
            self.fillOutDicWithNNOut(abstractionString, "", fullResults, outOri, outOri, [])
            self.printTime()

            # calc SAX model
            abstractionString = "SAX"
            outSax = modelCreator.doAbstractedTraining(x_train1, x_val, x_test, y_train1, y_val, y_testy, batchSize, self.seed_value, self.num_of_classes, self.dataName, self.fold, self.symbolCount, numEpochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves=skipDebugSaves, rMA = None, useEmbed = useEmbed, earlystop = self.earlystop, useSaves=useSaves, 
                abstractionType=abstractionString, takeAvg = takeAvg, heatLayer=heatLayer, rate=dropeOutRate, calcComplexity=calcComplexity, abstraction = 0, reductionInt = -1, thresholdSet=None, order=None, step1=None, step2=None, step3=None, doMax=False, earlyPredictorZ = None)
            self.printTime()
            self.fillOutDicWithNNOut(abstractionString, "", fullResults, outSax, outOri, outSax)   

            predictions = np.argmax(y_train1,axis=1) +1

            #shapelet calculation on Ori and SAX data
            if doShapeletsBase:
                for sLen in shapeletLenghts:    
                    if(sLen <= 1):
                        sLen = int(len(x_train1[0]) * sLen)
                    if(calcOri):
                        abstractionString = 'Original Shapelets '#  + str(sLen) #+ ' ' + str(self.fold)
                        configString = "slen: " + str(sLen)
                        print(abstractionString + configString)
                        self.printTime()
                        try:
                            #oriShapelets, baseline = LASA.trainShapelets(outOri[6], y_train1, time_contract_in_mins =  time_contract_in_mins, initial_num_shapelets_per_case = initial_num_shapelets_per_case, verbose = 2, min_shapelet_length = sLen)
                            oriShapelets, baseline = LASA.trainShapelets(outOri[6], np.array(outSax[2][0]), time_contract_in_mins =  time_contract_in_mins, initial_num_shapelets_per_case = initial_num_shapelets_per_case, verbose = 2, min_shapelet_length = sLen, reduceTrainy=False)

                            #lasaSOriOut = LASA.evaluateShapelets(oriShapelets, X_test_ori, y_testy, baseline, outOri, outSax) 
                            lasaSOriOut = LASA.evaluateShapelets(oriShapelets, X_test_ori, np.array(outSax[3]), baseline, outOri, outSax) 
                            
                            if abstractionString not in fullResults.keys():
                                fullResults[abstractionString] = dict()
                            if configString not in fullResults[abstractionString].keys():
                                fullResults[abstractionString][configString] = []
                            fullResults[abstractionString][configString].append(lasaSOriOut)
                        except Exception:
                            if abstractionString not in fullResults.keys():
                                fullResults[abstractionString] = dict()
                            if configString not in fullResults[abstractionString].keys():
                                fullResults[abstractionString][configString] = []
                            fullResults[abstractionString][configString].append("Exception!")

                    abstractionString = 'SAX Shapelets '#  + str(sLen) #+ ' ' + str(self.fold)
                    configString = "slen: " + str(sLen)
                    print(abstractionString + configString)
                    self.printTime()
                    try:
                        #saxShapelets, baseline = LASA.trainShapelets(outSax[6], y_train1, time_contract_in_mins =  time_contract_in_mins, initial_num_shapelets_per_case = initial_num_shapelets_per_case, verbose = 2, min_shapelet_length = sLen)
                        saxShapelets, baseline = LASA.trainShapelets(outSax[6], np.array(outSax[2][0]), time_contract_in_mins =  time_contract_in_mins, initial_num_shapelets_per_case = initial_num_shapelets_per_case, verbose = 2, min_shapelet_length = sLen, reduceTrainy=False)

                        #lasaSSAXOut = LASA.evaluateShapelets(saxShapelets, x_test, y_testy, baseline, outOri, outSax) 
                        lasaSSAXOut = LASA.evaluateShapelets(saxShapelets, x_test, np.array(outSax[3]), baseline, outOri, outSax) 

                        if abstractionString not in fullResults.keys():
                            fullResults[abstractionString] = dict()
                        if configString not in fullResults[abstractionString].keys():
                            fullResults[abstractionString][configString] = []
                        fullResults[abstractionString][configString].append(lasaSSAXOut)
                    except Exception:
                        if abstractionString not in fullResults.keys():
                            fullResults[abstractionString] = dict()
                        if configString not in fullResults[abstractionString].keys():
                            fullResults[abstractionString][configString] = []
                        fullResults[abstractionString][configString].append("Exception!")

            #LASA calculations for all LAAV combies and thresholds
            if doLASA:   
                for order in layerCombis:
                    for step1 in steps:
                        for step2 in steps:
                            earlyPredictor = outSax[-8]
                            key = order+step1+step2
                            if key not in consistancyDict.keys():
                                consistancyDict[key] = []
                            smallX = np.array(x_test)[consistancyIndicies]
                            consistancyDict[key].append(helper.collectLAAMs(earlyPredictor, smallX, order, step1, step2))
                            
                            for step3 in steps:
                                combination = order +'-' +step1 +'-' +step2 +'-' +step3
                                

                                if doBaseLasa:

                                    for tSet in localAvgThresholds:
                                        configString = 'Combination: ' + combination + "; tSet: "+ str(tSet[0]) + ',' + str(tSet[1])
                                        doMax = False
                                        abstractionString = "LASA Avg Inter "# + combination+ ' ' + str(tSet[0]) + ',' + str(tSet[1])  #+ ' ' + str(self.fold)
                                        
                                        self.printTime()

                                        outAvgLASA = modelCreator.doAbstractedTraining(x_train1, x_val, x_test, y_train1, y_val, y_testy, batchSize, self.seed_value, self.num_of_classes, self.dataName, self.fold, self.symbolCount, numEpochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves=skipDebugSaves, rMA = None, useEmbed = useEmbed, earlystop = self.earlystop, useSaves=useSaves, 
                                            abstractionType=abstractionString, takeAvg = takeAvg, heatLayer=heatLayer, rate=dropeOutRate, calcComplexity=calcComplexity, abstraction = 2, reductionInt = -1, thresholdSet=tSet, order=order, step1=step1, step2=step2, step3=step3, doMax=doMax, earlyPredictorZ = earlyPredictor)

                                        self.fillOutDicWithNNOut(abstractionString, configString, fullResults, outAvgLASA, outOri, outSax)

                                        if doFidelity:
                                            abstractionString = "LASA Fidelity Avg Inter "# + combination+ ' ' + str(tSet[0]) + ',' + str(tSet[1])   #+ ' ' + str(self.fold)
                                            print(abstractionString + configString)
                                            self.printTime()
                                        
                                            outAvgLASAFidelity = modelCreator.doAbstractedTraining(x_train1, x_val, x_test, y_train1, y_val, y_testy, batchSize, self.seed_value, self.num_of_classes, self.dataName, self.fold, self.symbolCount, numEpochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves=skipDebugSaves, rMA = None, useEmbed = useEmbed, earlystop = self.earlystop, useSaves=useSaves, 
                                                abstractionType=abstractionString, takeAvg = takeAvg, heatLayer=heatLayer, rate=dropeOutRate, calcComplexity=calcComplexity, abstraction = 2, reductionInt = -1, thresholdSet=tSet, order=order, step1=step1, step2=step2, step3=step3, doMax=doMax, earlyPredictorZ = earlyPredictor, doFidelity=True)

                                            self.fillOutDicWithNNOut(abstractionString, configString, fullResults, outAvgLASAFidelity, outOri, outSax)

                                        #abstractionString = "" + combination+ ' ' + str(tSet[0]) + ',' + str(tSet[1])  #+ ' ' + str(self.fold)
                                        if doShapeletsLASA:
                                            for sLen in shapeletLenghts:
                                                
                                                if(sLen <= 1):
                                                    sLen = int(len(x_train1[0]) * sLen)
                                                abstractionString = 'LASA Avg Inter Shapelets ' # + str(sLen)  + ' ' + str(self.fold)
                                                configString = configString + '; sLen: ' + str(sLen) 
                                                print(abstractionString + configString)
                                                self.printTime()      
                                                try:                                                      
                                                    avgLASAShapelets, baseline = LASA.trainShapelets(outAvgLASA[6], y_train1, time_contract_in_mins =  time_contract_in_mins, initial_num_shapelets_per_case = initial_num_shapelets_per_case, verbose = 2, min_shapelet_length = sLen)
                                                    lasaSAvgOut = LASA.evaluateShapelets(avgLASAShapelets, x_test, y_testy, baseline, outOri, outSax) 
                                                    if abstractionString not in fullResults.keys():
                                                        fullResults[abstractionString] = dict()
                                                    if configString not in fullResults[abstractionString].keys():
                                                        fullResults[abstractionString][configString] = []
                                                    fullResults[abstractionString][configString].append(lasaSAvgOut)
                                                except Exception:
                                                    if abstractionString not in fullResults.keys():
                                                        fullResults[abstractionString] = dict()
                                                    if configString not in fullResults[abstractionString].keys():
                                                        fullResults[abstractionString][configString] = []
                                                    fullResults[abstractionString][configString].append("Exception!")
                                    for tSet in localMaxThresholds:
                                        configString = 'Combination: ' + combination + "; tSet: "+ str(tSet[0]) + ',' + str(tSet[1])
                                        doMax = True
                                        abstractionString = "LASA Max Inter "# + combination + ' ' + str(tSet[0]) + ',' + str(tSet[1])  #+ ' ' + str(self.fold)
                                        print(abstractionString + configString)
                                        self.printTime()

                                        outMaxLASA = modelCreator.doAbstractedTraining(x_train1, x_val, x_test, y_train1, y_val, y_testy, batchSize, self.seed_value, self.num_of_classes, self.dataName, self.fold, self.symbolCount, numEpochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves=skipDebugSaves, rMA = None, useEmbed = useEmbed, earlystop = self.earlystop, useSaves=useSaves, 
                                            abstractionType=abstractionString, takeAvg = takeAvg, heatLayer=heatLayer, rate=dropeOutRate, calcComplexity=calcComplexity, abstraction = 2, reductionInt = -1, thresholdSet=tSet, order=order, step1=step1, step2=step2, step3=step3, doMax=doMax, earlyPredictorZ = earlyPredictor)

                                        self.fillOutDicWithNNOut(abstractionString, configString, fullResults, outMaxLASA, outOri, outSax)

                                        if doFidelity:
                                            abstractionString = "LASA Fidelity Max Inter "# + combination + ' ' + str(tSet[0]) + ',' + str(tSet[1])#  + ' ' + str(self.fold)
                                            print(abstractionString + configString)
                                            self.printTime()
                                            outMaxLASAFidelity = modelCreator.doAbstractedTraining(x_train1, x_val, x_test, y_train1, y_val, y_testy, batchSize, self.seed_value, self.num_of_classes, self.dataName, self.fold, self.symbolCount, numEpochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves=skipDebugSaves, rMA = None, useEmbed = useEmbed, earlystop = self.earlystop, useSaves=useSaves, 
                                                abstractionType=abstractionString, takeAvg = takeAvg, heatLayer=heatLayer, rate=dropeOutRate, calcComplexity=calcComplexity, abstraction = 2, reductionInt = -1, thresholdSet=tSet, order=order, step1=step1, step2=step2, step3=step3, doMax=doMax, earlyPredictorZ = earlyPredictor, doFidelity=True)

                                            self.fillOutDicWithNNOut(abstractionString, configString, fullResults, outMaxLASAFidelity, outOri, outSax)
                                        

                                        #abstractionString = "LASA Max Inter " + combination + ' ' + str(tSet[0]) + ',' + str(tSet[1])  #+ ' ' + str(self.fold)
                                        if doShapeletsLASA:
                                            for sLen in shapeletLenghts:
                                                configString = configString + '; sLen: ' + str(sLen) 
                                                if(sLen <= 1):
                                                    sLen = int(len(x_train1[0]) * sLen)
                                                try:
                                                    maxLASAShapelets, baseline = LASA.trainShapelets(outMaxLASA[6], y_train1,time_contract_in_mins =  time_contract_in_mins, initial_num_shapelets_per_case = initial_num_shapelets_per_case, verbose = 2, min_shapelet_length = sLen)
                                                    abstractionString = 'LASA Max Inter Shapelets '# + str(sLen)
                                                    
                                                    print(abstractionString + configString)
                                                    self.printTime()
                                                    lasaSMaxOut = LASA.evaluateShapelets(maxLASAShapelets, x_test, y_testy, baseline, outOri, outSax)  
                                                    if abstractionString not in fullResults.keys():
                                                        fullResults[abstractionString] = dict()
                                                    if configString not in fullResults[abstractionString].keys():
                                                        fullResults[abstractionString][configString] = []
                                                    fullResults[abstractionString][configString].append(lasaSMaxOut)
                                                except Exception:
                                                    if abstractionString not in fullResults.keys():
                                                        fullResults[abstractionString] = dict()
                                                    if configString not in fullResults[abstractionString].keys():
                                                        fullResults[abstractionString][configString] = []
                                                    fullResults[abstractionString][configString].append("Exception!")


            print("total lasa time")
            self.printTime()

            if doGCR:
                for order in layerCombis:
                    for step1 in steps:
                        for step2 in steps:
                            print('soon start global attention making')
                            combination = order +'-' +step1 +'-' +step2
                            abstractionString = "Original GCR "# + combination  #+ ' ' + str(self.fold)
                            configString = "Combinations: " + combination
                            print(abstractionString + configString)
                            self.printTime()
                            
                            if doOriginalGCR:
                                #Train model with original target labels
                                rMA, rMS, rM = GCRPlus.makeAttention(np.array(outSax[9]), x_train1, y_train1, order, step1, step2, self.num_of_classes, self.valuesA, doThreshold=False, doMax=False, doPenalty=False)
                                
                                # Train model and reduce labels to 1 prediction rather an array
                                #rMA, rMS, rM = GCRPlus.makeAttention(np.array(outSax[9]), x_train1,  y_train1, order, step1, step2, self.num_of_classes, self.valuesA, doThreshold=False, doMax=False, doPenalty=False, reducePredictions=True)
                                
                                # Train model with SAX model predicted target values
                                #rMA, rMS, rM = GCRPlus.makeAttention(np.array(outSax[9]), x_train1,  np.array(outSax[2][0]), order, step1, step2, self.num_of_classes, self.valuesA, doThreshold=False, doMax=False, doPenalty=False, reducePredictions=False)

                                # classify with original labels
                                bestGTMIndex, resultDict = mainHelper.doGCRClassify(abstractionString, rMA, rMS, rM, np.array(outOri[3]),  np.array(outSax[3]), np.array(outOri[2][0]), np.array(outSax[2][0]), x_train1, predictions, x_test, y_testy, self.gtmAbstractions, self.valuesA)
                                
                                # classify with model prediction labels
                                #bestGTMIndex, resultDict = mainHelper.doGCRClassify(abstractionString, rMA, rMS, rM, np.array(outOri[3]),  np.array(outSax[3]), np.array(outOri[2][0]), np.array(outSax[2][0]), x_train1, np.array(outSax[2][0]), x_test, np.array(outSax[3]), self.gtmAbstractions, self.valuesA)

                                for key in resultDict.keys():
                                    self.fillOutDicWithGCRFull(key, configString, fullResults, resultDict[key])

                            if doGIA:
                                for tSet in giathreshold:
                                    giaOut, fidelityOut, giaAbstractionString, fidelityGiaAbstractionString, configString = mainHelper.doGIAProcess(self.gtmAbstractions[bestGTMIndex], combination, tSet, self.fold, outOri, outSax, x_train1, x_val, x_test, y_train1, y_val, y_testy, batchSize, 
                                        self.seed_value, self.num_of_classes, self.dataName, self.symbolCount, numEpochs, numOfAttentionLayers, dmodel, header, dff, skipDebugSaves, rMA, useEmbed, self.earlystop, useSaves, abstractionString, 
                                        takeAvg, heatLayer, dropeOutRate, calcComplexity, bestGTMIndex, doFidelity)
                                    self.fillOutDicWithNNOutFull(giaAbstractionString, configString, fullResults, giaOut)
                                    if doFidelity:
                                        self.fillOutDicWithNNOutFull(fidelityGiaAbstractionString, configString, fullResults, fidelityOut)

                            if doThresholdGCR:
                                print("Start TGCR")
                                self.printTime()
                                answers = []

                                # Original Labels
                                answers = Parallel(n_jobs=14, prefer="threads")(delayed(mainHelper.doTThresholdGCRFull)(tSet, combination, self.fold, np.array(outSax[9]), x_train1, y_train1, order, step1, step2, self.num_of_classes, self.valuesA, np.array(outOri[3]), np.array(outSax[3]), np.array(outOri[2][0]), np.array(outSax[2][0]),predictions, x_test,y_testy, self.gtmAbstractions) for tSet in globalThresholds)
                                
                                # Model predicted Labels
                                answers = Parallel(n_jobs=14, prefer="threads")(delayed(mainHelper.doTThresholdGCRFull)(tSet, combination, self.fold, np.array(outSax[9]), x_train1, np.array(outSax[2][0]), order, step1, step2, self.num_of_classes, self.valuesA, np.array(outOri[3]), np.array(outSax[3]), np.array(outOri[2][0]), np.array(outSax[2][0]), np.array(outSax[2][0]), x_test, np.array(outSax[3]), self.gtmAbstractions) for tSet in globalThresholds)
                                print("End TGCR")
                                self.printTime()
                                for ans, configString in answers:
                                    for key in ans.keys():
                                        self.fillOutDicWithGCRFull(key, configString, fullResults, ans[key])

                                if doFidelity:
                                    print("Start TGCR fidelity")
                                    self.printTime()
                                    answers = []
                                    answers = Parallel(n_jobs=14, prefer="threads")(delayed(mainHelper.doTThresholdGCRFidelityFull)(tSet, combination, self.fold, np.array(outSax[9]), x_train1, y_train1, order, step1, step2, self.num_of_classes, self.valuesA, np.array(outOri[3]), np.array(outSax[3]), np.array(outOri[2][0]), np.array(outSax[2][0]), np.array(outSax[2][0]), x_test, y_testy, self.gtmAbstractions) for tSet in globalThresholds)
                                    print("End TGCR Fidelity")
                                    self.printTime()
                                    for ans, configString in answers:
                                        for key in ans.keys():
                                            self.fillOutDicWithGCRFull(key, configString, fullResults, ans[key])

                            if doPenaltyGCR:
                                print("Start PGCR")
                                self.printTime()
                                answers = []

                                #Original labels
                                answers = Parallel(n_jobs=14, prefer="threads")(delayed(mainHelper.doPenaltyGCRFull)(pMode, combination, self.fold, outSax[9], x_train1, y_train1, order, step1, step2, self.num_of_classes, self.valuesA, outOri[3], outSax[3], outOri[2][0], outSax[2][0], predictions, x_test, y_testy, self.gtmAbstractions) for pMode in penalityModes)
                                
                                #Model predicted labels
                                #answers = Parallel(n_jobs=14, prefer="threads")(delayed(mainHelper.doPenaltyGCRFull)(pMode, combination, self.fold, np.array(outSax[9]), x_train1, np.array(outSax[2][0]), order, step1, step2, self.num_of_classes, self.valuesA, np.array(outOri[3]), np.array(outSax[3]), np.array(outOri[2][0]), np.array(outSax[2][0]), np.array(outSax[2][0]), x_test, np.array(outSax[3]), self.gtmAbstractions) for pMode in penalityModes)
                                print("End PGCR")
                                self.printTime()
                                for ans, configString in answers:
                                    for key in ans.keys():
                                        self.fillOutDicWithGCRFull(key, configString, fullResults, ans[key])

                            print("finished combi: " + str(combination))
                            self.printTime()   
            print("finished fold: " + str(self.fold))
            self.printTime()         
        forconsistantInfo = helper.laamConsistency(consistancyDict, 0, consistancyLabels)
        fullResults['consistancy dict'] = forconsistantInfo
        fullResults['consistancy indices'] = consistancyIndicies
        print("Done done")
        saveName = modelCreator.getWeightName(self.dataName, self.fold, self.symbolCount, numOfAttentionLayers, "results", header, learning = False, results = True)
        print(saveName)
        helper.save_obj(fullResults, str(saveName))
        

        self.printTime()

        return saveName


  

# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.trainExperiment()