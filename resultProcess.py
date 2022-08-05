import os
import numpy as np

from modules import helper
import scipy.stats as ss
from scipy.stats.stats import pearsonr

from sacred import Experiment
import numpy as np
import seml



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


@ex.automain
def run(max_epochs: int, symbolCount: int, header:int, numOfAttentionLayers:int):

    lasaMaxTresholds = [[2,3],[1.8,-1]]
    lasaAvgTresholds = [[1,1.2],[0.8,1.5]]
    giaThresholds = [[0.9,0.9],[0.8,1.2]]
    combis = ['max', 'sum']
    shapeLens = [2,0.3]
    folds = 5
    goodLimit = 0.75


    def rankData(toRank):
        preRanked = ss.rankdata(toRank, method='max')
        subber = len(preRanked) +1
        return [(a - subber)*-1 for a in preRanked]
        

    # create a lot of data structures methods
    def createShapeletBase(shapeletsDict):
        for s in shapeLens:
            shapeletsDict[s] = dict()
            shapeletsDict[s]['Acc'] = []
            shapeletsDict[s]['Precicion'] = []
            shapeletsDict[s]['Recall'] = []
            shapeletsDict[s]['F1'] = []

            shapeletsDict[s]['Train Model Fidelity (Ori)'] = []
            shapeletsDict[s]['Train Model Fidelity (Sax)'] = []
            shapeletsDict[s]['Test Model Fidelity (Ori)'] = []
            shapeletsDict[s]['Test Model Fidelity (Sax)'] = []
            shapeletsDict[s]['Avg info gain'] = []
            shapeletsDict[s]['Avg info gain top 5'] = []
            shapeletsDict[s]['Avg len'] = []
            shapeletsDict[s]['Avg len top 5'] = []
            shapeletsDict[s]['Avg CE'] = []
            shapeletsDict[s]['Avg CE top 5'] = []

    pKeys = ["values", "ranking"]


    def createGiaStructure(lasaResults):
        lasaResults['all'] = dict()
        lasaResults['good'] = dict()
        lasaResults['bad'] = dict()
        for k in lasaResults.keys():
            lasaResults[k]['best'] = dict()
            lasaResults[k]['best']['performance'] = dict()
            lasaResults[k]['best']['performance']['Acc'] = [] # je datensatz
            lasaResults[k]['best']['performance']['Precicion'] =[]
            lasaResults[k]['best']['performance']['Recall'] = []
            lasaResults[k]['best']['performance']['F1'] = []
            lasaResults[k]['best']['performance']['reduction'] = []
            lasaResults[k]['best']['combination'] = []
            lasaResults[k]['best']['fidelity'] = []
            lasaResults[k]['best']['fidelity combination'] = []


            for p in pKeys:
                lasaResults[k][p] = dict()

                lasaResults[k][p]['avg'] = dict()
                for t in giaThresholds:
                    lasaResults[k][p]['avg'][str(t)] = dict()
                    for c1 in combis:
                        for c2 in combis:
                            for c3 in combis:
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)] = dict()
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['Acc'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['Precicion'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['Recall'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['F1'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['reduction'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['reduction comp'] = []
                                
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['train fidelity (ori)'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['val fidelity (ori)'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['test fidelity (ori)'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['train fidelity (sax)'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['val fidelity (sax)'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['test fidelity (sax)'] = []         
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['test permutationsEntropy'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['test spectralEntropy'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['test svdEntropy'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['test approximateEntropy'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['test sampleEntropy'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['test CE'] = []      
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['test shifts'] = []   
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)]['gtm abstraction'] = []                                   

    def createLasaStructure(lasaResults):
        lasaResults['all'] = dict()
        lasaResults['good'] = dict()
        lasaResults['bad'] = dict()
        for k in lasaResults.keys():
            lasaResults[k]['best'] = dict()
            lasaResults[k]['best']['performance'] = dict()
            lasaResults[k]['best']['performance']['Acc'] = [] # je datensatz
            lasaResults[k]['best']['performance']['Precicion'] =[]
            lasaResults[k]['best']['performance']['Recall'] = []
            lasaResults[k]['best']['performance']['F1'] = []
            lasaResults[k]['best']['performance']['reduction'] = []
            lasaResults[k]['best']['combination'] = []
            lasaResults[k]['best']['fidelity'] = []
            lasaResults[k]['best']['fidelity combination'] = []

            for p in pKeys:
                lasaResults[k][p] = dict()
                lasaResults[k][p]['max'] = dict()
                for t in lasaMaxTresholds:
                    lasaResults[k][p]['max'][str(t)] = dict()
                    for c1 in combis:
                        for c2 in combis:
                            for c3 in combis:
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)] = dict()
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['Acc'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['Precicion'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['Recall'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['F1'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['reduction'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['train fidelity (ori)'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['val fidelity (ori)'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test fidelity (ori)'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['train fidelity (sax)'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['val fidelity (sax)'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test fidelity (sax)'] = []

                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test permutationsEntropy o'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test spectralEntropy o'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test svdEntropy o'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test approximateEntropy o'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test sampleEntropy o'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test CE o'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test shifts o'] = []

                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test permutationsEntropy s'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test spectralEntropy s'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test svdEntropy s'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test approximateEntropy s'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test sampleEntropy s'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test CE s'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test shifts s'] = []
                                
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['reduction comp'] = []
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['reduction'] = []
                                

                lasaResults[k][p]['avg'] = dict()
                for t in lasaAvgTresholds:
                    lasaResults[k][p]['avg'][str(t)] = dict()
                    for c1 in combis:
                        for c2 in combis:
                            for c3 in combis:
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)] = dict()
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['Acc'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['Precicion'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['Recall'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['F1'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['reduction'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['reduction comp'] = []
                                
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['train fidelity (ori)'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['val fidelity (ori)'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test fidelity (ori)'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['train fidelity (sax)'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['val fidelity (sax)'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test fidelity (sax)'] = []         

                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test permutationsEntropy o'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test spectralEntropy o'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test svdEntropy o'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test approximateEntropy o'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test sampleEntropy o'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test CE o'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test shifts o'] = []

                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test permutationsEntropy s'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test spectralEntropy s'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test svdEntropy s'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test approximateEntropy s'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test sampleEntropy s'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test CE s'] = []
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)]['test shifts s'] = []

    def createShapeletLasaStructure(lasaResults):
        lasaResults['all'] = dict()
        lasaResults['good'] = dict()
        lasaResults['bad'] = dict()
        for k in lasaResults.keys():
            lasaResults[k]['best'] = dict()
            lasaResults[k]['best']['performance'] = dict()
            lasaResults[k]['best']['performance']['Acc'] = [] # je datensatz
            lasaResults[k]['best']['performance']['Precicion'] =[]
            lasaResults[k]['best']['performance']['Recall'] = []
            lasaResults[k]['best']['performance']['F1'] = []
            lasaResults[k]['best']['performance']['Avg info gain top 5'] = []
            lasaResults[k]['best']['performance']['Avg len top 5'] = []
            lasaResults[k]['best']['combination'] = []
            lasaResults[k]['best']['fidelity'] = []
            lasaResults[k]['best']['fidelity combination'] = []


            for p in pKeys:
                lasaResults[k][p] = dict()
                lasaResults[k][p]['max'] = dict()
                for t in lasaMaxTresholds:
                    lasaResults[k][p]['max'][str(t)] = dict()
                    for c1 in combis:
                        for c2 in combis:
                            for c3 in combis:
                                
                                lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)] = dict()
                                createShapeletBase(lasaResults[k][p]['max'][str(t)]['hl'+str(c1)+str(c2)+str(c3)])
                                

                lasaResults[k][p]['avg'] = dict()
                for t in lasaAvgTresholds:
                    lasaResults[k][p]['avg'][str(t)] = dict()
                    for c1 in combis:
                        for c2 in combis:
                            for c3 in combis:
                                lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)] = dict()
                                createShapeletBase(lasaResults[k][p]['avg'][str(t)]['hl'+str(c1)+str(c2)+str(c3)])


    gcrArten = ['Sum FCAM', 'r.Avg FCAM', 'Sum CRCAM', 'r.Avg CRCAM', 'max', 'max+', 'average', 'average+', 'median', 'median+']

    def createGCRStrcture(lasaResults):
        lasaResults['all'] = dict()
        lasaResults['good'] = dict()
        lasaResults['bad'] = dict()
        for k in lasaResults.keys():
            lasaResults[k]['best'] = dict()
            lasaResults[k]['best']['performance'] = dict()
            lasaResults[k]['best']['performance']['Acc'] = [] # per datensatz
            lasaResults[k]['best']['performance']['Precicion'] =[]
            lasaResults[k]['best']['performance']['Recall'] = []
            lasaResults[k]['best']['performance']['F1'] = []
            lasaResults[k]['best']['performance']['reduction'] = []
            lasaResults[k]['best']['combination'] = []
            lasaResults[k]['best']['fidelity'] = []
            lasaResults[k]['best']['fidelity combination'] = []

            for p in pKeys:
                lasaResults[k][p] = dict()
                for g in gcrArten:
                    lasaResults[k][p][g] = dict()
                    for c1 in combis:
                            for c2 in combis:
                                combiString = 'hl'+str(c1)+str(c2)
                                lasaResults[k][p][g][combiString] = dict()

                                lasaResults[k][p][g][combiString]['Acc'] = []
                                lasaResults[k][p][g][combiString]['Precicion'] = []
                                lasaResults[k][p][g][combiString]['Recall'] = []
                                lasaResults[k][p][g][combiString]['F1'] = []

                                lasaResults[k][p][g][combiString]['train fidelity (ori)'] = []
                                lasaResults[k][p][g][combiString]['test fidelity (ori)'] = []
                                lasaResults[k][p][g][combiString]['train fidelity (sax)'] = []
                                lasaResults[k][p][g][combiString]['test fidelity (sax)'] = []

                                # #confidence ranks top80Acc, top50Acc, top20Acc, top10Acc
                                lasaResults[k][p][g][combiString]['train confidence 80'] = []
                                lasaResults[k][p][g][combiString]['train confidence 50'] = []
                                lasaResults[k][p][g][combiString]['train confidence 20'] = []
                                lasaResults[k][p][g][combiString]['train confidence 10'] = []
                                lasaResults[k][p][g][combiString]['test confidence 80'] = []    
                                lasaResults[k][p][g][combiString]['test confidence 50'] = []    
                                lasaResults[k][p][g][combiString]['test confidence 20'] = []    
                                lasaResults[k][p][g][combiString]['test confidence 10'] = []

                                lasaResults[k][p][g][combiString]['train confidence 80F'] = []
                                lasaResults[k][p][g][combiString]['train confidence 50F'] = []
                                lasaResults[k][p][g][combiString]['train confidence 20F'] = []
                                lasaResults[k][p][g][combiString]['train confidence 10F'] = []
                                lasaResults[k][p][g][combiString]['test confidence 80F'] = []    
                                lasaResults[k][p][g][combiString]['test confidence 50F'] = []    
                                lasaResults[k][p][g][combiString]['test confidence 20F'] = []    
                                lasaResults[k][p][g][combiString]['test confidence 10F'] = []                          




    def fillBaseStates(baseStats, dataString, res):
        oriTestAcc = 0
        oriTestPrec = 0
        oriTestRec = 0
        oriTestF1 = 0
        oriTrainFidelityOri = 0
        oriValFidelityOri = 0
        oriTestFidelityOri = 0



        for a in res[dataString]['']:
            oriTestAcc += a['Test Accuracy']
            oriTestPrec += a['Test Precision']
            oriTestRec += a['Test Recall']
            oriTestF1 += a['Test F1']
            oriTrainFidelityOri += a['Train Model Fidelity (Ori)']
            oriValFidelityOri += a['Val Model Fidelity (Ori)']
            oriTestFidelityOri += a['Test Model Fidelity (Ori)']

            baseStats['test permutationsEntropy'].append(a['Test Complexity']['permutationsEntropy'])
            baseStats['test spectralEntropy'].append(a['Test Complexity']['spectralEntropy'])
            baseStats['test svdEntropy'].append(a['Test Complexity']['svdEntropy'])
            baseStats['test approximateEntropy'].append(a['Test Complexity']['approximateEntropy'])
            baseStats['test sampleEntropy'].append(a['Test Complexity']['sample entropy'])
            baseStats['test CE'].append(a['Test Complexity']['CE'])
            baseStats['test shifts'].append(a['Test Shifts'])

        # ori info
        baseStats['Acc'].append(oriTestAcc/folds)
        baseStats['Precicion'].append(oriTestPrec/folds)
        baseStats['Recall'].append(oriTestRec/folds)
        baseStats['F1'].append(oriTestF1/folds)
        baseStats['train fidelity (ori)'].append(oriTrainFidelityOri/folds)
        baseStats['val fidelity (ori)'].append(oriValFidelityOri/folds)
        baseStats['test fidelity (ori)'].append(oriTestFidelityOri/folds)

    def fillShapeletBase(shapeletsDict, dataString, subString, res, shapLen, oriPredictions, saxPredictions, saxAcc = 0, relativValues = True): 
        #for sOri in res[dataString]:
            #if sOri.endswith("2"):
            #    s = 2
            #else:
            #    s = 0.3
            baseSAcc = 0
            baseSprecision = 0
            baseSrecall = 0
            baseSf1 = 0
            baseSinfoGain = 0
            baseSinfoGainTop5 = 0
            baseSavgLenShapelets = 0
            baseSavgLenShapeletsTop5 = 0
            baseSavgCE = 0
            baseSavgCETop5 = 0
            trainFiOri = 0
            testFiOri = 0
            trainFiSax = 0
            testFiSax = 0

            
            folds = 5
            
            if subString in res[dataString]: 
                folds =len(res[dataString][subString])
                fold = 0
                for a in res[dataString][subString]:
                    if a == "Exception!":
                        folds -= 1
                        if folds == 0:
                            folds = 1
                        fold += 1
                        continue
                        
                    baseSAcc += a['accuracy']
                    baseSprecision += a['precision']
                    baseSrecall += a['recall']
                    baseSf1 += a['f1 score']
                    baseSinfoGain += a['Avg info gain']
                    baseSinfoGainTop5 += a['Avg info gain top 5']
                    baseSavgLenShapelets += a['Avg len']
                    baseSavgLenShapeletsTop5 += a['Avg len top 5']
                    baseSavgCE += a['Avg CE']
                    baseSavgCETop5 += a['Avg CE top 5']
                    
                    trainFiOri += a['Train Model Fidelity (Ori)']
                    trainFiSax += a['Train Model Fidelity (Sax)']
                    testFiOri += a['Test Model Fidelity (Ori)']
                    testFiSax += a['Test Model Fidelity (Sax)']
                    #trainFiOri += helper.modelFidelity(a['train predictions'], (oriPredictions[fold]['Train Predictions']+1))
                    #trainFiSax += helper.modelFidelity(a['train predictions'], (saxPredictions[fold]['Train Predictions'] +1))
                    #testFiOri += helper.modelFidelity(a['test predictions'], (oriPredictions[fold]['Test Predictions']+1))
                    #testFiSax +=helper.modelFidelity(a['test predictions'], (saxPredictions[fold]['Test Predictions']+1))
                    fold += 1

            if not relativValues:
                shapeletsDict[shapLen]['Acc'].append((baseSAcc/folds))
                shapeletsDict[shapLen]['Precicion'].append((baseSprecision/folds))
                shapeletsDict[shapLen]['Recall'].append((baseSrecall/folds))
                shapeletsDict[shapLen]['F1'].append((baseSf1/folds))
            else:
                shapeletsDict[shapLen]['Acc'].append((baseSAcc/folds)- saxAcc)
                shapeletsDict[shapLen]['Precicion'].append((baseSprecision/folds)- saxAcc)
                shapeletsDict[shapLen]['Recall'].append((baseSrecall/folds)- saxAcc)
                shapeletsDict[shapLen]['F1'].append((baseSf1/folds)- saxAcc)

            shapeletsDict[shapLen]['Train Model Fidelity (Ori)'].append(trainFiOri/folds)
            shapeletsDict[shapLen]['Train Model Fidelity (Sax)'].append(trainFiSax/folds)
            shapeletsDict[shapLen]['Test Model Fidelity (Ori)'].append(testFiOri/folds)
            shapeletsDict[shapLen]['Test Model Fidelity (Sax)'].append(testFiSax/folds)
            shapeletsDict[shapLen]['Avg info gain'].append(baseSinfoGain/folds)
            shapeletsDict[shapLen]['Avg info gain top 5'].append(baseSinfoGainTop5/folds)
            shapeletsDict[shapLen]['Avg len'].append(baseSavgLenShapelets/folds)
            shapeletsDict[shapLen]['Avg len top 5'].append(baseSavgLenShapeletsTop5/folds)
            shapeletsDict[shapLen]['Avg CE'].append(baseSavgCE/folds)
            shapeletsDict[shapLen]['Avg CE top 5'].append(baseSavgCETop5/folds)
            folds = 5


    def fillLASACombisValue(lasaResults, dataString, tKindKey, tresholdSet, res, saxAcc, relativValues = True, gia=False):
        #k = all good bad
        #p = values ranking

        for t in tresholdSet:
            combisAcc = []
            combisPrec = []
            combisRec = []
            combisF1 = []
            combisReduction = []
            combisTrainFidelityOri = []
            combisValFidelityOri = []
            combisTestFidelityOri = []
            combisTrainFidelitySax = []
            combisValFidelitySax = []
            combisTestFidelitySax = []

            for c1 in combis:
                for c2 in combis:
                    adone = False
                    for c3 in combis:
                        if gia and adone:
                            continue
                        else:
                            adone = True
                        #res[dataString]['Combination: hl-max-sum-max; tSet: 1,1.2']:
                        testAcc = 0
                        testPrec = 0
                        testRec = 0
                        testF1 = 0
                        reduction = 0
                        trainFidelityOri = 0
                        valFidelityOri = 0
                        testFidelityOri = 0
                        trainFidelitySax = 0
                        valFidelitySax = 0
                        testFidelitySax = 0
                        
                        combiString = ''
                        usedAbstraction = ''
                        if gia:
                            reduceStrings = ['max','max+','average','average+','median','median+']
                            for abstraction in reduceStrings:
                                combiString = 'Combinations: hl-'+c1+'-'+c2+'; Abstraction: ' + str(abstraction) + '; threshold: ' + str(t[0]) + ',' + str(t[1])
                                if combiString in res[dataString].keys():
                                    usedAbstraction = abstraction
                                    break
                        else:
                            combiString = 'Combination: hl-'+c1+'-'+c2+'-'+c3+'; tSet: ' + str(t[0])+','+str(t[1])

                        p = 'values'
                        k = 'all'
                        if gia:
                            oCombiString = 'hl'+str(c1)+str(c2)
                        else:
                            oCombiString = 'hl'+str(c1)+str(c2)+str(c3)
                        base = lasaResults[k][p][tKindKey][str(t)][oCombiString]

                        folds = 5
                        i = 0 
                        for a in res[dataString][combiString]:
                            if a == "Exception!":
                                folds -= 1
                                if folds == 0:
                                    folds = 1
                                continue
                            testAcc += a['Test Accuracy']
                            testPrec += a['Test Precision']
                            testRec += a['Test Recall']
                            testF1 += a['Test F1']
                            reduction += np.mean(a['Test Reduction'])
                            trainFidelityOri += a['Train Model Fidelity (Ori)']
                            valFidelityOri += a['Val Model Fidelity (Ori)']
                            testFidelityOri += a['Test Model Fidelity (Ori)']
                            trainFidelitySax += a['Train Model Fidelity (Sax)']
                            valFidelitySax += a['Val Model Fidelity (Sax)']
                            testFidelitySax += a['Test Model Fidelity (Sax)']
                            

                            coc = res['Original'][''][i]
                            c = 1- np.array(a['Test Complexity']['permutationsEntropy'])/np.array(coc['Test Complexity']['permutationsEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            print(a['Test Complexity']['permutationsEntropy'])
                            print('#####################################')
                            print(np.array(coc['Test Complexity']['permutationsEntropy']))
                            print('#####################################')
                            print(c[z])
                            print('++++++++++++++++++++++++++++++++++++++')
                            base['test permutationsEntropy o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test permutationsEntropy o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['spectralEntropy'])/np.array(coc['Test Complexity']['spectralEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test spectralEntropy o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test spectralEntropy o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['svdEntropy'])/np.array(coc['Test Complexity']['svdEntropy'])
                            
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test svdEntropy o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test svdEntropy o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['approximateEntropy'])/np.array(coc['Test Complexity']['approximateEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test approximateEntropy o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test approximateEntropy o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['sample entropy'])/np.array(coc['Test Complexity']['sample entropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test sampleEntropy o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test sampleEntropy o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['CE'])/np.array(coc['Test Complexity']['CE'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test CE o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test CE o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Shifts'])/np.array(coc['Test Shifts'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test shifts o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test shifts o'].append(np.mean(c[z]))

                            coc = res['SAX'][''][i]
                            c = 1- np.array(a['Test Complexity']['permutationsEntropy'])/np.array(coc['Test Complexity']['permutationsEntropy'])
                              #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test permutationsEntropy s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test permutationsEntropy s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['spectralEntropy'])/np.array(coc['Test Complexity']['spectralEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test spectralEntropy s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test spectralEntropy s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['svdEntropy'])/np.array(coc['Test Complexity']['svdEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test svdEntropy s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test svdEntropy s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['approximateEntropy'])/np.array(coc['Test Complexity']['approximateEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test approximateEntropy s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test approximateEntropy s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['sample entropy'])/np.array(coc['Test Complexity']['sample entropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test sampleEntropy s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test sampleEntropy s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['CE'])/np.array(coc['Test Complexity']['CE'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test CE s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test CE s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Shifts'])/np.array(coc['Test Shifts'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test shifts s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test shifts s'].append(np.mean(c[z]))

                            base['reduction comp'].append(np.mean(a['Test Reduction']))

                            i = i+1
                        if gia:
                            base['gtm abstraction'].append(usedAbstraction)

                        testAcc = testAcc/folds
                        testPrec = testPrec/folds
                        testRec = testRec/folds
                        testF1 = testF1/folds
                        reduction = reduction/folds
                        trainFidelityOri = trainFidelityOri/folds
                        valFidelityOri = valFidelityOri/folds
                        testFidelityOri = testFidelityOri/folds
                        trainFidelitySax = trainFidelitySax/folds
                        valFidelitySax = valFidelitySax/folds
                        testFidelitySax =testFidelitySax/folds
                        folds = 5
                        
                        
                        
                        if not relativValues:
                            base['Acc'].append(testAcc)
                            base['Precicion'].append(testPrec)
                            base['Recall'].append(testRec)
                            base['F1'].append(testF1)
                        else:
                            base['Acc'].append(testAcc- saxAcc)
                            base['Precicion'].append(testPrec- saxAcc)
                            base['Recall'].append(testRec- saxAcc)
                            base['F1'].append(testF1- saxAcc)

                        base['reduction'].append(reduction)
                        base['train fidelity (ori)'].append(trainFidelityOri)
                        base['val fidelity (ori)'].append(valFidelityOri)
                        base['test fidelity (ori)'].append(testFidelityOri)
                        base['train fidelity (sax)'].append(trainFidelitySax)
                        base['val fidelity (sax)'].append(valFidelitySax)
                        base['test fidelity (sax)'].append(testFidelitySax)

                        combisAcc.append(testAcc)
                        combisPrec.append(testPrec)
                        combisRec.append(testRec)
                        combisF1.append(testF1)
                        combisReduction.append(reduction)
                        combisTrainFidelityOri.append(trainFidelityOri)
                        combisValFidelityOri.append(valFidelityOri)
                        combisTestFidelityOri.append(testFidelityOri)
                        combisTrainFidelitySax.append(trainFidelitySax)
                        combisValFidelitySax.append(valFidelitySax)
                        combisTestFidelitySax.append(testFidelitySax)

                        if saxAcc >= goodLimit:
                            k = 'good'
                        else:
                            k = 'bad'

                        base = lasaResults[k][p][tKindKey][str(t)][oCombiString]

                        i = 0
                        for a in res[dataString][combiString]:
                            if a == "Exception!":
                                folds -= 1
                                if folds == 0:
                                    folds = 1
                                continue
                            
                            coc = res['Original'][''][i]
                            c = 1- np.array(a['Test Complexity']['permutationsEntropy'])/np.array(coc['Test Complexity']['permutationsEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            print(a['Test Complexity']['permutationsEntropy'])
                            print('#####################################')
                            print(np.array(coc['Test Complexity']['permutationsEntropy']))
                            print('#####################################')
                            print(c[z])
                            print('++++++++++++++++++++++++++++++++++++++')
                            base['test permutationsEntropy o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test permutationsEntropy o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['spectralEntropy'])/np.array(coc['Test Complexity']['spectralEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test spectralEntropy o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test spectralEntropy o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['svdEntropy'])/np.array(coc['Test Complexity']['svdEntropy'])
                            
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test svdEntropy o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test svdEntropy o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['approximateEntropy'])/np.array(coc['Test Complexity']['approximateEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test approximateEntropy o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test approximateEntropy o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['sample entropy'])/np.array(coc['Test Complexity']['sample entropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test sampleEntropy o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test sampleEntropy o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['CE'])/np.array(coc['Test Complexity']['CE'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test CE o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test CE o'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Shifts'])/np.array(coc['Test Shifts'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test shifts o'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test shifts o'].append(np.mean(c[z]))

                            coc = res['SAX'][''][i]
                            c = 1- np.array(a['Test Complexity']['permutationsEntropy'])/np.array(coc['Test Complexity']['permutationsEntropy'])
                              #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test permutationsEntropy s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test permutationsEntropy s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['spectralEntropy'])/np.array(coc['Test Complexity']['spectralEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test spectralEntropy s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test spectralEntropy s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['svdEntropy'])/np.array(coc['Test Complexity']['svdEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test svdEntropy s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test svdEntropy s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['approximateEntropy'])/np.array(coc['Test Complexity']['approximateEntropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test approximateEntropy s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test approximateEntropy s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['sample entropy'])/np.array(coc['Test Complexity']['sample entropy'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test sampleEntropy s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test sampleEntropy s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Complexity']['CE'])/np.array(coc['Test Complexity']['CE'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test CE s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test CE s'].append(np.mean(c[z]))
                            c = 1- np.array(a['Test Shifts'])/np.array(coc['Test Shifts'])
                            #c = np.where(np.isnan(c), 0, c)
                            z = np.where((c != np.inf) & (c != -np.inf) & (c == c))
                            base['test shifts s'].append(pearsonr(np.array(a['Test Reduction'])[z], c[z]))
                            #base['test shifts s'].append(np.mean(c[z]))

                            base['reduction comp'].append(np.mean(a['Test Reduction']))

                            i = i+1
                        if gia:
                            base['gtm abstraction'].append(usedAbstraction)
                        
                        if not relativValues:
                            base['Acc'].append(testAcc)
                            base['Precicion'].append(testPrec)
                            base['Recall'].append(testRec)
                            base['F1'].append(testF1)
                        else:
                            base['Acc'].append(testAcc- saxAcc)
                            base['Precicion'].append(testPrec- saxAcc)
                            base['Recall'].append(testRec- saxAcc)
                            base['F1'].append(testF1- saxAcc)
                        base['reduction'].append(reduction)
                        base['train fidelity (ori)'].append(trainFidelityOri)
                        base['val fidelity (ori)'].append(valFidelityOri)
                        base['test fidelity (ori)'].append(testFidelityOri)
                        base['train fidelity (sax)'].append(trainFidelitySax)
                        base['val fidelity (sax)'].append(valFidelitySax)
                        base['test fidelity (sax)'].append(testFidelitySax)
            
            ##ranking
            combisAcc = rankData(combisAcc)
            combisPrec = rankData(combisPrec)
            combisRec = rankData(combisRec)
            combisF1 = rankData(combisF1)
            combisReduction = rankData(combisReduction)
            combisTrainFidelityOri = rankData(combisTrainFidelityOri)
            combisValFidelityOri = rankData(combisValFidelityOri)
            combisTestFidelityOri = rankData(combisTestFidelityOri)
            combisTrainFidelitySax = rankData(combisTrainFidelitySax)
            combisValFidelitySax = rankData(combisValFidelitySax)
            combisTestFidelitySax = rankData(combisTestFidelitySax)

            index = 0
            for c1 in combis:
                for c2 in combis:
                    adone = False
                    for c3 in combis:
                        if gia and adone:
                            continue
                        else:
                            adone = True

                        if gia:
                            reduceStrings = ['max','max+','average','average+','median','median+']
                            for abstraction in reduceStrings:
                                combiString = 'Combinations: hl-'+c1+'-'+c2+'; Abstraction: ' + str(abstraction) + '; threshold: ' + str(t[0]) + ',' + str(t[1])
                                if combiString in res[dataString].keys():
                                    usedAbstraction = abstraction
                                    break
                        else:
                            combiString = 'Combination: hl-'+c1+'-'+c2+'-'+c3+'; tSet: ' + str(t[0])+','+str(t[1])

                        p = 'ranking'
                        k = 'all'

                        if gia:
                            oCombiString = 'hl'+str(c1)+str(c2)
                        else:
                            oCombiString = 'hl'+str(c1)+str(c2)+str(c3)

                        base = lasaResults[k][p][tKindKey][str(t)][oCombiString]
                        base['Acc'].append(combisAcc[index])
                        base['Precicion'].append(combisPrec[index])
                        base['Recall'].append(combisRec[index])
                        base['F1'].append(combisF1[index])
                        base['reduction'].append(combisReduction[index])
                        base['train fidelity (ori)'].append(combisTrainFidelityOri[index])
                        base['val fidelity (ori)'].append(combisValFidelityOri[index])
                        base['test fidelity (ori)'].append(combisTestFidelityOri[index])
                        base['train fidelity (sax)'].append(combisTrainFidelitySax[index])
                        base['val fidelity (sax)'].append(combisValFidelitySax[index])
                        base['test fidelity (sax)'].append(combisTestFidelitySax[index])

                        if saxAcc >= goodLimit:
                            k = 'good'
                        else:
                            k = 'bad'

                        base = lasaResults[k][p][tKindKey][str(t)][oCombiString]
                        
                        base['Acc'].append(combisAcc[index])
                        base['Precicion'].append(combisPrec[index])
                        base['Recall'].append(combisRec[index])
                        base['F1'].append(combisF1[index])
                        base['reduction'].append(combisReduction[index])
                        base['train fidelity (ori)'].append(combisTrainFidelityOri[index])
                        base['val fidelity (ori)'].append(combisValFidelityOri[index])
                        base['test fidelity (ori)'].append(combisTestFidelityOri[index])
                        base['train fidelity (sax)'].append(combisTrainFidelitySax[index])
                        base['val fidelity (sax)'].append(combisValFidelitySax[index])
                        base['test fidelity (sax)'].append(combisTestFidelitySax[index])

                        index += 1


    def fillLASAShapeletCombis(lasaResults, dataString, tKindKey, tresholdSet, res, saxAcc, sEnd, relativValues = True):
            
        for t in tresholdSet:
            for s in shapeLens:
                combiSAcc = []
                combiSPrec = []
                combiSRec = []
                combiSF1 = []
                combiSinfoGain = []
                combiSinfoGainTop5 = []
                combiSavgLenShapelets = []
                combiSavgLenShapeletsTop5 = []
                combiSavgCE = []
                combiSavgCETop5 = []
                combiSTrainFidelityOri = []
                combiSTestFidelityOri = []
                combiSTrainFidelitySax = []
                combiSTestFidelitySax = []

                for c1 in combis:
                    for c2 in combis:
                        for c3 in combis:
                            p = 'values'
                            k = 'all'
                            if s == 2:
                                combiString = 'Combination: hl-'+c1+'-'+c2+'-'+c3+'; tSet: ' + str(t[0])+','+str(t[1]) + '; sLen: ' + str(s) 
                            else:
                                combiString = 'Combination: hl-'+c1+'-'+c2+'-'+c3+'; tSet: ' + str(t[0])+','+str(t[1]) + '; sLen: ' + str(2)  + '; sLen: ' + sEnd
                                
                            results = lasaResults[k][p][tKindKey][str(t)]['hl'+str(c1)+str(c2)+str(c3)]
                            fillShapeletBase(results, dataString, combiString, res, s, res['Original'][''], res['SAX'][''], saxAcc = saxAcc,relativValues=relativValues)
                            results = results[s]
                            combiSAcc.append(results['Acc'])
                            combiSPrec.append(results['Precicion'])
                            combiSRec.append(results['Recall'])
                            combiSF1.append(results['F1'])
                            combiSinfoGain.append(results['Avg info gain'])
                            combiSinfoGainTop5.append(results['Avg info gain top 5'])
                            combiSavgLenShapelets.append(results['Avg len'])
                            combiSavgLenShapeletsTop5.append(results['Avg len top 5'])
                            combiSavgCE.append(results['Avg CE'])
                            combiSavgCETop5.append(results['Avg CE top 5'])
                            combiSTrainFidelityOri.append(results['Train Model Fidelity (Ori)'])
                            combiSTrainFidelitySax.append(results['Train Model Fidelity (Sax)'])
                            combiSTestFidelityOri.append(results['Test Model Fidelity (Ori)'])
                            combiSTestFidelitySax.append(results['Test Model Fidelity (Sax)'])
                            

                            if saxAcc >= goodLimit:
                                k = 'good'
                            else:
                                k = 'bad'

                            results = lasaResults[k][p][tKindKey][str(t)]['hl'+str(c1)+str(c2)+str(c3)]
                            fillShapeletBase(results, dataString, combiString, res, s, res['Original'][''], res['SAX'][''], saxAcc = saxAcc)
                
                combiSAcc = rankData(combiSAcc)
                combiSPrec = rankData(combiSPrec)
                combiSRec = rankData(combiSRec)
                combiSF1 = rankData(combiSF1)
                combiSinfoGain = rankData(combiSinfoGain)
                combiSinfoGainTop5 = rankData(combiSinfoGainTop5)
                combiSavgLenShapelets = rankData(combiSavgLenShapelets)
                combiSavgLenShapeletsTop5 = rankData(combiSavgLenShapeletsTop5)
                combiSavgCE = rankData(combiSavgCE)
                combiSavgCETop5 =rankData(combiSavgCETop5)
                combiSTrainFidelityOri = rankData(combiSTrainFidelityOri)
                combiSTestFidelityOri = rankData(combiSTestFidelityOri)
                combiSTrainFidelitySax = rankData(combiSTrainFidelitySax)
                combiSTestFidelitySax = rankData(combiSTestFidelitySax)

                index = 0
                for c1 in combis:
                    for c2 in combis:
                        for c3 in combis:
                            p = 'ranking'
                            k = 'all'

                            base = lasaResults[k][p][tKindKey][str(t)]['hl'+str(c1)+str(c2)+str(c3)]
                            base[s]['Acc'].append(combiSAcc[index])
                            base[s]['Precicion'].append(combiSPrec[index])
                            base[s]['Recall'].append(combiSRec[index])
                            base[s]['F1'].append(combiSF1[index])

                            base[s]['Train Model Fidelity (Ori)'].append(combiSTrainFidelityOri[index])
                            base[s]['Train Model Fidelity (Sax)'].append(combiSTrainFidelitySax[index])
                            base[s]['Test Model Fidelity (Ori)'].append(combiSTestFidelityOri[index])
                            base[s]['Test Model Fidelity (Sax)'].append(combiSTestFidelitySax[index])
                            base[s]['Avg info gain'].append(combiSinfoGain[index])
                            base[s]['Avg info gain top 5'].append(combiSinfoGainTop5[index])
                            base[s]['Avg len'].append(combiSavgLenShapelets[index])
                            base[s]['Avg len top 5'].append(combiSavgLenShapeletsTop5[index])
                            base[s]['Avg CE'].append(combiSavgCE[index])
                            base[s]['Avg CE top 5'].append(combiSavgCETop5[index])

                            if saxAcc >= goodLimit:
                                k = 'good'
                            else:
                                k = 'bad'

                            base = lasaResults[k][p][tKindKey][str(t)]['hl'+str(c1)+str(c2)+str(c3)]
                            
                            base[s]['Acc'].append(combiSAcc[index])
                            base[s]['Precicion'].append(combiSPrec[index])
                            base[s]['Recall'].append(combiSRec[index])
                            base[s]['F1'].append(combiSF1[index])

                            base[s]['Train Model Fidelity (Ori)'].append(combiSTrainFidelityOri[index])
                            base[s]['Train Model Fidelity (Sax)'].append(combiSTrainFidelitySax[index])
                            base[s]['Test Model Fidelity (Ori)'].append(combiSTestFidelityOri[index])
                            base[s]['Test Model Fidelity (Sax)'].append(combiSTestFidelitySax[index])
                            base[s]['Avg info gain'].append(combiSinfoGain[index])
                            base[s]['Avg info gain top 5'].append(combiSinfoGainTop5[index])
                            base[s]['Avg len'].append(combiSavgLenShapelets[index])
                            base[s]['Avg len top 5'].append(combiSavgLenShapeletsTop5[index])
                            base[s]['Avg CE'].append(combiSavgCE[index])
                            base[s]['Avg CE top 5'].append(combiSavgCETop5[index])

                            index += 1


    def fillGCRCombis(gcrResults, dataString, subModString, res, saxAcc, oriPredictions, saxPredictions, relativValues = True):
        for g in gcrArten:
            combisAcc = []
            combisPrec = []
            combisRec = []
            combisF1 = []
            combisTestConfidence80 = []
            combisTestConfidence50 = []
            combisTestConfidence20 = []
            combisTestConfidence10 = []
            combisTrainConfidence80 = []
            combisTrainConfidence50 = []
            combisTrainConfidence20 = []
            combisTrainConfidence10 = []
            combisTrainFidelityOri = []
            combisTestFidelityOri = []
            combisTrainFidelitySax = []
            combisTestFidelitySax = []
            for c1 in combis:
                for c2 in combis:

                    p = 'values'
                    k = 'all'

                    testAcc = 0
                    testPrec = 0
                    testRec = 0
                    testF1 = 0
                    trainConfidence80 = 0
                    trainConfidence50 = 0
                    trainConfidence20 = 0
                    trainConfidence10 = 0
                    testConfidence80 = 0
                    testConfidence50 = 0
                    testConfidence20 = 0
                    testConfidence10 = 0
                    trainConfidence80F = 0
                    trainConfidence50F = 0
                    trainConfidence20F = 0
                    trainConfidence10F = 0
                    testConfidence80F = 0
                    testConfidence50F = 0
                    testConfidence20F = 0
                    testConfidence10F = 0
                    trainFidelityOri = 0
                    testFidelityOri = 0
                    trainFidelitySax = 0
                    testFidelitySax = 0

                    mainString = dataString + g #dataString for example "Original GCR  "
                    subString = subModString + 'hl-'+c1+'-'+c2

                    fold = 0
                    for a in res[mainString][subString]:
                        testAcc += a['test Accuracy']
                        testPrec += a['test Precision']
                        testRec += a['test Recall']
                        testF1 += a['test F1']
                        trainConfidence80 += a['train confidence'][0]
                        trainConfidence50 += a['train confidence'][1]
                        trainConfidence20 += a['train confidence'][2]
                        trainConfidence10 += a['train confidence'][3]
                        testConfidence80 += a['test confidence'][0]
                        testConfidence50 += a['test confidence'][1]
                        testConfidence20 += a['test confidence'][2]
                        testConfidence10 += a['test confidence'][3]

                        trainFidelityOri += a['train Model Fidelity (Ori)']
                        testFidelityOri += a['test Model Fidelity (Ori)']
                        trainFidelitySax += a['train Model Fidelity (Sax)']
                        testFidelitySax += a['test Model Fidelity (Sax)']
                        #trainFidelityOri += helper.modelFidelity(a['train Predictions'], (oriPredictions[fold]['Train Predictions']+1))
                        #testFidelityOri += helper.modelFidelity(a['test Predictions'], (oriPredictions[fold]['Test Predictions'] +1))
                        #trainFidelitySax += helper.modelFidelity(a['train Predictions'], (saxPredictions[fold]['Train Predictions']+1))
                        #testFidelitySax +=helper.modelFidelity(a['test Predictions'], (saxPredictions[fold]['Test Predictions']+1))


                        #add +1 for old results
                        confidenceFidelityTrain = helper.fidelityConfidenceGCR(a['train Biggest Scores'], a['train Predictions'], saxPredictions[fold]['Train Predictions'])
                        confidenceFidelityTest = helper.fidelityConfidenceGCR(a['test Biggest Scores'], a['test Predictions'], saxPredictions[fold]['Test Predictions'])

                        trainConfidence80F += confidenceFidelityTrain[0]
                        trainConfidence50F += confidenceFidelityTrain[1]
                        trainConfidence20F += confidenceFidelityTrain[2]
                        trainConfidence10F += confidenceFidelityTrain[3]
                        testConfidence80F += confidenceFidelityTest[0]
                        testConfidence50F += confidenceFidelityTest[1]
                        testConfidence20F += confidenceFidelityTest[2]
                        testConfidence10F += confidenceFidelityTest[3]

                        fold += 1
                        


                    testAcc = testAcc/folds
                    testPrec = testPrec/folds
                    testRec = testRec/folds
                    testF1 = testF1/folds
                    
                    trainFidelityOri = trainFidelityOri/folds
                    testFidelityOri = testFidelityOri/folds
                    trainFidelitySax = trainFidelitySax/folds
                    testFidelitySax =testFidelitySax/folds   
                    trainConfidence80 = trainConfidence80/folds
                    trainConfidence50 = trainConfidence50/folds
                    trainConfidence20 = trainConfidence20/folds
                    trainConfidence10 = trainConfidence10/folds
                    testConfidence80 = testConfidence80/folds
                    testConfidence50 = testConfidence50/folds
                    testConfidence20 = testConfidence20/folds
                    testConfidence10 = testConfidence10/folds
                    trainConfidence80F = trainConfidence80F/folds
                    trainConfidence50F = trainConfidence50F/folds
                    trainConfidence20F = trainConfidence20F/folds
                    trainConfidence10F = trainConfidence10F/folds
                    testConfidence80F = testConfidence80F/folds
                    testConfidence50F = testConfidence50F/folds
                    testConfidence20F = testConfidence20F/folds
                    testConfidence10F = testConfidence10F/folds


                    combiString = 'hl'+str(c1)+str(c2)
                    base = gcrResults[k][p][g][combiString]

                    if not relativValues:
                        base['Acc'].append(testAcc)
                        base['Precicion'].append(testPrec )
                        base['Recall'].append(testRec)
                        base['F1'].append(testF1)
                    else: 
                        base['Acc'].append(testAcc - saxAcc)
                        base['Precicion'].append(testPrec - saxAcc)
                        base['Recall'].append(testRec - saxAcc)
                        base['F1'].append(testF1 - saxAcc)

                    base['train fidelity (ori)'].append(trainFidelityOri)
                    base['test fidelity (ori)'].append(testFidelityOri)
                    base['train fidelity (sax)'].append(trainFidelitySax)
                    base['test fidelity (sax)'].append(testFidelitySax)

                    # #confidence ranks top80Acc, top50Acc, top20Acc, top10Acc
                    base['train confidence 80'].append(trainConfidence80)
                    base['train confidence 50'].append(trainConfidence50)
                    base['train confidence 20'].append(trainConfidence20)
                    base['train confidence 10'].append(trainConfidence10) 
                    base['test confidence 80'].append(testConfidence80)
                    base['test confidence 50'].append(testConfidence50)   
                    base['test confidence 20'].append(testConfidence20)
                    base['test confidence 10'].append(testConfidence10)    
                    base['train confidence 80F'].append(trainConfidence80F)
                    base['train confidence 50F'].append(trainConfidence50F)
                    base['train confidence 20F'].append(trainConfidence20F)
                    base['train confidence 10F'].append(trainConfidence10F) 
                    base['test confidence 80F'].append(testConfidence80F)
                    base['test confidence 50F'].append(testConfidence50F)   
                    base['test confidence 20F'].append(testConfidence20F)
                    base['test confidence 10F'].append(testConfidence10F)  

                    combisAcc.append(testAcc)
                    combisPrec.append(testPrec)
                    combisRec.append(testRec)
                    combisF1.append(testF1)
                    combisTestConfidence80.append(testConfidence80)
                    combisTestConfidence50.append(testConfidence50)
                    combisTestConfidence20.append(testConfidence20)
                    combisTestConfidence10.append(testConfidence10)
                    combisTrainConfidence80.append(trainConfidence80)
                    combisTrainConfidence50.append(trainConfidence50)
                    combisTrainConfidence20.append(trainConfidence20)
                    combisTrainConfidence10.append(trainConfidence10)
                    combisTrainFidelityOri.append(trainFidelityOri)
                    combisTestFidelityOri.append(testFidelityOri)
                    combisTrainFidelitySax.append(trainFidelitySax)
                    combisTestFidelitySax.append(testFidelitySax)


                    if saxAcc >= goodLimit:
                        k = 'good'
                    else:
                        k = 'bad'
                    
                    base = gcrResults[k][p][g][combiString]


                    if not relativValues:
                        base['Acc'].append(testAcc)
                        base['Precicion'].append(testPrec )
                        base['Recall'].append(testRec)
                        base['F1'].append(testF1)
                    else: 
                        base['Acc'].append(testAcc - saxAcc)
                        base['Precicion'].append(testPrec - saxAcc)
                        base['Recall'].append(testRec - saxAcc)
                        base['F1'].append(testF1 - saxAcc)

                    base['train fidelity (ori)'].append(trainFidelityOri)
                    base['test fidelity (ori)'].append(testFidelityOri)
                    base['train fidelity (sax)'].append(trainFidelitySax)
                    base['test fidelity (sax)'].append(testFidelitySax)

                    # #confidence ranks top80Acc, top50Acc, top20Acc, top10Acc 
                    base['train confidence 80'].append(trainConfidence80)
                    base['train confidence 50'].append(trainConfidence50)
                    base['train confidence 20'].append(trainConfidence20)
                    base['train confidence 10'].append(trainConfidence10) 
                    base['test confidence 80'].append(testConfidence80)
                    base['test confidence 50'].append(testConfidence50)   
                    base['test confidence 20'].append(testConfidence20)
                    base['test confidence 10'].append(testConfidence10)

                    base['train confidence 80F'].append(trainConfidence80F)
                    base['train confidence 50F'].append(trainConfidence50F)
                    base['train confidence 20F'].append(trainConfidence20F)
                    base['train confidence 10F'].append(trainConfidence10F) 
                    base['test confidence 80F'].append(testConfidence80F)
                    base['test confidence 50F'].append(testConfidence50F)   
                    base['test confidence 20F'].append(testConfidence20F)
                    base['test confidence 10F'].append(testConfidence10F)  
            
            combisAcc = rankData(combisAcc)
            combisPrec = rankData(combisPrec)
            combisRec = rankData(combisRec)
            combisF1 = rankData(combisF1)
            combisTestConfidence80 = rankData(combisTestConfidence80)
            combisTestConfidence50 = rankData(combisTestConfidence50)
            combisTestConfidence20 = rankData(combisTestConfidence20)
            combisTestConfidence10 = rankData(combisTestConfidence10)
            combisTrainConfidence80 = rankData(combisTrainConfidence80)
            combisTrainConfidence50 = rankData(combisTrainConfidence50)
            combisTrainConfidence20 = rankData(combisTrainConfidence20)
            combisTrainConfidence10 = rankData(combisTrainConfidence10)
            combisTrainFidelityOri = rankData(combisTrainFidelityOri)
            combisTestFidelityOri = rankData(combisTestFidelityOri)
            combisTrainFidelitySax = rankData(combisTrainFidelitySax)
            combisTestFidelitySax = rankData(combisTestFidelitySax)

            index = 0
            for c1 in combis:
                for c2 in combis:

                    p = 'ranking'
                    k = 'all'

                    combiString = 'hl'+str(c1)+str(c2)
                    base = gcrResults[k][p][g][combiString]

                    base['Acc'].append(combisAcc[index])
                    base['Precicion'].append(combisPrec[index])
                    base['Recall'].append(combisRec[index])
                    base['F1'].append(combisF1[index])

                    base['train fidelity (ori)'].append(combisTrainFidelityOri[index])
                    base['test fidelity (ori)'].append(combisTestFidelityOri[index])
                    base['train fidelity (sax)'].append(combisTrainFidelitySax[index])
                    base['test fidelity (sax)'].append(combisTestFidelitySax[index])

                    # #confidence ranks top80Acc, top50Acc, top20Acc, top10Acc
                    base['train confidence 80'].append(combisTrainConfidence80[index])
                    base['test confidence 80'].append(combisTestConfidence80[index])
                    base['train confidence 50'].append(combisTrainConfidence50[index])
                    base['test confidence 50'].append(combisTestConfidence50[index]) 
                    base['train confidence 20'].append(combisTrainConfidence20[index])
                    base['test confidence 20'].append(combisTestConfidence20[index])   
                    base['train confidence 10'].append(combisTrainConfidence10[index])
                    base['test confidence 10'].append(combisTestConfidence10[index])   

                    if saxAcc >= goodLimit:
                            k = 'good'
                    else:
                            k = 'bad'

                    base = gcrResults[k][p][g][combiString]

                    base['Acc'].append(combisAcc[index])
                    base['Precicion'].append(combisPrec[index])
                    base['Recall'].append(combisRec[index])
                    base['F1'].append(combisF1[index])

                    base['train fidelity (ori)'].append(combisTrainFidelityOri[index])
                    base['test fidelity (ori)'].append(combisTestFidelityOri[index])
                    base['train fidelity (sax)'].append(combisTrainFidelitySax[index])
                    base['test fidelity (sax)'].append(combisTestFidelitySax[index])

                    # #confidence ranks top80Acc, top50Acc, top20Acc, top10Acc 
                    base['train confidence 80'].append(combisTrainConfidence80[index])
                    base['test confidence 80'].append(combisTestConfidence80[index])
                    base['train confidence 50'].append(combisTrainConfidence50[index])
                    base['test confidence 50'].append(combisTestConfidence50[index]) 
                    base['train confidence 20'].append(combisTrainConfidence20[index])
                    base['test confidence 20'].append(combisTestConfidence20[index])   
                    base['train confidence 10'].append(combisTrainConfidence10[index])
                    base['test confidence 10'].append(combisTestConfidence10[index])   

                    index += 1


    directory = './presults/'
    files = []
 
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            # add conditions if needed!
            if symbolCount == 0:
                files.append(f[len(directory):-4:1])
            else:
                #f = f.replace("\uf03a", ":")
                if "symbols "+ str(symbolCount) +" -layers "+ str(numOfAttentionLayers) in filename and 'headers ' + str(header) in filename:
                    files.append(f[len(directory):-4:1])

    baseStatsOri = dict() #
    baseStatsOri['Acc'] = []
    baseStatsOri['Precicion'] = []
    baseStatsOri['Recall'] = []
    baseStatsOri['F1'] = []
    baseStatsOri['train fidelity (ori)'] = []
    baseStatsOri['val fidelity (ori)'] = []
    baseStatsOri['test fidelity (ori)'] = []
    baseStatsOri['test permutationsEntropy'] = []
    baseStatsOri['test spectralEntropy'] = []
    baseStatsOri['test svdEntropy'] = []
    baseStatsOri['test approximateEntropy'] = []
    baseStatsOri['test sampleEntropy'] = []
    baseStatsOri['test CE'] = []
    baseStatsOri['test shifts'] = []  


    baseStatsSAX = dict() #
    baseStatsSAX['Acc'] = []
    baseStatsSAX['Precicion'] = []
    baseStatsSAX['Recall'] = []
    baseStatsSAX['F1'] = []
    baseStatsSAX['train fidelity (ori)'] = []
    baseStatsSAX['val fidelity (ori)'] = []
    baseStatsSAX['test fidelity (ori)'] = []
    baseStatsSAX['test permutationsEntropy'] = []
    baseStatsSAX['test spectralEntropy'] = []
    baseStatsSAX['test svdEntropy'] = []
    baseStatsSAX['test approximateEntropy'] = []
    baseStatsSAX['test sampleEntropy'] = []
    baseStatsSAX['test CE'] = []
    baseStatsSAX['test shifts'] = []  

    shapeletsBaseOri = dict() #
    createShapeletBase(shapeletsBaseOri)
    shapeletsBaseSAX = dict() #
    createShapeletBase(shapeletsBaseSAX)


    lasaBaseResults = dict() #
    createLasaStructure(lasaBaseResults)
    lasaFidelityResults = dict() #
    createLasaStructure(lasaFidelityResults)

    giaBaseResults = dict() #
    createGiaStructure(giaBaseResults)
    giaFidelityResults = dict() #
    createGiaStructure(giaFidelityResults)

    lasaShapeletResults = dict() #
    createShapeletLasaStructure(lasaShapeletResults)


    gcrOriginalResults = dict() #
    createGCRStrcture(gcrOriginalResults)

    gcrCountPenaltyResults = dict() #
    createGCRStrcture(gcrCountPenaltyResults)
    gcrEntropyPenaltyResults = dict() #
    createGCRStrcture(gcrEntropyPenaltyResults)

    gcrTResults10 = dict() #
    createGCRStrcture(gcrTResults10)
    gcrTResults13 = dict() #
    createGCRStrcture(gcrTResults13)
    gcrTResults16 = dict() #
    createGCRStrcture(gcrTResults16)

    consistancies = []


    for f in files:
        results = helper.load_obj(directory+f)
        res = dict()
        for index, v in np.ndenumerate(results):
            res = v
        
        print("Start ########################################")
        print(f)

        #Original
        fillBaseStates(baseStatsOri, "Original",res)
        #SAX
        fillBaseStates(baseStatsSAX, "SAX", res)

        if(True):
            #base shapelets
            for sOri in res['Original Shapelets ']:
                if sOri.split(" ")[-1] == "2":
                    s = 2
                else:
                    s = 0.3
                fillShapeletBase(shapeletsBaseOri, 'Original Shapelets ', sOri, res, s,res['Original'][''], res['SAX'][''])
            sEnd = "2"
            for sOri in res['SAX Shapelets ']:
                if sOri.split(" ")[-1] == "2":
                    s = 2
                else:
                    sEnd = sOri.split(" ")[-1]
                    s = 0.3
                fillShapeletBase(shapeletsBaseSAX, 'SAX Shapelets ', sOri, res, s,res['Original'][''], res['SAX'][''])

        if True:
            #lasa all combis
            fillLASACombisValue(lasaBaseResults, 'LASA Avg Inter ', 'avg', lasaAvgTresholds, res, baseStatsSAX['Acc'][-1])
            fillLASACombisValue(lasaBaseResults, 'LASA Max Inter ', 'max', lasaMaxTresholds, res, baseStatsSAX['Acc'][-1])
        

        #lasa fidelity all combis
        #fillLASACombisValue(lasaFidelityResults, 'LASA Fidelity Avg Inter ', 'avg', lasaAvgTresholds, res, baseStatsSAX['Acc'][-1])
        #fillLASACombisValue(lasaFidelityResults, 'LASA Fidelity Max Inter ', 'max', lasaMaxTresholds, res, baseStatsSAX['Acc'][-1])

        if(False):
            fillLASACombisValue(giaBaseResults, 'GIA ', 'avg', giaThresholds, res, baseStatsSAX['Acc'][-1], gia=True)

            fillLASAShapeletCombis(lasaShapeletResults, 'LASA Avg Inter Shapelets ','avg', lasaAvgTresholds, res, baseStatsSAX['Acc'][-1], sEnd)
            fillLASAShapeletCombis(lasaShapeletResults, 'LASA Max Inter Shapelets ','max', lasaMaxTresholds, res, baseStatsSAX['Acc'][-1], sEnd)
        
        if True:
            fillGCRCombis(gcrOriginalResults, "Original GCR  ", "Combinations: ", res, baseStatsSAX['Acc'][-1], res['Original'][''], res['SAX'][''])
            fillGCRCombis(gcrCountPenaltyResults, "Penalty GCR  ", "PenaltyMode: counter; Combination: ", res, baseStatsSAX['Acc'][-1], res['Original'][''], res['SAX'][''])
            fillGCRCombis(gcrEntropyPenaltyResults, "Penalty GCR  ", "PenaltyMode: entropy; Combination: ", res, baseStatsSAX['Acc'][-1], res['Original'][''], res['SAX'][''])

            fillGCRCombis(gcrTResults10, "Threshold GCR  ", "Threshold: 1.0; Combination: ", res, baseStatsSAX['Acc'][-1], res['Original'][''], res['SAX'][''])
            fillGCRCombis(gcrTResults13, "Threshold GCR  ", "Threshold: 1.3; Combination: ", res, baseStatsSAX['Acc'][-1], res['Original'][''], res['SAX'][''])
            fillGCRCombis(gcrTResults16, "Threshold GCR  ", "Threshold: 1.6; Combination: ", res, baseStatsSAX['Acc'][-1], res['Original'][''], res['SAX'][''])


        consistancies.append(res['consistancy dict'])


    results = dict()

    results['files'] = files
    results['ori'] = baseStatsOri
    results['sax'] = baseStatsSAX
    results['shapelets ori'] = shapeletsBaseOri
    results['shapelets sax'] = shapeletsBaseSAX
    results['lasa'] = lasaBaseResults
    results['lasaFidelity'] = lasaFidelityResults
    results['lasaShapelets'] = lasaShapeletResults
    results['gcr'] = gcrOriginalResults
    results['gcr counter'] = gcrCountPenaltyResults
    results['gcr entropy'] = gcrEntropyPenaltyResults
    results['gcr t10'] = gcrTResults10
    results['gcr t13'] = gcrTResults13
    results['gcr t16'] = gcrTResults16
    results['consistancy'] = consistancies
    results['gia'] = giaBaseResults


    helper.save_obj(results, "./aresults/fullProcessedResults-s"+ str(symbolCount) +"-l"+ str(numOfAttentionLayers)+"-h"+ str(header))
    #print(results)
    return results