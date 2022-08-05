from tslearn.datasets import UCR_UEA_datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import cwru2
import numpy as np
from os import walk
from random import randint
from time import sleep

#select one dataset
def datasetSelector(dataset, seed_Value, number, takeName = True, use_cache=True):
    if dataset == 'utc':
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes = doUTC(seed_Value, number, takeName = takeName, use_cache=use_cache)
    elif dataset == 'gearbox':
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes = doGearBox(seed_Value)
    elif dataset == 'cwru':
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes = doCRWU(seed_Value)
    elif dataset == 'counting':
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes = doCounting(seed_Value)
    elif dataset == 'frequency':
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes= doFrequency(seed_Value)
    else:
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes = []

    y_train = np.array(y_train)
    y_train = y_train.astype(float)
    y_test = np.array(y_test)
    y_test = y_test.astype(float)
    X_test = np.array(X_test)
    X_test = X_test.astype(float)
    X_train = np.array(X_train)
    X_train = X_train.astype(float)   
    y_testy = np.array(y_testy)
    y_trainy = np.array(y_trainy)

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes




def doUTC(seed_value, number, takeName = True, retry=0, use_cache=True):
    try:
        datasets = UCR_UEA_datasets(use_cache=use_cache)
        dataset_list = datasets.list_univariate_datasets()
        if takeName:
            print(str(number) + " WRONG#########")
            datasetName = number
        else:
            print(str(number) + " RIGHT#########")
            datasetName = dataset_list[number]
        
        X_train, y_trainy, X_test, y_testy = datasets.load_dataset(datasetName)
        #X_train, y_trainy, X_test, y_testy = datasets.load_dataset('SyntheticControl')
        
        setY = list(set(y_testy))
        setY.sort()
        print(setY)

        num_of_classes = len(set(y_testy))
        seqSize = len(X_train[0])

        X_train, y_trainy = shuffle(X_train, y_trainy, random_state = seed_value)

        y_train = []
        print(num_of_classes)
        for y in y_trainy:
            y_train_puffer = np.zeros(num_of_classes)
            y_train_puffer[setY.index(y)] = 1
            y_train.append(y_train_puffer)

        y_trainy = np.argmax(y_train,axis=1) +1 
            
        y_test = []
        for y in y_testy:
            y_puffer = np.zeros(num_of_classes)
            y_puffer[setY.index(y)] = 1
            y_test.append(y_puffer)
            
        y_testy = np.argmax(y_test,axis=1) +1 
    
    except Exception as e:
        print(e)
        if retry < 5:
            sleep(randint(10,30))

            if retry == 4:
                return doUTC(seed_value, number, takeName = takeName, retry=retry+1, use_cache=False)
            else:
                return doUTC(seed_value, number, takeName = takeName, retry=retry+1) 

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, datasetName, num_of_classes

def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


def doGearBox(seed_value):
    data_path1 = './Datasets/GearBox/BrokenTooth Data/'
    data_path2 = './Datasets/GearBox/Healthy Data/'
    dataName = 'GearBox'
    chunk_size = 128
    seqSize = chunk_size
    features = []
    labels = []
    for (dirpath, dirnames, filenames) in walk(data_path1):
        print(filenames)
        for file in filenames:
            data = pd.read_csv(data_path1+file, sep="	", header=None)
            feature = data.copy()
            feature.pop(4)
            for chunk in split_dataframe(feature, chunk_size = chunk_size)[:-1]:
                features.append(chunk.to_numpy().flatten())
                labels.append(1)
            #print(abalone_features)
        break

    for (dirpath, dirnames, filenames) in walk(data_path2):
        print(filenames)
        for file in filenames:
            data = pd.read_csv(data_path2+file, sep="	", header=None)
            feature = data.copy()
            feature.pop(4)
            for chunk in split_dataframe(feature, chunk_size = chunk_size)[:-1]:
                features.append(chunk.to_numpy().flatten())
                labels.append(0)
            #print(abalone_features)
        break
    features = np.array(features)
    labels = np.array(labels)
    features.shape

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, shuffle=True, random_state=seed_value)

    y_trainy = y_train +1
    y_train = []
    X_train = X_train
    #y_testy_full = data_test[:,-1].astype(int) 
    y_testy = y_test +1
    y_test = []
    X_test = X_test
    num_of_classes = len(set(y_trainy))

    for y in y_trainy:
        y_train_puffer = np.zeros(num_of_classes)
        y_train_puffer[y-1] = 1
        y_train.append(y_train_puffer)

    for y in y_testy:
        y_puffer = np.zeros(num_of_classes)
        y_puffer[y-1] = 1
        y_test.append(y_puffer)

    y_train = np.array(y_train)
    y_train = y_train.astype(float)
    y_test_full = np.array(y_test)
    y_test_full = y_test_full.astype(float)
    y_test = y_test_full  
    y_test = y_test.astype(float)

    print(X_test.shape)
    print(X_train.shape)
    print(y_test.shape)
    print(y_train.shape)
    seqSize = X_train.shape[1]

    X_test = X_test.astype(float)
    X_train = X_train.astype(float)

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes


def doCRWU(seed_value):
    dataSize = 296
    seqSize = dataSize
    data = cwru2.CWRU("12FanEndFault", dataSize, 0.8, 1, '1797','1750')
    dataName = 'cwru'

    # Create dummies for the labels
    print(data.y_train)
    data.y_train = pd.DataFrame(data.y_train, columns=['label'])
    dummies = pd.get_dummies(data.y_train['label']) # Classification
    products = dummies.columns
    y_train = dummies.values

    data.y_test = pd.DataFrame(data.y_test, columns=['label'])
    dummies = pd.get_dummies(data.y_test['label']) # Classification
    products = dummies.columns
    y_test = dummies.values

    X_train = data.X_train
    X_test = data.X_test
    print(y_train)

    y_trainy = np.argmax(y_train,axis=1) +1  
    y_train = []
    X_train = X_train
    #y_testy_full = data_test[:,-1].astype(int) 
    y_testy = np.argmax(y_test,axis=1) +1
    y_test = []
    X_test = X_test

    X_train, y_trainy = shuffle(X_train, y_trainy, random_state = seed_value)


    num_of_classes = len(set(y_trainy.tolist()))

    for y in y_trainy:
        y_train_puffer = np.zeros(num_of_classes)
        y_train_puffer[y-1] = 1
        y_train.append(y_train_puffer)

    for y in y_testy:
        y_puffer = np.zeros(num_of_classes)
        y_puffer[y-1] = 1
        y_test.append(y_puffer)

    y_train = np.array(y_train)
    y_train = y_train.astype(float)
    y_test_full = np.array(y_test)
    y_test_full = y_test_full.astype(float)
    y_test = y_test_full  
    y_test = y_test.astype(float)

    print(X_test.shape)
    print(X_train.shape)
    print(y_test.shape)
    print(y_train.shape)

    X_test = X_test.astype(float)
    X_train = X_train.astype(float)

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes

def doCounting(seed_value):
    symbols = 2
    dataLen = 10
    seqSize = dataLen
    dataName = 'counting'
    size = pow(symbols, dataLen)
    #dataSet = np.zeros((size,dataLen))
    dataSet = []
    labelSet = []

    n = -1
    for i in range(size):
        binary = format(i, '0'+str(dataLen)+'b')
        #if int(binary.count('1')) == 4 or int(binary.count('1')) == 5:
        if True:
            dataSet.append(np.zeros(dataLen))
            n+=1
            for j in range(dataLen):
                if int(binary[j]) == 0:
                    dataSet[n][j] = -1
                else:
                    dataSet[n][j] = int(binary[j])
            #labelSet.append(int(binary.count('1') > 4))
            labelSet.append(int(binary.count('1')))
    num_of_classes = len(set(labelSet))
    dataSet = np.array(dataSet)

    X_train, X_test, y_train, y_test = train_test_split(dataSet, labelSet, test_size=0.50, shuffle=True, random_state=seed_value)


    y_trainy = np.array(y_train) +1
    y_train = []
    X_train = X_train
    #y_testy_full = data_test[:,-1].astype(int) 
    y_testy = np.array(y_test) +1
    y_test = []
    X_test = X_test
    

    for y in y_trainy:
        y_train_puffer = np.zeros(num_of_classes)
        y_train_puffer[y-1] = 1
        y_train.append(y_train_puffer)

    for y in y_testy:
        y_puffer = np.zeros(num_of_classes)
        y_puffer[y-1] = 1
        y_test.append(y_puffer)

    y_train = np.array(y_train)
    y_train = y_train.astype(float)
    y_test_full = np.array(y_test)
    y_test_full = y_test_full.astype(float)
    y_test = y_test_full  
    y_test = y_test.astype(float)

    print(X_test.shape)
    print(X_train.shape)
    print(y_test.shape)
    print(y_train.shape)

    X_test = X_test.astype(float)
    X_train = X_train.astype(float)

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes

def doFrequency(seed_value):
    #TODO
    X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes = []
    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes