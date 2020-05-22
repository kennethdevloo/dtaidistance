from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from enum import Enum
import numpy as np
import random
import dtaidistance 
from dtaidistance import alignment, clustering, dtw
import math
import scipy.sparse as sparse
import dtaidistance.quantizer_c as q
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram,linkage
from matplotlib import pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from scipy import stats
import random
import time    

def _size_cond( size):
        n = int(size)
        return int((n * (n - 1)) / 2)


def condenseDists(dists):
    dists_cond = np.zeros(_size_cond(dists.shape[0]))
    idx = 0
    for r in range(dists.shape[0] - 1):
        dists_cond[idx:idx + dists.shape[0] - r - 1] = dists[r, r + 1:]
        idx += dists.shape[0] - r - 1
    return dists_cond

def zNormalize(X):
    return (X-np.mean(X,axis=1,keepdims=True)/np.std(X, axis=1,keepdims=True))

def loadDataSet(file1, file2=None, maxTrainDataPoints = None, maxTestDataPoints = None, seed = 0):
    X = np.loadtxt(file1)


    X2 = np.loadtxt(file2)

    Y= X[:,0]
    X_scaled = zNormalize(X[:,1:len(X[0])])
   
    Y2= X2[:,0]
    X2_scaled = zNormalize(X2[:,1:len(X2[0])])

    random.seed(seed)
    if maxTrainDataPoints is not None and maxTrainDataPoints < len(X):
        nb_series = len(X_scaled)
        nb_samples=maxTrainDataPoints
        indiceList = [ i for i in range(nb_series)]
        random.shuffle(indiceList)
        samples = indiceList[0:nb_samples]
        X_scaled = X_scaled[samples]
        Y = Y[samples] 

    if maxTestDataPoints is not None and maxTestDataPoints < len(X2):
        nb_series = len(X2_scaled)
        nb_samples=maxTestDataPoints
        indiceList = [ i for i in range(nb_series)]
        random.shuffle(indiceList)
        samples = indiceList[0:nb_samples]
        X2_scaled2 = X2_scaled[samples]
        Y2 = Y2[samples] 

#    print(X2_scaled[2])

    return X_scaled, Y, X2_scaled, Y2 


def loadYogaDataSetForClustering(traionSize, testSize):
    return loadDataSet('Yoga_TRAIN.txt', 'Yoga_TEST.txt',traionSize, testSize)

def loadElectricDevicesDataSetForClustering(traionSize, testSize):
    return loadDataSet('ElectricDevices_TRAIN.txt', 'ElectricDevices_TEST.txt',traionSize, testSize)

def loadFaceUCRDataSetForClustering(traionSize, testSize):
    return loadDataSet('FacesUCR_TRAIN.txt', 'FacesUCR_TEST.txt',traionSize, testSize)

def loadEcg500DataSetForClustering(traionSize, testSize):
    return loadDataSet('ECG5000_TRAIN.txt', 'ECG5000_TEST.txt',traionSize, testSize)

def loadInsectWingDataSetForClustering(traionSize, testSize):
    return loadDataSet('InsectWingbeatSound_TRAIN.txt', 'InsectWingbeatSound_TEST.txt',traionSize, testSize)

def loadStarLightDataSetForClustering(traionSize, testSize):
    return loadDataSet('StarLightCurves_TRAIN.txt', 'StarLightCurves_TEST.txt',traionSize, testSize)
    

#based on https://thispointer.com/python-how-to-find-keys-by-value-in-dictionary/
def getKeys(dictOfElements):
    listOfItems = dictOfElements.items()
    keys = list()
    for item  in listOfItems:
        keys.append(item[0]) 
    return keys

def changeKeyName(dictionary, old_key, new_key):
    if new_key in dictionary:
        dictionary[new_key]=dictionary[new_key].union(dictionary.pop(old_key))  
    else:
        dictionary[new_key] = dictionary.pop(old_key)

def getKeyByValue(dictOfElements, valueToFind):
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if valueToFind in item[1]:
            return item[0]


def jaccard_similarity(s1, s2):
    intersectionL = len(s1.intersection(s2))
    unionL = len(s1) + len(s2) - intersectionL
    return float(intersectionL) / unionL


def convertToLabelArray(idx, len):
    ret = np.zeros((len), np.int32)
    items = idx.items()
    for id in range(0, len):
        for item in items :
            if id in item[1]:
                ret[id]=item[0]
                break
    return ret


def triangularTest(dists, attempts):
    if dists.shape[0]<3:
        return 1
    ratio = 0
    right=0
    wrong=0
    for i in range(attempts):
        s = [ random.randint(0, int(dists.shape[0])-1) for j in range(0,3)]
        while s[0] == s[2] or s[1] == s[2] or s[0] == s[1]: 
            s = [ random.randint(0, int(dists.shape[0])-1) for j in range(0,3)]
        
        triangle = (dists[min(s[0], s[1]), max(s[0], s[1])] + dists[min(s[2], s[1]), max(s[2], s[1])])>=dists[min(s[2], s[0]), max(s[2], s[0])]
        if triangle:
            right = right + 1
        else:
            wrong = wrong + 1
    ratio = float (right)/attempts
    return ratio

def scoreDistanceMatrices(truth, pred, nb_series, triangleTries = 10000):
    conDistTruth=condenseDists(truth)
    #print(pred[0:6, 0:6])
    #print(truth[0:6, 0:6])
    conDistPred=condenseDists(pred)
    rmse = 0
    me = 0
    spearManCorrelation = 0
    spearManP = 0
    pearson = 0
    triangularisationScore  = 0
    rmse = np.sqrt(np.mean((conDistPred-conDistTruth)**2))
    me = np.mean(conDistTruth-conDistPred)
    
    #print(conDistTruth[0:6])
    #print(conDistPred[0:6])

    spearManCorrelation, spearManP = stats.spearmanr((conDistTruth), (conDistPred))
#    print('Pearson', stats.pearsonr(np.argsort(conDistTruth), np.argsort(conDistPred)))
    triangularisationScore = triangularTest(pred, triangleTries)/triangularTest(truth, triangleTries)

    return me, rmse, spearManCorrelation, spearManP, triangularisationScore

#idx1 is th reference truth
def equaliseClusterLabelsAndCalculateScores(idx1, idx2, X):
    jaccard=0
    ari=0

    keyList1 = getKeys(idx1)
    keyList2 = getKeys(idx2)

    keyChanges = dict()

    dists = np.full((len(idx2), len(idx1)), -1, dtype=np.float32)

    for key2Id in range(len(keyList2)):
        key2=keyList2[key2Id]
        for key1Id in range(len(keyList1)):
            key1=keyList1[key1Id]
            sim = jaccard_similarity(idx1[key1], idx2[key2])
     
            dists[key2Id, key1Id]=sim
    #print(dists)

    for i in range(len(keyList2)):
        max_value = np.max(dists)
        max_idxs = np.argwhere(dists == max_value)

        jaccard=max_value+jaccard
        keyChanges[keyList2[max_idxs[0,0]]]= keyList1[max_idxs[0,1]]
        dists[max_idxs[0,0], :]=-1
        dists[:, max_idxs[0,1]]=-1

    newClust ={}
    for i in keyChanges.items():
        newClust[i[1]]=idx2[i[0]]
#        changeKeyName(idx2, i[0], i[1])
    idx2=newClust

    pred = convertToLabelArray(idx2, X.shape[0])
    truth = convertToLabelArray(idx1, X.shape[0]) 
    ari = adjusted_rand_score(truth, pred)

    keyList2 = getKeys(idx2)
    return jaccard, ari


def equalClusterLabels(idx1, idx2):
    keys = getKeys(idx1)
    for key in keys:
        changeKeyName(idx2, getKeyByValue(idx2,key), key)
   

def printClusters(idx):
    listOfItems = idx.items()
    for item in listOfItems:
        print (item)



def clusterECG5000Test():
    traionSize = 500
    valSize = 1000
    testSize = 3500
    seed = 0
    distParams={'window': 14, 'psi': 0}
    quantizationDistParams={'window': 10, 'psi': 0}
    nwDistParams={'window': 2, 'psi': 0}
    kmeansWindowSize=2
    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    pqClusterParams = {'k':40000,'quantizer_usage':clustering.QuantizerUsage.TOP_K_ONLY_AT_INITIALISATION}
    
    qParams=[]
    qParams.append(q.ProductQuantiserParameters(50,500,distParams=distParams, 
        subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP,
        kmeansWindowSize=kmeansWindowSize,
        distanceCalculation=q.DISTANCECALCULATION.ASYMMETRIC,
        computeDistanceCorrection = True,
        quantizationDistParams=quantizationDistParams,
        max_iters=2))
        
    #qParams.append(q.ProductQuantiserParameters(9,4,computeDistanceCorrection=False))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(20,40,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))

    qNWPileParams=[]
    qNWPileParams.append(q.ProductQuantiserParameters(20,20,
    quantizerType=q.QuantizerType.VQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
    
    XTrain,YTrain,XTest,YTest=loadEcg500DataSetForClustering(traionSize, testSize+valSize)
    XVal, YVal, XTest, YTest = take2RandDataParts(XTest,YTest,valSize, testSize, seed)
    #print(YVal[1500:1700], np.unique(YVal))
    distanceAndClusterTests(XTrain,XVal, qParams, distParams, generalClusterParams, pqClusterParams)
    

def clusterStarLightTest():
    traionSize = 500
    valSize = 2000
    testSize = 5000
    seed = 0
    distParams={'window': 30, 'psi': 0}
    quantizationDistParams={'window': 15, 'psi': 0}
    nwDistParams={'window': 2, 'psi': 0}
    kmeansWindowSize=2
    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    pqClusterParams = {'k':199800,'quantizer_usage':clustering.QuantizerUsage.TOP_K_ONLY_AT_INITIALISATION}
    
    qParams=[]
    qParams.append(q.ProductQuantiserParameters(128,500,distParams=distParams, 
        subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP,
        kmeansWindowSize=kmeansWindowSize,
        distanceCalculation=q.DISTANCECALCULATION.ASYMMETRIC_KEOGH_WHEM_0DIST,
        computeDistanceCorrection = True,
        quantizationDistParams=quantizationDistParams,
        max_iters=0))
        
    #qParams.append(q.ProductQuantiserParameters(9,4,computeDistanceCorrection=False))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(20,40,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))

    qNWPileParams=[]
    qNWPileParams.append(q.ProductQuantiserParameters(20,20,
    quantizerType=q.QuantizerType.VQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
    
    XTrain,YTrain,XTest,YTest=loadStarLightDataSetForClustering(traionSize, testSize+valSize)
    XVal, YVal, XTest, YTest = take2RandDataParts(XTest,YTest,valSize, testSize, seed)
    #print(YVal[1500:1700], np.unique(YVal))
    distanceAndClusterTests(XTrain,XVal, qParams, distParams, generalClusterParams, pqClusterParams)


def clusterInsecTwingBeatsTest():
    traionSize = 1000
    valSize = 5000
    testSize = 5000
    seed = 0
    distParams={'window': 3, 'psi': 0}
    quantizationDistParams={'window': 2, 'psi': 0}
    nwDistParams={'window': 2, 'psi': 0}
    kmeansWindowSize=2
    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    pqClusterParams = {'k':199800,'quantizer_usage':clustering.QuantizerUsage.TOP_K_ONLY_AT_INITIALISATION}
    
    qParams=[]
    qParams.append(q.ProductQuantiserParameters(15,1000,distParams=distParams, 
        subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP,
        kmeansWindowSize=kmeansWindowSize,
        distanceCalculation=q.DISTANCECALCULATION.ASYMMETRIC_KEOGH_WHEM_0DIST,
        computeDistanceCorrection = True,
        quantizationDistParams=quantizationDistParams,
        max_iters=0))
        
    #qParams.append(q.ProductQuantiserParameters(9,4,computeDistanceCorrection=False))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(20,40,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))

    qNWPileParams=[]
    qNWPileParams.append(q.ProductQuantiserParameters(20,20,
    quantizerType=q.QuantizerType.VQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
    
    XTrain,YTrain,XTest,YTest=loadInsectWingDataSetForClustering(traionSize, testSize+valSize)
    XVal, YVal, XTest, YTest = take2RandDataParts(XTest,YTest,valSize, testSize, seed)
    #print(YVal[1500:1700], np.unique(YVal))
    distanceAndClusterTests(XTrain,XVal, qParams, distParams, generalClusterParams, pqClusterParams)


def clusterYogaTest():
    traionSize = 300
    valSize = 1000
    testSize = 2000
    seed = 0
    distParams={'window': 4, 'psi': 0}
    quantizationDistParams={'window': 3, 'psi': 0}
    nwDistParams={'window': 2, 'psi': 0}
    kmeansWindowSize=2
    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    pqClusterParams = {'k':199800,'quantizer_usage':clustering.QuantizerUsage.TOP_K_ONLY_AT_INITIALISATION}
    
    qParams=[]
    qParams.append(q.ProductQuantiserParameters(15,1000,distParams=distParams, 
        subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP,
        kmeansWindowSize=kmeansWindowSize,
        distanceCalculation=q.DISTANCECALCULATION.ASYMMETRIC_KEOGH_WHEM_0DIST,
        computeDistanceCorrection = True,
        quantizationDistParams=quantizationDistParams,
        max_iters=0))
        
    #qParams.append(q.ProductQuantiserParameters(9,4,computeDistanceCorrection=False))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(20,40,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
  
    XTrain,YTrain,XTest,YTest=loadYogaDataSetForClustering(traionSize, testSize+valSize)
    XVal, YVal, XTest, YTest = take2RandDataParts(XTest,YTest,valSize, testSize, seed)
    #print(YVal[1500:1700], np.unique(YVal))
    distanceAndClusterTests(XTrain,XVal, qParams, distParams, generalClusterParams, pqClusterParams)

def clusterElectricalDevicesTest():
    traionSize = 1000
    valSize = 2000
    testSize = 5000
    seed = 0
    distParams={'window': 15, 'psi': 0}
    quantizationDistParams={'window': 10, 'psi': 0}
    nwDistParams={'window': 2, 'psi': 0}
    kmeansWindowSize=2
    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    pqClusterParams = {'k':199800,'quantizer_usage':clustering.QuantizerUsage.TOP_K_ONLY_AT_INITIALISATION}
    
    qParams=[]
    qParams.append(q.ProductQuantiserParameters(15,1000,distParams=distParams, 
        subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP,
        kmeansWindowSize=kmeansWindowSize,
        distanceCalculation=q.DISTANCECALCULATION.ASYMMETRIC_KEOGH_WHEM_0DIST,
        computeDistanceCorrection = True,
        quantizationDistParams=quantizationDistParams,
        max_iters=0))
        
    #qParams.append(q.ProductQuantiserParameters(9,4,computeDistanceCorrection=False))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(20,40,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
 
    XTrain,YTrain,XTest,YTest=loadElectricDevicesDataSetForClustering(traionSize, testSize+valSize)
    XVal, YVal, XTest, YTest = take2RandDataParts(XTest,YTest,valSize, testSize, seed)
    #print(YVal[1500:1700], np.unique(YVal))
    distanceAndClusterTests(XTrain,XVal, qParams, distParams, generalClusterParams, pqClusterParams)


def trainQuantizer(XTrain, qParams):
    return q.PyProductQuantizer(XTrain, qParams)
    

def clusterFacesTest():
    traionSize = 200
    valSize = 500
    testSize = 1500
    seed = 0
    distParams={'window': 13, 'psi': 0}
    quantizationDistParams={'window': 8, 'psi': 0}
    nwDistParams={'window': 2, 'psi': 0}
    kmeansWindowSize=2
    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    pqClusterParams = {'k':199800,'quantizer_usage':clustering.QuantizerUsage.TOP_K_ONLY_AT_INITIALISATION}
    
    qParams=[]
    qParams.append(q.ProductQuantiserParameters(15,1000,distParams=distParams, 
        subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP,
        kmeansWindowSize=kmeansWindowSize,
        distanceCalculation=q.DISTANCECALCULATION.ASYMMETRIC_KEOGH_WHEM_0DIST,
        computeDistanceCorrection = True,
        quantizationDistParams=quantizationDistParams,
        max_iters=0))
        
    #qParams.append(q.ProductQuantiserParameters(9,4,computeDistanceCorrection=False))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(20,40,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))

    XTrain,YTrain,XTest,YTest=loadFaceUCRDataSetForClustering(traionSize, testSize+valSize)
    XVal, YVal, XTest, YTest = take2RandDataParts(XTest,YTest,valSize, testSize, seed)
    #print(YVal[1500:1700], np.unique(YVal))
    distanceAndClusterTests(XTrain,XVal, qParams, distParams, generalClusterParams, pqClusterParams)


def calcTrue(XTest, distParams):
    return dtaidistance.dtw.distance_matrix_fast(XTest, **distParams)

def calcPred(pq, XTest):
    return pq.constructApproximateDTWDistanceMatrix(XTest)

def distanceAndClusterTests(XTrain,XTest, qParams, distParams, generalClusterParams, pqClusterParams, seed = 0):
    print('Training PQ')
    start = time.clock()
    pq = trainQuantizer(XTrain, qParams)
    end = time.clock()
    print('training time: ', end-start)
    #print('Initialization done')

    distResults = distanceTest(pq,XTest, distParams, seed)
    print ('distanceTestResults',distResults)
    clustResults = clusterTest(pq,XTest, distParams, generalClusterParams, pqClusterParams)
    print ('clusterTestResults',clustResults)


def distanceTest(pq, XTest, distParams, seed = 0):
        
    print('regular')
    start = time.clock()
    truth = calcTrue(XTest, distParams)
    end = time.clock()
    truetime= end-start
    #print('regular Calc done')
    
    print('approximate')
    start = time.clock()
    pred = calcPred(pq,XTest)
    end = time.clock()
    predtime = end - start
    #print('approximating done')
    #print(pred[0:5, 0:5])
    #print(truth[0:5, 0:5])

    me, rmse, spearManCorrelation, spearManP, triangularisationScore = scoreDistanceMatrices(truth,pred, pred.shape[0])
    
    print ('end distance test')
    return {'me':me, 'rmse':rmse, 'spearManCorrelation':spearManCorrelation, 'spearManP':spearManP, 'triangularisationScore':triangularisationScore, 'DTWtime':truetime, 'PQtime':predtime}


def clusterTest(pq,XTest, distParams, generalClusterParams, pqClusterParams ):
    #quantizer model
    model = clustering.HierarchicalWithQuantizer(
        dtaidistance.dtw.distance_fast, distParams,
        **generalClusterParams,
        **pqClusterParams)

    #normal model
    
    modelN = clustering.Hierarchical(
        dtaidistance.dtw.distance_matrix_fast, distParams,
        **generalClusterParams)

    model.setQuantizer(pq)
    

    print('Exact clustering')
    start = time.clock()
    cluster_idxN = modelN.fit(XTest)
    end = time.clock()
    dtwTime = end-start
    #print('Exact clustering done')

    print('Approximate clustering')
    start = time.clock()
    cluster_idx = model.fit(XTest)
    end = time.clock()
    #print('Approximate clustering done')
    pqTime = end-start

    #model2.plot("~/hierarchy.jpg")
    #model2N.plot("~/hierarchy2.jpg")

    jaccard, ari = equaliseClusterLabelsAndCalculateScores(cluster_idxN, cluster_idx, XTest)
    return {'jaccard':jaccard, 'ari':ari, 'DTWTime': dtwTime, 'PQTime':pqTime}

def take2RandDataParts(series, seriesY, len1, len2, seed = 0 ):
    nb_series = len1+len2
    random.seed(seed)
    indiceList = [ i for i in range(nb_series)]
    random.shuffle(indiceList)
    samples = indiceList[0:len1]
    testData = indiceList[len1:len2+len1]
    return series[samples], seriesY[samples], series[testData], seriesY[testData]

def splitData(series, seriesY, ratio, seed = 0):
    nb_series = len(series)
    random.seed(seed)
    nb_samples=int(min(nb_series,ratio*nb_series))
    indiceList = [ i for i in range(nb_series)]
    random.shuffle(indiceList)
    samples = indiceList[0:nb_samples]
    testData = indiceList[nb_samples:nb_series]
    
    return series[samples], seriesY[samples], series[testData], seriesY[testData]


def keoghTest():
    from dtaidistance import dtw
    X  = np.array([[1,5,6,1,1],[1,2,7,2,1],[25,22,15,41,21]], dtype=np.double)
    X2  = np.array([[1,5,6,1,1],[25,22,15,41,21],[25,22,15,41,21],[1,2,7,2,1]], dtype=np.double)
   
    print(X)
    L,U = dtw.lb_keogh_enveloppes_fast(X2,2)
    Y = dtw.nearest_neighbour_lb_keogh_fast(X,X2,L, U, distParams={'window':2, 'psi':0})
    print(Y)



#keoghTest()
#clusterECG5000Test()
clusterStarLightTest()
#clusterEcg200Test()
#clusterGunPointTest()


