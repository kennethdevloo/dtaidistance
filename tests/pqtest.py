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

def loadDataSet(file1, file2=None, maxDataPoints = None):
    X = np.loadtxt(file1)
    lenX = len(X)
    lenX2 = 0

    if file2 is not None:
        X2 = np.loadtxt(file2)
        lenX2 = len(X2)
        X=np.concatenate((X, X2))

    Y= X[:,-1]
    X_scaled = preprocessing.scale(X[:,0:-1])

    lenTotal = len(X)
    if maxDataPoints is not None:
        lenTotal = np.min(maxDataPoints, lenTotal) 

    return X_scaled[0:lenTotal,:], Y[0:lenTotal], lenX, lenX2


def loadEcg500DataSetForClustering():
    return loadDataSet('ECG5000_TRAIN.txt', 'ECG5000_TEST.txt',1000)
    

def loadEcg200DataSetForClustering():
    return loadDataSet('ECG200_TRAIN.txt', 'ECG200_TEST.txt')


def loadGunPointDataSetForClustering():
    return loadDataSet('GunPoint_TRAIN.txt', 'GunPoint_TEST.txt')
    

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
    conDistPred=condenseDists(pred)
    rmse = 0
    me = 0
    spearManCorrelation = 0
    spearManP = 0
    pearson = 0
    triangularisationScore  = 0
    rmse = np.sqrt(np.mean((conDistPred-conDistTruth)**2))
    me = np.mean(conDistTruth-conDistPred)
    
    spearManCorrelation, spearManP = stats.spearmanr((conDistTruth), (conDistPred))
#    print('Pearson', stats.pearsonr(np.argsort(conDistTruth), np.argsort(conDistPred)))
    triangularisationScore = triangularTest(pred, triangleTries)/triangularTest(truth, triangleTries)

    print(me, rmse, spearManCorrelation, spearManP, triangularisationScore)

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



def clusterGunPointTest():
    ratio = 0.2
    seed = 0
    distParams={'window': 10, 'psi': 0}
    nwDistParams={'window': 1, 'psi': None}
    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':None, 'min_clusters':5}
    pqClusterParams = {'k':2500, 'quantizer_usage':clustering.QuantizerUsage.TOP_K_ONLY_AT_INITIALISATION}
    
    qParams=[]
    qParams.append(q.ProductQuantiserParameters(25,30,distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
    #qParams.append(q.ProductQuantiserParameters(9,4))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(2,2,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.NO_OVERLAP))

    qNWPileParams=[]
    qNWPileParams.append(q.ProductQuantiserParameters(6,4,
    quantizerType=q.QuantizerType.VQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams))

    X,Y=loadGunPointDataSetForClustering()
    XTrain, YTrain, XTest, YTest = splitData(X,Y, ratio, seed)

    distanceAndClusterTests(XTrain,XTest, qParams, distParams, generalClusterParams, pqClusterParams)

def keoghTest():
    X,_=loadEcg200DataSetForClustering()
    Xt = X[43:45, 0:4]-2.0 
    X = X[2:3, 0:4]

    L, U  = dtw.lb_keogh_enveloppes(X, 1, False)
    #print (X)
    print(L)
    print(U)
    lb = dtw.lb_keogh_distance_fast(Xt, L, U)
    print(Xt)
    print(lb)

    print(dtw.distance_matrix_fast(Xt,window=1))
    

def clusterEcg200Test():
    ratio = 0.3
    seed = 0
    kmWindowSize = 0
    distParams={'window': 5, 'psi': 0}
    nwDistParams={'window': 2, 'psi': 0}
    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':None, 'min_clusters':10}
    pqClusterParams = {'k':20,'quantizer_usage':clustering.QuantizerUsage.TOP_K}
    
    
    qParams=[]
    qParams.append(q.ProductQuantiserParameters(32,75,distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP, kmeansWindowSize=kmWindowSize))
    #qParams.append(q.ProductQuantiserParameters(9,2))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(16,40,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))

    qNWPileParams=[]
    qNWPileParams.append(q.ProductQuantiserParameters(16,20,
    quantizerType=q.QuantizerType.VQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))

    X,Y,_,_=loadEcg200DataSetForClustering()
    XTrain, YTrain, XTest, YTest = splitData(X,Y, ratio, seed)

    distanceAndClusterTests(XTrain,XTest, qParams, distParams, generalClusterParams, pqClusterParams)


def clusterECG5000Test():
    ratio = 0.15
    seed = 0
    distParams={'window': 10, 'psi': 1}
    nwDistParams={'window': 2, 'psi': 0}
    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':None, 'min_clusters':10}
    pqClusterParams = {'k':20,'quantizer_usage':clustering.QuantizerUsage.TOP_K}
    
    qParams=[]
    qParams.append(q.ProductQuantiserParameters(35,10,distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
    #qParams.append(q.ProductQuantiserParameters(9,4))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(20,40,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))

    qNWPileParams=[]
    qNWPileParams.append(q.ProductQuantiserParameters(20,20,
    quantizerType=q.QuantizerType.VQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
    
    X,Y,_,_=loadEcg500DataSetForClustering()
    print('data loaded, size', X.shape)
    XTrain, YTrain, XTest, YTest = splitData(X,Y, ratio, seed)

    distanceAndClusterTests(XTrain,XTest, qParams, distParams, generalClusterParams, pqClusterParams)
    

def trainQuantizer(XTrain, qParams):
    return q.PyProductQuantizer(XTrain, qParams)
    

def calcTrue(XTest, distParams):
    return dtaidistance.dtw.distance_matrix_fast(XTest, **distParams)

def distanceAndClusterTests(XTrain,XTest, qParams, distParams, generalClusterParams, pqClusterParams, seed = 0):
    print('Init distance test')
    pq = trainQuantizer(XTrain, qParams)
    print('Initialization done')

    distResults = distanceTest(pq,XTest, distParams, seed)
    print ('distanceTestResults',distResults)
    clustResults = clusterTest(pq,XTest, distParams, generalClusterParams, pqClusterParams)
    print ('clusterTestResults',clustResults)


def distanceTest(pq, XTest, distParams, seed = 0):
        
    print('regular')
    truth = calcTrue(XTest, distParams)
    print('regular Calc done')
    
    print('approximate')
    pred = pq.constructApproximateDTWDistanceMatrix(XTest)
    print('approximating done')
    #print(pred[0:5, 0:5])
    #print(truth[0:5, 0:5])

    me, rmse, spearManCorrelation, spearManP, triangularisationScore = scoreDistanceMatrices(truth,pred, pred.shape[0])
    
    print ('end distance test')
    return {'me':me, 'rmse':rmse, 'spearManCorrelation':spearManCorrelation, 'spearManP':spearManP, 'triangularisationScore':triangularisationScore}


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
    

    model2 = clustering.HierarchicalTree(model)
    print('Approximate clustering')
    cluster_idx = model2.fit(XTest)
    print('Approximate clustering done')
   
    model2N = clustering.HierarchicalTree(modelN)
    print('Exact clustering')
    cluster_idxN = model2N.fit(XTest)
    print('Exact clustering done')

    #model2.plot("~/hierarchy.jpg")
    #model2N.plot("~/hierarchy2.jpg")

    jaccard, ari = equaliseClusterLabelsAndCalculateScores(cluster_idxN, cluster_idx, XTest)
    return {'jaccard':jaccard, 'ari':ari }

def splitData(series, seriesY, ratio, seed = 0):
    import random
    nb_series = len(series)
    random.seed(seed)
    nb_samples=int(min(nb_series,ratio*nb_series))
    indiceList = [ i for i in range(nb_series)]
    random.shuffle(indiceList)
    samples = indiceList[0:nb_samples]
    testData = indiceList[nb_samples:nb_series]
    return series[samples], seriesY[samples], series[testData], seriesY[testData]


#keoghTest()
clusterECG5000Test()
#clusterEcg200Test()
#clusterGunPointTest()

