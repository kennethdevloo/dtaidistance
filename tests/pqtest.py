from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from enum import Enum
import numpy as np
import random
import dtaidistance 
from dtaidistance import alignment, clustering
import math
import scipy.sparse as sparse
import dtaidistance.quantizer as q
import dtaidistance.quantizer_c as qc
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

def doLinkage(dists, method = 'complete'):
    dists_cond = condenseDists(dists)
    return linkage(dists_cond, method=method, metric='euclidean')

def createLinkageClustering():
    #   A = np.array([1, 7, 9, 2, 0.1, 17, 17, 1.5])
    #  k = 3
    #idx = np.argpartition(A, k)
    pass



def pqTest(datatrain, datatest):
    if False:
        model = clustering.Hierarchical(dtaidistance.dtw.distance_matrix_fast, {},dists_merger=clustering.completeLinkageUpdater)
        model2 = clustering.HierarchicalTree(model)
        cluster_idx = model2.fit(datatest[0:6])

        model2.plot("hierarchy.jpg")

        model3 = clustering.LinkageTree(dtaidistance.dtw.distance_matrix_fast, {})
        cluster_idx = model3.fit(datatest[0:6])
        model3.plot("hierarchy2.jpg")

        return

    qParams=[]
    qParams.append(q.ProductQuantiserParameters(33,10,None))
    qParams.append(q.ProductQuantiserParameters(33,10,None))

    qNonRecParams=[]
    qNonRecParams.append(q.ProductQuantiserParameters(33,10,None))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(6,10,None,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwWindowSize=2, psi=2))

    qNWPileParams=[]
    qNWPileParams.append(q.ProductQuantiserParameters(6,10,None,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwWindowSize=2, psi=2))
    
    pq = q.ProductQuantizer(datatrain, qParams)
    pqnonrec = q.ProductQuantizer(datatrain, qNonRecParams)
    pqnw = q.ProductQuantizer(datatrain, qNWParams)
    pqvqnw = q.ProductQuantizer(datatrain, qNWPileParams)
    #return  
#  codedTestSet = retrieveCodedData(datatest)
    if False:
        dists = pq.constructApproximateDTWDistanceMatrix(datatest)
        links=doLinkage(dists)
        #print(links)
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(links)
        plt.show()
        input("Press Enter to continue...")

    vqDistMatrix = pq.constructApproximateDTWDistanceMatrix(datatest[0:6,:])
    vqnorecDistMatrix = pqnonrec.constructApproximateDTWDistanceMatrix(datatest[0:6,:])
    vqnwDistMatrix = pqnw.constructApproximateDTWDistanceMatrix(datatest[0:6,:])
    vqnwPileDistMatrix = pqvqnw.constructApproximateDTWDistanceMatrix(datatest[0:6,:]) 
    dtaiDistMatrix = dtaidistance.dtw.distance_matrix_fast(datatest[0:6,:])
    
    print (vqDistMatrix)
    print (vqnorecDistMatrix)
    print (vqnwDistMatrix)
    print(vqnwPileDistMatrix)
    print(dtaiDistMatrix)
    
    #print(pqnw.substitution.distances)

def loadEcg200DataSets():
    X_train = np.loadtxt('ECG200_TRAIN.txt')
    X_train_scaled = preprocessing.scale(X_train[:,0:-1])
    Y_train = X_train[:,-1]

    X_test = np.loadtxt('ECG200_TEST.txt')
    X_test_scaled = preprocessing.scale(X_test[:,0:-1])
    Y_test = X_test[:,-1]

    return X_train_scaled, Y_train, X_test_scaled, Y_train

def loadEcg500DataSetForClustering():
    X_train = np.loadtxt('ECG5000_TRAIN.txt')

    X_test = np.loadtxt('ECG5000_TEST.txt')
    X=np.concatenate((X_train, X_test))

    Y= X[:,-1]
    X_scaled = preprocessing.scale(X[:,0:-1])
    return X_scaled[0:1000,:], Y[0:1000]
    

def loadEcg200DataSetForClustering():
    X_train = np.loadtxt('ECG200_TRAIN.txt')
    Y_train = X_train[:,-1]

    X_test = np.loadtxt('ECG200_TEST.txt')
    X=np.concatenate((X_train, X_test))

    Y= X[:,-1]
    X_scaled = preprocessing.scale(X[:,0:-1])
    return X_scaled, Y

def loadGunPointDataSetForClustering():
    X_train = np.loadtxt('GunPoint_TRAIN.txt')
    Y_train = X_train[:,-1]

    X_test = np.loadtxt('GunPoint_TEST.txt')
    X=np.concatenate((X_train, X_test))

    Y= X[:,-1]
    X_scaled = preprocessing.scale(X[:,0:-1])

    print (X_scaled.shape)

    return X_scaled, Y

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
    print('Pearson', stats.pearsonr(np.argsort(conDistTruth), np.argsort(conDistPred)))
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
    clust = clusterTest(XTrain,XTest, qParams, distParams, generalClusterParams, pqClusterParams)
    #qnwdist = distanceTest(X, Y, qNWParams, distParams)
    qdist = distanceTest(XTrain, XTest, qParams, distParams)
    #print(qnwdist)
    print(qdist)
    print (clust)

def clusterEco200Test():
    ratio = 0.3
    seed = 0
    distParams={'window': 5, 'psi': 1}
    nwDistParams={'window': 2, 'psi': 0}
    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':None, 'min_clusters':10}
    pqClusterParams = {'k':20,'quantizer_usage':clustering.QuantizerUsage.TOP_K}
    
    qParams=[]
    qParams.append(q.ProductQuantiserParameters(32,75,distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
    #qParams.append(q.ProductQuantiserParameters(9,4))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(16,40,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))

    qNWPileParams=[]
    qNWPileParams.append(q.ProductQuantiserParameters(16,20,
    quantizerType=q.QuantizerType.VQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
    
    X,Y=loadEcg200DataSetForClustering()
    XTrain, YTrain, XTest, YTest = splitData(X,Y, ratio, seed)
    #clust = clusterTest(XTrain,XTest, qNWPileParams, distParams, generalClusterParams, pqClusterParams)
    #qnwdist = distanceTest(X, Y, qNWParams, distParams)
    qdist = distanceTest(XTrain,XTest, qNWPileParams, distParams)
    #print(qnwdist)
    print(qdist)
    #print (clust)


def clusterECG5000Test():
    ratio = 0.15
    seed = 0
    distParams={'window': 10, 'psi': 1}
    nwDistParams={'window': 2, 'psi': 0}
    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':None, 'min_clusters':10}
    pqClusterParams = {'k':20,'quantizer_usage':clustering.QuantizerUsage.TOP_K}
    
    qParams=[]
    qParams.append(q.ProductQuantiserParameters(35,75,distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
    #qParams.append(q.ProductQuantiserParameters(9,4))

    qNWParams=[]
    qNWParams.append(q.ProductQuantiserParameters(20,40,
    quantizerType=q.QuantizerType.PQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))

    qNWPileParams=[]
    qNWPileParams.append(q.ProductQuantiserParameters(20,20,
    quantizerType=q.QuantizerType.VQNeedlemanWunsch,nwDistParams=nwDistParams, distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
    
    X,Y=loadEcg500DataSetForClustering()
    print('data loaded, size', X.shape)
    XTrain, YTrain, XTest, YTest = splitData(X,Y, ratio, seed)
    print('data split')
    #clust = clusterTest(XTrain,XTest, qParams, distParams, generalClusterParams, pqClusterParams)
    #qnwdist = distanceTest(X, Y, qNWParams, distParams)
    qdist = distanceTest(XTrain,XTest, qParams, distParams)
    print(qdist)
    #print(qdist)
    print (clust)


def clusterECG5000FastTest():
    ratio = 0.01
    seed = 0
    
    #distParams=qc.DistanceParams(windowSize=10, psi = 1)
    distParams={'window': 10, 'psi': 1}

    #generalClusterParams = {'dists_merger':clustering.singleLinkageUpdater, 'min_clusters':10}
    generalClusterParams = {'dists_merger':None, 'min_clusters':10}
    pqClusterParams = {'k':20,'quantizer_usage':clustering.QuantizerUsage.TOP_K}
    
    qParams=[]
    qParams.append(qc.ProductQuantiserParameters(35,75,distParams=distParams, subsetType=q.SubsetSelectionType.DOUBLE_OVERLAP))
    #qParams.append(q.ProductQuantiserParameters(9,4))

    X,Y=loadEcg500DataSetForClustering()
    print('data loaded, size', X.shape)
    XTrain, YTrain, XTest, YTest = splitData(X,Y, ratio, seed)
    print('data split')
    #clust = clusterTest(XTrain,XTest, qParams, distParams, generalClusterParams, pqClusterParams)
    #qnwdist = distanceTest(X, Y, qNWParams, distParams)
    qdist = fastDistanceTest(XTrain,XTest, qParams, distParams)
    print(qdist)
    #print(qdist)
    #print (clust)



def fastDistanceTest(XTrain,XTest, qParams, distParams, ratio = 0.5, seed = 0):
    print('Init distance test')
    
    pq = qc.PyProductQuantizer(XTrain, qParams)
    print('regular')
    truth = dtaidistance.dtw.distance_matrix_fast(XTest, **distParams)
    print('regular Calc done')
    print('prediction')
    pred = pq.constructApproximateDTWDistanceMatrix(XTest)
    print('approximating done')

    me, rmse, spearManCorrelation, spearManP, triangularisationScore = scoreDistanceMatrices(truth,pred, pred.shape[0])
    print ('end distance test')
    return {'me':me, 'rmse':rmse, 'spearManCorrelation':spearManCorrelation, 'spearManP':spearManP, 'triangularisationScore':triangularisationScore}

def distanceTest(XTrain,XTest, qParams, distParams, ratio = 0.5, seed = 0):
    print('Init distance test')
    
    pq = q.ProductQuantizer(XTrain, qParams)
    print('regular')
    truth = dtaidistance.dtw.distance_matrix_fast(XTest, **distParams)
    print('regular Calc done')
    print('prediction')
    pred = pq.constructApproximateDTWDistanceMatrix(XTest)
    print('approximating done')

    me, rmse, spearManCorrelation, spearManP, triangularisationScore = scoreDistanceMatrices(truth,pred, pred.shape[0])
    print ('end distance test')
    return {'me':me, 'rmse':rmse, 'spearManCorrelation':spearManCorrelation, 'spearManP':spearManP, 'triangularisationScore':triangularisationScore}


def clusterTest(XTrain,XTest, qParams, distParams, generalClusterParams, pqClusterParams ):
    model = clustering.HierarchicalWithQuantizer(
        dtaidistance.dtw.distance_fast, distParams,
        **generalClusterParams,
        **pqClusterParams)

    modelN = clustering.Hierarchical(
        dtaidistance.dtw.distance_matrix_fast, distParams,
        **generalClusterParams)

    model.trainQuantizer(XTrain,qParams)
    

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

#import cProfile, pstats, io
#pr = cProfile.Profile()
#pr.enable()


clusterECG5000FastTest()


#pr.disable()
#s = io.StringIO()
#sortby = 'cumulative'
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
#print(s.getvalue())


  
#clusterEco200Test()
#clusterGunPointTest()

