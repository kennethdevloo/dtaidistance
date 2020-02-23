from sklearn.cluster import KMeans
from enum import Enum
import numpy as np
import math
from .dtw import distance_matrix, distance
from .alignment import needleman_wunsch,  SparseSubstitutionMatrix


from scipy.cluster.hierarchy import dendrogram,linkage
from matplotlib import pyplot as plt

from tslearn.clustering import TimeSeriesKMeans

class SubsetSelectionType(Enum):
    NO_OVERLAP = 1
    DOUBLE_OVERLAP=2

class Metric(Enum):
    DTW=0
    ED=1   #ONLY CLUSTERING IMPLEMENTED ATM, not distance, perfect for upperbounds
    

class QuantizerType(Enum):
    PQDICT=0 #With recursion to smaller chunks
    PQNeedlemanWunsch=1
    VQNeedlemanWunsch=2 #NOT YET IMPLEMENTED, should be with VQ+index over calculated centres

class PQDictionaryCalculator(Enum): #NOT IMPLEMENTED
    KMEANS=0 #Uses tslearm, add barycentrix mean + kmeans to dtaidistance
    TADPOLE=1 #Just use R bridge?

class clusterType(Enum):
    SINGLELINKAGE=0
    COMPLETELINKAGE=1

class ProductQuantiserParameters():
    def __init__(self, subsetSize, dictionarySize, distParams={}, 
    withIndex=False, residuals=False, distanceMetric = Metric.DTW, 
    quantizerType=QuantizerType.PQDICT, nwDistParams = {},
    subsetType = SubsetSelectionType.NO_OVERLAP):
        self.subsetSize=subsetSize
        self.dictionarySize=dictionarySize
        self.distParams= distParams
        self.withIndex=withIndex
        self.residuals=residuals
        self.distanceMetric=distanceMetric
        self.quantizerType=quantizerType
        self.nwDistParams=nwDistParams
        self.subsetType=subsetType



class PQDictionary():
    def __init__(self, data, pqParams, depth, centers = None):
        self.recursiveLayer = False
        if  depth+1 < len(pqParams):
            self.recursiveLayer=True
        self.codeBook = centers
        self.index=None
        self.centeredData=None
        self.data=data
        self.distanceDTWMatrix=None
        self.productQuantisers = []
        self.params=pqParams[depth]

      
        if self.params.distanceMetric is Metric.ED:
            print('ED is not implemented')
            pass


        if self.params.distanceMetric is Metric.DTW:
            if self.params.residuals:
                #makes no sense to subtract mean from time series
                pass        
            self.kmeans = TimeSeriesKMeans(n_clusters=self.params.dictionarySize,random_state=0,metric="dtw",).fit(data)
            self.codeBook = self.kmeans.cluster_centers_
            if self.params.withIndex or self.recursiveLayer:
                predictions = self.kmeans.predict(data)
                self.index=[]
                for i in range(0,self.params.dictionarySize):
                    self.index.append(np.where(predictions == i))
            if self.recursiveLayer:
                for i in range(0,self.kmeans.cluster_centers_.shape[0]):
                    self.productQuantisers.append(ProductQuantizer(data[self.index[i],:][0], pqParams, depth+1))
            if self.params.quantizerType ==  QuantizerType.PQDICT:
                self.distanceDTWMatrix = distance_matrix(self.codeBook,**(self.params.distParams))
            # print(self.distanceDTWMatrix)

    def replaceCodeBook(self, codeBook, data=None):
        self.kmeans.cluster_centers_= codeBook
        self.codeBook=codeBook

        if data is not None:
            if params.residuals:
                self.avg = np.mean(data)
                data = data - self.avg


                if self.params.withIndex or self.recursiveLayer:
                    predictions = self.kmeans.predict(data)
                    self.index=[]
                    for i in range(0,self.params.dictionarySize):
                        self.index.append(np.where(predictions == i))
                    
                if self.recursiveLayer:
                    for i in range(0,self.kmeans.cluster_centers_.shape[0]):
                        self.productQuantisers.append(ProductQuantizer(data[self.index[i],:][0], pqParams, depth+1))
                if params.quantizerType ==  QuantizerType.PQDICT:
                    self.distanceDTWMatrix = distance_matrix(self.codeBook,window= params.windowSize)
                
            
            



    def retrieveCodes(self, data):
        return self.kmeans.predict(data)

    
    def retrieveApprDTWDistance(self, code1, code2):
        if code1 == code2:
            return 0
        else :
            return self.distanceDTWMatrix[int(min(code1, code2)), int(max(code1, code2))]

    def retrieveRecApprDTWDistance(self, code, data1, data2):
        if self.recursiveLayer is False:
            return 0
        return self.productQuantisers[int(code)].calculateApprDTWDistanceForData(data1, data2)


         # averaging time series makes no sense? reimplement kmeans with averages according to ?
         # So average according to presentation! (Adapt a kmeans imlementation)
         # Use a lot more clusters then there are classes-> Support for moving of features betwen subclasses



class ProductQuantizer():
    def __init__(self,data, pqParams, depth=0):
        self.nrDictionaries = math.ceil(data.shape[1]/pqParams[depth].subsetSize) 
        self.dictionaries = []
        self.subsetSize = pqParams[depth].subsetSize
        self.params=pqParams[depth]
        if self.params.subsetType is SubsetSelectionType.NO_OVERLAP:
            for i in range(0,self.nrDictionaries):
                self.dictionaries.append(PQDictionary(data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])],pqParams, depth)) 
        else:
            self.dictionaries.append(PQDictionary(data[:,0*self.subsetSize:min((0+1)*self.subsetSize,data.shape[1])],pqParams, depth))
            self.halfSubsetSize=int(self.subsetSize/2)
            for i in range(1,self.nrDictionaries):
                self.dictionaries.append(PQDictionary(data[:,(i*self.subsetSize-self.halfSubsetSize):min((i+1)*self.subsetSize-self.halfSubsetSize,data.shape[1])],pqParams, depth))
                self.dictionaries.append(PQDictionary(data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])],pqParams, depth)) 

        if pqParams[depth].quantizerType==QuantizerType.PQNeedlemanWunsch or pqParams[depth].quantizerType==QuantizerType.VQNeedlemanWunsch:
            codeBook=self.dictionaries[0].codeBook
            for i in range(1, len(self.dictionaries)):
                codeBook = np.concatenate((codeBook,self.dictionaries[i].codeBook))
            self.substitution = SparseSubstitutionMatrix(codeBook,distParams=pqParams[depth].distParams)

        if pqParams[depth].quantizerType==QuantizerType.VQNeedlemanWunsch:
            self.dictionaries[0].replaceCodeBook(codeBook)
            self.dictionaries=self.dictionaries[0]


    def retrieveCodedData(self, data):
        codedData=None
        if self.params.subsetType is SubsetSelectionType.NO_OVERLAP:
            codedData = np.zeros((data.shape[0],self.nrDictionaries))
        if self.params.subsetType is SubsetSelectionType.DOUBLE_OVERLAP:
            codedData = np.zeros((data.shape[0],self.nrDictionaries*2-1))
        
        if self.params.quantizerType is not QuantizerType.VQNeedlemanWunsch:
            if self.params.subsetType is SubsetSelectionType.NO_OVERLAP:
                for i in range(0,self.nrDictionaries):
                    codedData[:, i] = self.dictionaries[i].retrieveCodes(
                        data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])])
                    if self.params.quantizerType is  QuantizerType.PQNeedlemanWunsch:
                        codedData[:, i] = codedData[:, i] + i* self.params.dictionarySize
            else:
                codedData[:, 0] = self.dictionaries[0].retrieveCodes(
                    data[:,0*self.subsetSize:min((0+1)*self.subsetSize,data.shape[1])])
                for i in range(1,self.nrDictionaries):
                    codedData[:, 2*i-1] = self.dictionaries[2*i-1].retrieveCodes(
                        data[:,i*self.subsetSize-self.halfSubsetSize:min((i+1)*self.subsetSize-self.halfSubsetSize,data.shape[1])])
                    codedData[:, 2*i] = self.dictionaries[2*i].retrieveCodes(
                        data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])])
                    if self.params.quantizerType is  QuantizerType.PQNeedlemanWunsch:
                        codedData[:, 2*i-1] + (2*i-1)* self.params.dictionarySize
                        codedData[:, 2*i] + (2*i)* self.params.dictionarySize
        else:
            if self.params.subsetType is SubsetSelectionType.NO_OVERLAP:
                for i in range(0,self.nrDictionaries):
                    codedData[:, i] = self.dictionaries.retrieveCodes(
                        data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])])
                   
            else:
                codedData[:, 0] = self.dictionaries.retrieveCodes(
                    data[:,0*self.subsetSize:min((0+1)*self.subsetSize,data.shape[1])])
                for i in range(1,self.nrDictionaries):
                    codedData[:, 2*i-1] = self.dictionaries.retrieveCodes(
                        data[:,i*self.subsetSize-self.halfSubsetSize:min((i+1)*self.subsetSize-self.halfSubsetSize,data.shape[1])])
                    codedData[:, 2*i] = self.dictionaries.retrieveCodes(
                        data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])])

        return codedData

    def calculateApprDTWDistanceForCodes(self, code1, code2, data1, data2):
        if self.params.quantizerType==QuantizerType.PQNeedlemanWunsch or self.params.quantizerType==QuantizerType.VQNeedlemanWunsch:
            distance,m =needleman_wunsch(code1, code2,substitutionMatrix=self.substitution, **(self.params.nwDistParams))
            #print(np.sqrt(-m))
            if self.params.subsetType is SubsetSelectionType.NO_OVERLAP:
                return np.sqrt(distance)
            else:
                return np.sqrt(distance)*(self.nrDictionaries-1)/self.nrDictionaries

        distance = 0
        for i in range(0,code1.shape[0]):
            if self.dictionaries[i].recursiveLayer is True and code1[i] == code2[i]:
                if self.params.subsetType is SubsetSelectionType.NO_OVERLAP:
                    distance = distance + self.dictionaries[i].retrieveRecApprDTWDistance(code1[i],
                        data1[i*self.subsetSize:min((i+1)*self.subsetSize,data1.shape[0])],
                        data2[i*self.subsetSize:min((i+1)*self.subsetSize,data2.shape[0])])**2
                else:
                    if i % 2 == 0:
                        distance = distance + self.dictionaries[i].retrieveRecApprDTWDistance(code1[i],
                            data1[int(i/2*self.subsetSize):int(min((i/2+1)*self.subsetSize,data1.shape[0]))],
                            data2[int(i/2*self.subsetSize):int(min((i/2+1)*self.subsetSize,data2.shape[0]))])**2
                    else:
                        distance = distance + self.dictionaries[i].retrieveRecApprDTWDistance(code1[i],
                            data1[int((i+1)/2*self.subsetSize-self.halfSubsetSize):int(min(((i+1)/2+1)*self.subsetSize-self.halfSubsetSize,data1.shape[0]))],
                            data2[int((i+1)/2*self.subsetSize-self.halfSubsetSize):int(min(((i+1)/2+1)*self.subsetSize-self.halfSubsetSize,data2.shape[0]))])**2
            else:
                distance = distance + self.dictionaries[i].retrieveApprDTWDistance(code1[i], code2[i])**2
        
        if self.params.subsetType is SubsetSelectionType.NO_OVERLAP:
            return np.sqrt(distance)
        else:
            return np.sqrt(distance)*(self.nrDictionaries-1)/self.nrDictionaries

    def constructApproximateDTWDistanceMatrix(self, data):
        codedData = self.retrieveCodedData(data)
        approximateMatrix = np.zeros((data.shape[0], data.shape[0]))
        approximateMatrix[:]=np.inf
        for xi in range(0, data.shape[0]):
            for xj in range(xi+1, data.shape[0]):
                approximateMatrix[xi, xj]=self.calculateApprDTWDistanceForCodes(codedData[xi,:], codedData[xj,:], data[xi,:], data[xj,:])
        return approximateMatrix

    def calculateApprDTWDistanceForData(self, data1, data2):
        codedData = self.retrieveCodedData(np.array([data1, data2]))
        return self.calculateApprDTWDistanceForCodes(codedData[0,:], codedData[1,:], data1, data2)