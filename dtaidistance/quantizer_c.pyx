# cython: profile=True
from enum import Enum
import numpy as np
cimport numpy as np
import math
cimport cython
from .dtw import distance_matrix_fast as distance_matrix, nearest_neighbour_lb_keogh_fast, lb_keogh_enveloppes_fast #, distance_fast
from .alignment import needleman_wunsch,  SparseSubstitutionMatrix

from tslearn.clustering import TimeSeriesKMeans




cpdef enum SubsetSelectionType:
    NO_OVERLAP = 1
    DOUBLE_OVERLAP=2

cpdef enum Metric:
    DTW=0
    ED=1   #ONLY CLUSTERING IMPLEMENTED ATM, not distance, perfect for upperbounds
    

cpdef enum QuantizerType:
    PQDICT=0 
    PQNeedlemanWunsch=1
    VQNeedlemanWunsch=2 

cpdef enum PQDictionaryCalculator: #NOT IMPLEMENTED
    KMEANS=0 #Uses tslearm, add barycentrix mean + kmeans to dtaidistance
    TADPOLE=1 #Just use R bridge? Not supported right now


class DistanceParams():
    def __init__(self, windowSize = 10, psi = 0):
        self.windowSize = windowSize
        self.psi = psi


class ProductQuantiserParameters():
    def __init__(self, subsetSize = 35, dictionarySize = 30, distParams=DistanceParams(), 
    withIndex=False, residuals=False, 
    quantizerType=QuantizerType.PQDICT, nwDistParams = {},
    subsetType = SubsetSelectionType.NO_OVERLAP, barycenterMaxIter=25,
    max_iters =20, kmeansWindowSize = 0, useLbKeoghToFit = True):
        self.subsetSize=subsetSize
        self.dictionarySize=dictionarySize
        self.distParams= distParams
        self.withIndex=withIndex
        self.residuals=residuals
        self.quantizerType=quantizerType
        self.subsetType=subsetType
        self.barycenterMaxIter =barycenterMaxIter
        self.max_iters=max_iters
        self.useLbKeoghToFit = useLbKeoghToFit
        if kmeansWindowSize == 0:
            self.metric_params = None
        else:
            self.metric_params = {'global_constraint':"sakoe_chiba", 'sakoe_chiba_radius':kmeansWindowSize}
        self.nwDistParams=nwDistParams

cdef class PQDictionary():
    cdef public bint recursiveLayer
    cdef int depth

    #what I actually wanted cdef np.ndarray[np.int32_t, ndim=2] index
    cdef object index
    cdef object params
    cdef object kmeans
    cdef object L, U  #enveloppes
    #cdef np.ndarray[np.double_t, ndim=2] distanceDTWMatrix
    cdef public object distanceDTWMatrix
    cdef list productQuantisers
    #cdef ProductQuantiserParameters params
    
    def __init__(self, data, pqParams, depth, centers = None):
        self.recursiveLayer = False
        if  depth+1 < len(pqParams):
            self.recursiveLayer=True 
            self.productQuantisers = []
        self.params=pqParams[depth]

     
        self.kmeans = TimeSeriesKMeans(metric_params = self.params.metric_params, n_clusters=self.params.dictionarySize,random_state=0,metric="dtw", max_iter_barycenter=self.params.barycenterMaxIter, max_iter=self.params.max_iters).fit(data)
        self.L, self.U = lb_keogh_enveloppes_fast(np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])), self.params.distParams['window'])
        if self.params.withIndex or self.recursiveLayer:
            predictions = self.kmeans.predict(data)
            self.index=[]
            for i in range(0,self.params.dictionarySize):
                self.index.append(np.where(predictions == i))
        if self.recursiveLayer:
            for i in range(0,self.kmeans.cluster_centers_.shape[0]):
                if len(data[self.index[i],:][0]) == 0:
                    pass
                else:
                    self.productQuantisers.append(ProductQuantizer(data[self.index[i],:][0], pqParams, depth+1))
        if self.params.quantizerType ==  QuantizerType.PQDICT:
            self.distanceDTWMatrix = distance_matrix(np.reshape(self.kmeans.cluster_centers_ ,(self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])),**self.params.distParams )
            for i in range(0,self.distanceDTWMatrix.shape[0]):
                self.distanceDTWMatrix[i,i]=0
                for j in range(i+1,self.distanceDTWMatrix.shape[0]):
                    self.distanceDTWMatrix[j,i] = self.distanceDTWMatrix[i,j]

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef replaceCodeBook(self, np.ndarray[np.double_t, ndim=2] codeBook):
        self.kmeans.cluster_centers_= codeBook
        self.L, self.U = lb_keogh_enveloppes_fast(np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])), self.params.distParams['window'])

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef retrieveCodes(self, np.ndarray[np.double_t, ndim=2] data):
        cdef np.ndarray[np.int_t, ndim=1] distMatrix2
        cdef np.ndarray[np.double_t, ndim=2] distMatrix
        if not self.params.useLbKeoghToFit:
            distMatrix = distance_matrix(np.concatenate((data, np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])))),block=((0,data.shape[0]), (data.shape[0], data.shape[0]+self.kmeans.cluster_centers_.shape[0])),**self.params.distParams )
            distMatrix2=np.argmin(distMatrix, axis=1)[0:len(data)]
            return distMatrix2-len(data)
        #return self.kmeans.predict(data)
        elif self.params.useLbKeoghToFit:
            return nearest_neighbour_lb_keogh_fast(data, self.L, self.U,np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])), self.params.distParams)
    
    cpdef getCodeBook(self):
        return np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1]))


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef double retrieveApprDTWDistance(self, int code1, int code2):
        return self.distanceDTWMatrix[code1, code2]

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef double retrieveRecApprDTWDistance(self, int code, np.ndarray[np.double_t, ndim=1] data1, np.ndarray[np.double_t, ndim=1] data2):
        if self.recursiveLayer is False:
            return 0
        return self.productQuantisers[code].calculateApprDTWDistanceForData(data1, data2)



cdef class ProductQuantizer():
    cdef int nrDictionaries
    cdef int subsetSize
    cdef int halfSubsetSize
    cdef list dictionaries
    cdef object params
    cdef object substitution
    cdef double overlapCorrector
    #ProductQuantiserParameters params

    def __init__(self,data, pqParams, depth=0):
        cdef int i 
        cdef np.ndarray[dtype=np.double_t, ndim=2] codeBook, L, U
        
        self.nrDictionaries = math.ceil(data.shape[1]/pqParams[depth].subsetSize) 
        self.subsetSize = pqParams[depth].subsetSize
        self.halfSubsetSize=self.subsetSize/2
        self.params=pqParams[depth]
        self.dictionaries = []
        if self.params.subsetType is SubsetSelectionType.NO_OVERLAP:
            for i in range(0,self.nrDictionaries):
                self.dictionaries.append(PQDictionary(data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])],pqParams, depth)) 
        else:
            self.dictionaries.append(PQDictionary(data[:,0*self.subsetSize:min((0+1)*self.subsetSize,data.shape[1])],pqParams, depth))
            
            for i in range(1,self.nrDictionaries):
               # print (self.nrDictionaries, self.subsetSize, self.halfSubsetSize, data.shape)
                self.dictionaries.append(PQDictionary(data[:,(i*self.subsetSize-self.halfSubsetSize):min((i+1)*self.subsetSize-self.halfSubsetSize,data.shape[1])],pqParams, depth))
                self.dictionaries.append(PQDictionary(data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])],pqParams, depth)) 
              #  print (self.dictionaries[i].distanceDTWMatrix)

        if self.params.quantizerType==QuantizerType.PQNeedlemanWunsch or self.params.quantizerType==QuantizerType.VQNeedlemanWunsch:
            codeBook=self.dictionaries[0].getCodeBook()
            for i in range(1, len(self.dictionaries)):
                codeBook = np.concatenate((codeBook,self.dictionaries[i].getCodeBook()))
            self.substitution = SparseSubstitutionMatrix(codeBook,distParams=pqParams[depth].distParams)

        if self.params.quantizerType==QuantizerType.VQNeedlemanWunsch:
            for i in range(len(self.dictionaries)):
                self.dictionaries[i].replaceCodeBook(codeBook)
            
        
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef np.ndarray[np.int_t, ndim = 2] retrieveCodedData(self, np.ndarray[np.double_t,ndim=2] data):
        cdef np.ndarray[np.int_t, ndim=2] codedData
        cdef int i, dictionaryCount
        dictionaryCount=0
        if self.params.subsetType == SubsetSelectionType.NO_OVERLAP:
            codedData = np.zeros((data.shape[2],self.nrDictionaries), dtype=np.int)
        if self.params.subsetType == SubsetSelectionType.DOUBLE_OVERLAP:
            #print('priep') 
            codedData = np.zeros((data.shape[0],self.nrDictionaries*2-1), dtype=np.int)
        
        if self.params.subsetType is SubsetSelectionType.NO_OVERLAP:    
            for i in range(0,self.nrDictionaries):
                codedData[:, i] = self.dictionaries[i].retrieveCodes(
                    data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])])
            if self.params.quantizerType is  QuantizerType.PQNeedlemanWunsch:
                dictionaryCount = dictionaryCount + self.dictionaries[i].kmeans.cluster_centers_.shape[0]
                codedData[:, i] = codedData[:, i] + dictionaryCount
        else:
            test = self.dictionaries[0].retrieveCodes(
                data[:,0*self.subsetSize:min((0+1)*self.subsetSize,data.shape[1])])
            codedData[:, 0] = self.dictionaries[0].retrieveCodes(
                data[:,0*self.subsetSize:min((0+1)*self.subsetSize,data.shape[1])])
            for i in range(1,self.nrDictionaries):
                codedData[:, 2*i-1] = self.dictionaries[2*i-1].retrieveCodes(
                    data[:,i*self.subsetSize-self.halfSubsetSize:min((i+1)*self.subsetSize-self.halfSubsetSize,data.shape[1])])
                codedData[:, 2*i] = self.dictionaries[2*i].retrieveCodes(
                    data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])])
                if self.params.quantizerType is  QuantizerType.PQNeedlemanWunsch:
                    dictionaryCount = dictionaryCount + self.dictionaries[2*i-1].kmeans.cluster_centers_.shape[0]
                    codedData[:, 2*i-1] + dictionaryCount
                    dictionaryCount = dictionaryCount + self.dictionaries[2*i].kmeans.cluster_centers_.shape[0]
                    codedData[:, 2*i] + dictionaryCount
        
        return codedData
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef calculateApprDTWDistanceForCodes(self, np.ndarray[np.int_t, ndim=1] code1, np.ndarray[np.int_t, ndim=1] code2, np.ndarray[np.double_t, ndim=1] data1, np.ndarray[np.double_t, ndim=1] data2):
        cdef double distance = 0.0
        cdef int i
        cdef object m

        if self.params.quantizerType==QuantizerType.PQNeedlemanWunsch or self.params.quantizerType==QuantizerType.VQNeedlemanWunsch:
            distance,m =needleman_wunsch(code1, code2,substitutionMatrix=self.substitution, **(self.params.nwDistParams))
            if self.params.subsetType is SubsetSelectionType.NO_OVERLAP:
                return np.sqrt(distance)
            else:
                return np.sqrt(distance)*(self.nrDictionaries-1)/self.nrDictionaries
                
        for i in range(0,code1.shape[0]):
           # print (distance)
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
                #print(self.dictionaries[i].distanceDTWMatrix[code1[i], code2[i]], 'distadd', i)
                distance = distance + self.dictionaries[i].distanceDTWMatrix[code1[i], code2[i]]**2#retrieveApprDTWDistance(code1[i], code2[i])**2
        
        #print ('dist',distance, self.overlapCorrector)
        if self.params.subsetType is SubsetSelectionType.NO_OVERLAP:
            return np.sqrt(distance)
        else:
            return np.sqrt(distance*self.overlapCorrector)
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef public constructApproximateDTWDistanceMatrix(self,data):
        return self.constructApproximateDTWDistanceMatrixExecute(data)

    cdef constructApproximateDTWDistanceMatrixExecute(self, np.ndarray[np.double_t, ndim = 2] data):
        self.overlapCorrector = <double>(self.nrDictionaries-1)/<double>(2*self.nrDictionaries-1)
        cdef np.ndarray[np.int_t, ndim=2] codedData = self.retrieveCodedData(data)
        cdef np.ndarray[np.double_t, ndim=2] approximateMatrix = np.zeros((data.shape[0], data.shape[0]))
        approximateMatrix[:]=np.inf
        #print (data.shape[0])
        for xi in range(0, data.shape[0]):
            for xj in range(xi+1, data.shape[0]):
                approximateMatrix[xi, xj]=self.calculateApprDTWDistanceForCodes(codedData[xi,:], codedData[xj,:], data[xi,:], data[xj,:])
                
        return approximateMatrix

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef double calculateApprDTWDistanceForData(self, np.ndarray[np.double_t, ndim=1] data1,np.ndarray[np.double_t, ndim=1] data2):
        cdef np.ndarray[np.int_t, ndim=2] codedData = self.retrieveCodedData(np.array([data1, data2]))
        return self.calculateApprDTWDistanceForCodes(codedData[0,:], codedData[1,:], data1, data2)

class PyProductQuantizer():
    def __init__(self, data, pqParams):
        self.pq = ProductQuantizer(data, pqParams)

    def constructApproximateDTWDistanceMatrix(self, data):
        v=self.pq.constructApproximateDTWDistanceMatrix(data)
       # print(v[1:6,1:6])
        return v

