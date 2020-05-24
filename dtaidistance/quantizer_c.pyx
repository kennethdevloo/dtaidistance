from enum import Enum
import numpy as np
import pickle
cimport numpy as np
import math
cimport cython
from cython.parallel import prange
from .dtw import lb_keogh_distance_fast, distance_matrix_fast as distance_matrix, nearest_neighbour_lb_keogh_fast,  lb_keogh_enveloppes_fast, distance_fast
from .alignment import needleman_wunsch,  SparseSubstitutionMatrix

from tslearn.clustering import TimeSeriesKMeans

from tslearn.utils import to_time_series_dataset




cpdef enum SubsetSelectionType:
    NO_OVERLAP = 1
    DOUBLE_OVERLAP=2

cpdef enum DISTANCECALCULATION:
    SYMMETRIC=0  #DEFAULT CASE. ONLY USE PRECOMPUTED DISTANCES
    ASYMMETRIC=1 #Only supported for PQDICT
    #IN ASYM CASE: Create INV INDEX ->
    EXACT_WHEM_0DIST=2 #ALTERNATIVE TO RECURSION  
    ASYMMETRIC_KEOGH_WHEM_0DIST=3 
    RECURSION=4

cpdef enum QuantizerType:
    PQDICT=0 
    PQNeedlemanWunsch=1
    VQNeedlemanWunsch=2 



class ProductQuantiserParameters():
    def __init__(self, subsetSize = 35, dictionarySize = 30, 
    distParams={'window':5, 'psi':0},quantizationDistParams={'window':5, 'psi':0},  
    withIndex=False, 
    quantizerType=QuantizerType.PQDICT, nwDistParams = {},
    subsetType = SubsetSelectionType.NO_OVERLAP, barycenterMaxIter=10,
    max_iters =0, kmeansWindowSize = 0, useLbKeoghToFit = True, computeDistanceCorrection = True,
    distanceCalculation = DISTANCECALCULATION.ASYMMETRIC, km_init = "k-means++"):
        self.subsetSize=subsetSize
        self.dictionarySize=dictionarySize
        self.distParams= distParams
        self.quantizationDistParams = quantizationDistParams
        self.withIndex=withIndex
        self.residuals=False
        self.quantizerType=quantizerType
        self.subsetType=subsetType
        self.barycenterMaxIter =barycenterMaxIter
        self.max_iters=max_iters
        self.useLbKeoghToFit = useLbKeoghToFit
        self.computeDistanceCorrection=computeDistanceCorrection
        self.km_init=km_init
        if kmeansWindowSize == 0:
            self.metric_params = None
        else:
            self.metric_params = {"global_constraint":"sakoe_chiba", "sakoe_chiba_radius":kmeansWindowSize}
        self.nwDistParams=nwDistParams
        self.distanceCalculation = distanceCalculation

#@cython.auto_pickle(True) 
cdef class PQDictionary():
    cdef public bint recursiveLayer
    cdef int depth

    #what I actually wanted cdef np.ndarray[np.int32_t, ndim=2] index
    cdef object index
    cdef object params
    cdef object kmeans
    cdef double **dist

    cdef object LQuant, UQuant, LApprox, UApprox  #enveloppes
    #cdef np.ndarray[np.double_t, ndim=2] distanceDTWMatrix
    cdef public object distanceDTWMatrix
    cdef list productQuantisers
    cdef int k
    #cdef ProductQuantiserParameters params
    
    def __init__(self, data, pqParams, depth, centers = None):
        cdef np.ndarray[dtype=np.double_t, ndim = 2] cur_np
        self.recursiveLayer = False
        if  depth+1 < len(pqParams):
            self.recursiveLayer=True 
            self.productQuantisers = []
        self.params=pqParams[depth]
        self.kmeans = TimeSeriesKMeans(metric_params = self.params.metric_params, n_clusters=self.params.dictionarySize,random_state=0,verbose = 0,metric="dtw", max_iter_barycenter=self.params.barycenterMaxIter, max_iter=self.params.max_iters, init = self.params.km_init)
        if self.params.dictionarySize == len(data):
            self.kmeans.cluster_centers_=np.reshape(data,(data.shape[0], data.shape[1],1)).copy()

        else: 
            self.kmeans.fit(to_time_series_dataset(data))
        self.LQuant, self.UQuant = lb_keogh_enveloppes_fast(np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])), self.params.quantizationDistParams['window'])
        self.LApprox, self.UApprox = lb_keogh_enveloppes_fast(np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])), self.params.distParams['window'])
        if self.params.withIndex or self.recursiveLayer:
            predictions = self.kmeans.predict(data)
            self.createIndex(predictions)
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
    cpdef executePrecalculations(self):
        self.LQuant, self.UQuant = lb_keogh_enveloppes_fast(np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])), self.params.quantizationDistParams['window'])
        self.LApprox, self.UApprox = lb_keogh_enveloppes_fast(np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])), self.params.distParams['window'])
        self.distanceDTWMatrix = distance_matrix(np.reshape(self.kmeans.cluster_centers_ ,(self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])),**self.params.distParams )
        for i in range(0,self.distanceDTWMatrix.shape[0]):
            self.distanceDTWMatrix[i,i]=0
            for j in range(i+1,self.distanceDTWMatrix.shape[0]):
                self.distanceDTWMatrix[j,i] = self.distanceDTWMatrix[i,j]


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef createIndex(self, np.ndarray[np.int_t, ndim=1] codedData):
        self.index=[]
        for i in range(0,self.params.dictionarySize):
            self.index.append(np.where(codedData == i))
    
    
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef replaceCodeBook(self, np.ndarray[np.double_t, ndim=2] codeBook):
        self.kmeans.cluster_centers_= codeBook
        self.LApprox, self.UApprox = lb_keogh_enveloppes_fast(np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])), self.params.distParams['window'])
        self.LQuant, self.UQuant = lb_keogh_enveloppes_fast(np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])), self.params.quantizationDistParams['window'])

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef retrieveCodes(self, np.ndarray[np.double_t, ndim=2] data):
        cdef np.ndarray[np.int_t, ndim=1] distMatrix2
        cdef np.ndarray[np.double_t, ndim=2] distMatrix
        if False ==  self.params.useLbKeoghToFit:
            distMatrix = distance_matrix(np.concatenate((data, np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])))),block=((0,data.shape[0]), (data.shape[0], data.shape[0]+self.kmeans.cluster_centers_.shape[0])),**self.params.quantizationDistParams )
            distMatrix2=np.argmin(distMatrix, axis=1)[0:len(data)]
            return distMatrix2-len(data)
        #return self.kmeans.predict(data)
        elif self.params.useLbKeoghToFit:
    
            return nearest_neighbour_lb_keogh_fast(data,np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])),self.LQuant, self.UQuant, self.params.quantizationDistParams)
    
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef retrieveAsymDists(self, np.ndarray[np.double_t, ndim=2] data):
        cdef np.ndarray[np.int_t, ndim=1] distMatrix2
        cdef np.ndarray[np.double_t, ndim=2] distMatrix
        distMatrix = distance_matrix(np.concatenate((data, np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1])))),block=((0,data.shape[0]), (data.shape[0], data.shape[0]+self.kmeans.cluster_centers_.shape[0])),**self.params.distParams )
        distMatrix = distMatrix[0:data.shape[0], data.shape[0]: data.shape[0]+self.kmeans.cluster_centers_.shape[0]]
        return distMatrix

    
    cpdef getCodeBook(self):
        return np.reshape(self.kmeans.cluster_centers_, (self.kmeans.cluster_centers_.shape[0],self.kmeans.cluster_centers_.shape[1]))


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef double retrieveApprDTWDistance(self, int code1, int code2):
        return self.distanceDTWMatrix[code1, code2]

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef double retrieveRecApprDTWDistance(self, int code, np.ndarray[np.double_t, ndim=2] data):
        if not self.recursiveLayer:
            return 0
        return self.productQuantisers[code].calculateApprDTWDistanceForData(data)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef addScoresToDistMatrix(self, np.ndarray[ndim=1, dtype=np.int_t] code, np.ndarray[ndim=2, dtype=np.double_t] dist):
        self.addScoresToDistMatrix_c(code, dist, self.distanceDTWMatrix)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef addScoresToDistMatrix_c(self, np.ndarray[ndim=1, dtype=np.int_t] code, np.ndarray[ndim=2, dtype=np.double_t] dist, np.ndarray[ndim=2, dtype=np.double_t] precalc):
        cdef int i, j 
        for i in range(len(dist)):
            for j in range(i+1,len(dist)):
                dist[i,j] = dist[i,j] + precalc[code[i], code[j]]**2 

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef addScoresToDistMatrix_rec(self, np.ndarray[ndim=1, dtype=np.int_t] code, np.ndarray[ndim=2, dtype=np.double_t] dist, np.ndarray[ndim=2, dtype=np.double_t] data):
        self.addScoresToDistMatrix_rec_c(code, dist, data, self.distanceDTWMatrix)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef addScoresToDistMatrix_rec_c(self, np.ndarray[ndim=1, dtype=np.int_t] code, np.ndarray[ndim=2, dtype=np.double_t] dist, np.ndarray[ndim=2, dtype=np.double_t] data, np.ndarray[ndim=2, dtype=np.double_t] precalc):
        cdef int i, j, 
        cdef np.ndarray[dtype=np.double_t, ndim = 2] sliced =np.zeros((2, data.shape[1]), dtype=np.double)
        for i in range(len(dist)):
            for j in range(i+1,len(dist)):
                if code[i] == code[j]:
                    sliced[0,:] = data[i,:]
                    sliced[1,:] = data[j,:]
                    dist[i,j] = dist[i,j] + self.retrieveRecApprDTWDistance(code[i], sliced)
                else:
                    dist[i,j] = dist[i,j] + precalc[code[i], code[j]]**2

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef addScoresToDistMatrix_repl0(self, np.ndarray[ndim=1, dtype=np.int_t] code, np.ndarray[ndim=2, dtype=np.double_t] dist, np.ndarray[ndim=2, dtype=np.double_t] data):
        self.addScoresToDistMatrix_repl0_c(code, dist, data, self.distanceDTWMatrix)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef addScoresToDistMatrix_repl0_c(self, np.ndarray[ndim=1, dtype=np.int_t] code, np.ndarray[ndim=2, dtype=np.double_t] dist, np.ndarray[ndim=2, dtype=np.double_t] data, np.ndarray[ndim=2, dtype=np.double_t] precalc):
        cdef int i, j, 
        cdef np.ndarray[dtype=np.double_t, ndim = 2] sliced 
        for i in range(len(dist)):
            for j in range(i+1,len(dist)):
                if code[i] == code[j]:
                    dist[i,j] = dist[i,j] + distance_fast(data[i,:], data[j,:], **self.params.distParams)**2
                else:
                    dist[i,j] = dist[i,j] + precalc[code[i], code[j]]**2

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef addScoresToDistMatrix_keogh0(self, np.ndarray[ndim=1, dtype=np.int_t] code, np.ndarray[ndim=2, dtype=np.double_t] dist, np.ndarray[ndim=2, dtype=np.double_t] data):
        self.addScoresToDistMatrix_keogh0_c(code, dist, data, self.distanceDTWMatrix)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef addScoresToDistMatrix_keogh0_c(self, np.ndarray[ndim=1, dtype=np.int_t] code, np.ndarray[ndim=2, dtype=np.double_t] dist, np.ndarray[ndim=2, dtype=np.double_t] data, np.ndarray[ndim=2, dtype=np.double_t] precalc):
        #create keogh score matric + fitrs here per subset 
        cdef np.ndarray[np.double_t, ndim=2] lb = lb_keogh_distance_fast(data, self.LApprox, self.UApprox)
        cdef int i, j, 
        cdef np.ndarray[dtype=np.double_t, ndim = 2] sliced 
        for i in range(len(dist)):
            for j in range(i+1,len(dist)):
                if code[i] == code[j]:
                    dist[i,j] = dist[i,j] + max(lb[i,code[j]], lb[j,code[i]]) **2
                else:
                    dist[i,j] = dist[i,j] + precalc[code[i], code[j]]**2


    def resetParamsAndPrecalculate(self, pqParams, depth = 0):
        self.params = pqParams[depth]  
        self.executePrecalculations()
        for pq in self.productQuantisers:
            pq.resetParamsAndPrecalculate(pqParams, depth+1)


   


    def switchDistanceCalculation(self,dc):
        self.params.distanceCalculation=dc



cdef class ProductQuantizer():
    cdef int nrDictionaries
    cdef int subsetSize
    cdef int halfSubsetSize
    cdef list dictionaries
    cdef object params
    cdef object substitution
    cdef double overlapCorrector
    cdef double distanceRatio
    #ProductQuantiserParameters params

    def __init__(self,data, pqParams, depth=0):
        self.overlapCorrector = 1.0
        cdef int i 
        cdef np.ndarray[dtype=np.double_t, ndim=2] codeBook, L, U
        
        self.nrDictionaries = math.ceil(data.shape[1]/pqParams[depth].subsetSize) 
        self.subsetSize = pqParams[depth].subsetSize
        self.halfSubsetSize=self.subsetSize/2
        self.params=pqParams[depth]
        self.dictionaries = []
        self.distanceRatio = 1.0
        if self.params.subsetType == SubsetSelectionType.NO_OVERLAP:
            for i in range(0,self.nrDictionaries):
                self.dictionaries.append(PQDictionary(data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])],pqParams, depth)) 
        else:
            self.dictionaries.append(PQDictionary(data[:,0*self.subsetSize:min((0+1)*self.subsetSize,data.shape[1])],pqParams, depth))
            
            for i in range(1,self.nrDictionaries):
                self.dictionaries.append(PQDictionary(data[:,(i*self.subsetSize-self.halfSubsetSize):min((i+1)*self.subsetSize-self.halfSubsetSize,data.shape[1])],pqParams, depth))
                self.dictionaries.append(PQDictionary(data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])],pqParams, depth)) 
              
        if self.params.quantizerType==QuantizerType.PQNeedlemanWunsch or self.params.quantizerType==QuantizerType.VQNeedlemanWunsch:
            codeBook=self.dictionaries[0].getCodeBook()
            for i in range(1, len(self.dictionaries)):
                codeBook = np.concatenate((codeBook,self.dictionaries[i].getCodeBook()))
            self.substitution = SparseSubstitutionMatrix(codeBook,distParams=pqParams[depth].distParams)

        if self.params.quantizerType==QuantizerType.VQNeedlemanWunsch:
            for i in range(len(self.dictionaries)):
                self.dictionaries[i].replaceCodeBook(codeBook)
        if self.params.computeDistanceCorrection:
            self.determinRatioForScores(data)
    
    cpdef setRatioForScores(self, ratio):
        self.distanceRatio = ratio

    
    cdef determinRatioForScores(self, data):
        cdef np.ndarray[dtype=np.double_t, ndim=2] truth, pred
        cdef double sumRatio, cnt
        cdef int i, j

        sumRatio = 0.0
        cnt = 0.0
        pred = self.constructApproximateDTWDistanceMatrix(data)
        truth = distance_matrix(data, **self.params.distParams)
        for i in range(0, len(pred)):
            for j in range(i+1, len(pred)):
                if pred[i,j]>0.01 and truth[i,j] >0.01:
                    sumRatio = sumRatio + truth[i,j]/pred[i,j]
                    cnt = cnt +1.0
        self.distanceRatio = sumRatio / cnt
        print ('ratio', self.distanceRatio, 'overlapRatio', self.overlapCorrector)


        
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef np.ndarray[np.int_t, ndim = 2] retrieveCodedData(self, np.ndarray[np.double_t,ndim=2] data):
        cdef np.ndarray[np.int_t, ndim=2] codedData
        cdef int i, dictionaryCount
        dictionaryCount=0

        if self.params.subsetType == SubsetSelectionType.NO_OVERLAP:
            codedData = np.zeros((data.shape[0],self.nrDictionaries), dtype=np.int)
        if self.params.subsetType == SubsetSelectionType.DOUBLE_OVERLAP:
            codedData = np.zeros((data.shape[0],self.nrDictionaries*2-1), dtype=np.int)
        
        if self.params.subsetType == SubsetSelectionType.NO_OVERLAP:    
            for i in range(0,self.nrDictionaries):
                codedData[:, i] = self.dictionaries[i].retrieveCodes(
                    data[:,i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])])
                if self.params.quantizerType ==  QuantizerType.PQNeedlemanWunsch and i > 0:
                    dictionaryCount = dictionaryCount + self.dictionaries[i-1].getCodeBook().shape[0]
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
                if self.params.quantizerType ==  QuantizerType.PQNeedlemanWunsch:
                    dictionaryCount = dictionaryCount + self.dictionaries[2*i-2].getCodeBook().shape[0]
                    codedData[:, 2*i-1] + dictionaryCount
                    dictionaryCount = dictionaryCount + self.dictionaries[2*i-1].getCodeBook().shape[0]
                    codedData[:, 2*i] + dictionaryCount
        
        return codedData


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef double calculateApprDTWDistanceForCodes(self, np.ndarray[np.int_t, ndim=1] code1, np.ndarray[np.int_t, ndim=1] code2, np.ndarray[np.double_t, ndim=1] data1, np.ndarray[np.double_t, ndim=1] data2):
        cdef double distance = 0.0
        cdef int i
        cdef object m
        cdef np.ndarray[dtype=np.double_t, ndim=2] speedUp

        if self.params.quantizerType==QuantizerType.PQNeedlemanWunsch or self.params.quantizerType==QuantizerType.VQNeedlemanWunsch:
            distance,m =needleman_wunsch(code1, code2,substitutionMatrix=self.substitution, **(self.params.nwDistParams))
            if self.params.subsetType == SubsetSelectionType.NO_OVERLAP:
                return np.sqrt(distance)
            else:
                return np.sqrt(distance*self.overlapCorrector)

        #all this logic is just for layers after recursion (slower, not a matrix operation)
        for i in range(0,code1.shape[0]):
           # print (distance)
            if self.dictionaries[i].recursiveLayer and code1[i] == code2[i]:
                if self.params.subsetType == SubsetSelectionType.NO_OVERLAP:
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
                speedUp = self.dictionaries[i].distanceDTWMatrix
                distance = distance+speedUp[code1[i], code2[i]]**2#retrieveApprDTWDistance(code1[i], code2[i])**2
        
        #print ('dist',distance, self.overlapCorrector)
        if self.params.subsetType == SubsetSelectionType.NO_OVERLAP:
            return np.sqrt(distance)
        else:
            return np.sqrt(distance*self.overlapCorrector)


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef public constructApproximateDTWDistanceMatrix(self,data):
        return self.constructApproximateDTWDistanceMatrixExecute(data)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef np.ndarray[np.double_t, ndim=2] constructApproximateDTWDistanceMatrixExecute(self, np.ndarray[np.double_t, ndim = 2] data):
        self.overlapCorrector = <double>(self.nrDictionaries)/<double>(2*self.nrDictionaries-1)
        cdef np.ndarray[np.int_t, ndim=2] codedData = self.retrieveCodedData(data)
        cdef np.ndarray[np.double_t, ndim=2] approximateMatrix = np.zeros((data.shape[0], data.shape[0]))
        cdef int xi, xj, k, i
        cdef np.ndarray[np.double_t, ndim = 2] assymDists  
        approximateMatrix[:]=np.inf
        for  xi in range(0, data.shape[0]):
            for xj in range(xi+1, data.shape[0]):
                approximateMatrix[xi, xj]= 0

        if self.params.quantizerType != QuantizerType.PQDICT:
            print('Working with NW')
            for  xi in range(0, data.shape[0]):
                for xj in range(xi+1, data.shape[0]):
                    approximateMatrix[xi, xj]=self.calculateApprDTWDistanceForCodes(codedData[xi,:], codedData[xj,:], data[xi,:], data[xj,:])
        elif self.params.distanceCalculation == DISTANCECALCULATION.ASYMMETRIC:
            print('Working with assymetric quantizer')
            for i in range(len(self.dictionaries)):
               # self.dictionaries[k].createIndex(codedData[:,i])
                if self.params.subsetType == SubsetSelectionType.NO_OVERLAP:
                    assymDists = self.dictionaries[i].retrieveAsymDists(data[:, i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])]) 
                elif self.params.subsetType == SubsetSelectionType.DOUBLE_OVERLAP:
                    if i%2 == 0:
                        assymDists = self.dictionaries[i].retrieveAsymDists(data[:,int(i/2*self.subsetSize):int(min((i/2+1)*self.subsetSize,data.shape[1]))])
                    else:
                         assymDists = self.dictionaries[i].retrieveAsymDists(data[:,int((i+1)/2*self.subsetSize-self.halfSubsetSize):int(min(((i+1)/2+1)*self.subsetSize-self.halfSubsetSize,data.shape[1]))])
                for xi in range (data.shape[0]):
                    #for xj in range (xi+1,data.shape[0]):
                    approximateMatrix[xi, xi+1:data.shape[0]] = approximateMatrix[xi, xi+1:data.shape[0]] + assymDists[xi, codedData[xi+1:data.shape[0],i]]**2     
            if self.params.subsetType == SubsetSelectionType.DOUBLE_OVERLAP:
                approximateMatrix = np.sqrt(approximateMatrix*self.overlapCorrector)
            else:
                approximateMatrix = np.sqrt(approximateMatrix)

        elif self.params.distanceCalculation == DISTANCECALCULATION.EXACT_WHEM_0DIST:
            print('Working with exact replacements')
            if self.params.subsetType == SubsetSelectionType.DOUBLE_OVERLAP:
                for i in range(len(self.dictionaries)):
                    if i % 2 == 0:
                        self.dictionaries[i].addScoresToDistMatrix_repl0(codedData[:,i], approximateMatrix, data[:,int(i/2*self.subsetSize):int(min((i/2+1)*self.subsetSize,data.shape[1]))] )
                    else: 
                        self.dictionaries[i].addScoresToDistMatrix_repl0(codedData[:,i], approximateMatrix, data[:,int((i+1)/2*self.subsetSize-self.halfSubsetSize):int(min(((i+1)/2+1)*self.subsetSize-self.halfSubsetSize,data.shape[1]))])
                approximateMatrix = np.sqrt(approximateMatrix*self.overlapCorrector)
            else:
                for i in range(len(self.dictionaries)):
                    self.dictionaries[i].addScoresToDistMatrix_repl0(codedData[:,i], approximateMatrix, data[:, i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])] )
                approximateMatrix = np.sqrt(approximateMatrix)

        elif self.params.distanceCalculation == DISTANCECALCULATION.ASYMMETRIC_KEOGH_WHEM_0DIST:
            print('Working with keogh replacements')
            if self.params.subsetType == SubsetSelectionType.DOUBLE_OVERLAP:
                for i in range(len(self.dictionaries)):
                    if i % 2 == 0:
                        self.dictionaries[i].addScoresToDistMatrix_keogh0(codedData[:,i], approximateMatrix, data[:,int(i/2*self.subsetSize):int(min((i/2+1)*self.subsetSize,data.shape[1]))] )
                    else: 
                        self.dictionaries[i].addScoresToDistMatrix_keogh0(codedData[:,i], approximateMatrix, data[:,int((i+1)/2*self.subsetSize-self.halfSubsetSize):int(min(((i+1)/2+1)*self.subsetSize-self.halfSubsetSize,data.shape[1]))])
                approximateMatrix = np.sqrt(approximateMatrix*self.overlapCorrector)
            else:
                for i in range(len(self.dictionaries)):
                    self.dictionaries[i].addScoresToDistMatrix_keogh0(codedData[:,i], approximateMatrix, data[:, i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])] )
                approximateMatrix = np.sqrt(approximateMatrix)
            
        elif self.params.distanceCalculation == DISTANCECALCULATION.SYMMETRIC:
            print('Working with regular symmetric')
            for k in range(len(self.dictionaries)):
                self.dictionaries[k].addScoresToDistMatrix(codedData[:,k], approximateMatrix)
            if self.params.subsetType == SubsetSelectionType.DOUBLE_OVERLAP:
                approximateMatrix = np.sqrt(approximateMatrix*self.overlapCorrector)
            else:
                approximateMatrix = np.sqrt(approximateMatrix)
            
        else:#recursing  self.params.distanceCalculation == DISTANCECALCULATION.RECURSION
            print('Working with recursive replacements')
            if self.params.subsetType == SubsetSelectionType.DOUBLE_OVERLAP:
                for i in range(len(self.dictionaries)):
                    if i % 2 == 0:
                        self.dictionaries[i].addScoresToDistMatrix_rec(codedData[:,i], approximateMatrix, data[:,int(i/2*self.subsetSize):int(min((i/2+1)*self.subsetSize,data.shape[1]))] )
                    else: 
                        self.dictionaries[i].addScoresToDistMatrix_rec(codedData[:,i], approximateMatrix, data[:,int((i+1)/2*self.subsetSize-self.halfSubsetSize):int(min(((i+1)/2+1)*self.subsetSize-self.halfSubsetSize,data.shape[1]))])
                approximateMatrix = np.sqrt(approximateMatrix*self.overlapCorrector)
            else:
                for i in range(len(self.dictionaries)):
                    self.dictionaries[i].addScoresToDistMatrix_rec(codedData[:,i], approximateMatrix, data[:, i*self.subsetSize:min((i+1)*self.subsetSize,data.shape[1])] )
                approximateMatrix = np.sqrt(approximateMatrix)

        return approximateMatrix*self.distanceRatio

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef double calculateApprDTWDistanceForData(self, np.ndarray[np.double_t, ndim=2] data):
        cdef np.ndarray[np.int_t, ndim=2] codedData = self.retrieveCodedData(data)
        return self.calculateApprDTWDistanceForCodes(codedData[0,:], codedData[1,:], data[0,:], data[1,:])


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef indexData(self, np.ndarray[np.double_t,ndim=2] data):
        cdef int i
        cdef np.ndarray[np.int_t, ndim=2] codedData = self.retrieveCodedData(data)
        for i in range(len(self.dictionaries)):
            self.dictionaries[i].indexData(codedData[:,i])

    '''
        For each datapoint, for each codebook, determine all w nearest codes 
        Create set of all indexed datapoints
    '''
   

    def resetParamsAndPrecalculate(self, pqParams, depth = 0):
        self.params = pqParams[0]  
        for i in range(len(self.dictionaries)):
            self.dictionaries[i].resetParamsAndPrecalculate(pqParams, depth)

    def switchDistanceCalculation(self,dc):
        self.params.distanceCalculation=dc
        for dic in self.dictionaries:
            dic.switchDistanceCalculation(dc)


class PyProductQuantizer():
    def __init__(self, data=None, pqParams=None):
        if pqParams is None:
            print('no params, load from file?')
        self.pq = ProductQuantizer(data, pqParams)

    def loadFromFile(self, fileName):
        pickle_in = open(fileName,"rb")
        self.pq = pickle.load(pickle_in)

    
    '''redo hyp[erpparams, invoke reprecalculation with ther distance aparams, no recaulculation of codebooks!
        This functionality is there for quick testing purposes and is more a hack than a real functionality.
        use case examples:
            Change the handling of 0distance
            Chaneg window size of precalculation 
        not example use cases:
            Chanmeg dictionary/subset size
            recalculate underestimate/overestimatecorrection (no data provided)
            Change amount of recusrsion layer (But could turn of recursion)
    '''
    def resetParamsAndPrecalculate(self, pqParams):
        self.pq.resetParamsAndPrecalculate(pqParams)

    def switchDistanceCalculation(self, dc):
        self.pq.switchDistanceCalculation(dc)



    def constructApproximateDTWDistanceMatrix(self, data):
        v=self.pq.constructApproximateDTWDistanceMatrix(data)
       # print(v[1:6,1:6])
        return v

 

