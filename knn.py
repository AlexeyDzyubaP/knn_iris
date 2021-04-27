import math

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Distances
def euclidian_distance(v1, v2):
    length, summation = len(v1), 0
    for i in range(length):
        summation += math.pow(v1[i] - v2[i], 2)

    return math.sqrt(summation)


def manhat_distance(v1, v2):
    length, summation = len(v1), 0
    for i in range(length):
        summation += np.fabs(v1[i] - v2[i])

    return summation
    
 
def cheb_distance(u, v):

    return max(abs(u - v))
       
        
# Kernels  
def unif_kernel(ro_h):
    
    return 0.5


def triangular_kernel(ro_h):
    
    return 1-abs(ro_h)
    
 
def epan_kernel(ro_h):
    
    return 0.75*(1 - ro_h*ro_h)
    
    
def quar_kernel(ro_h):
    
    return 15/16*(1 - ro_h*ro_h)*(1 - ro_h*ro_h)
         
        
# Window size functions
def h_width(distMatrix):
    RD = np.amax(distMatrix)
    N = distMatrix.shape[0]   # |D|
    nnMax = int(math.sqrt(N))  # sqrt(|D|)
    print(nnMax)
    h = np.zeros((2, N, nnMax))
    
    for j in range(X.shape[0]):
        h[0,j,:] = distMatrix[j, 1:nnMax+1]  # nn width
        h[1,j,:] = np.linspace(RD/nnMax,RD,nnMax)
        #print('steps = ', RD/nnMax, h[1,0,1]-h[1,0,0])  # steps =  0.15231702435118044 0.15231702435118044
    
    return h


def compute_confusion_matrix(true, pred, Nlbl):
    result = np.zeros((Nlbl, Nlbl))    
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
        
    return result
    


def calcDistMatrix(X, distFunc):
    # Initializing dict of distances and variable with size of training set
    X_shape = X.shape
    print(X_shape)
    indxMatrix = np.zeros((X_shape[0],X_shape[0]))
    distMatrix = np.zeros((X_shape[0],X_shape[0]))

    # Calculating the Euclidean distance between the new
    # sample and the values of the training sample
    for j in range(X_shape[0]):
        distances = {}
        for i in range(X_shape[0]):
            d = distFunc(X[j,:], X[i,:])
            distances[i] = d
    
        # Selecting the |D| nearest neighbors
        #indx_neighbors = sorted(distances, key=distances.get)[:]
        #distMatrix[j,:] = indx_neighbors
        dist_neighbors = sorted(distances.items(), key=lambda x: x[1])
        indxMatrix[j,:] = [item[0] for item in dist_neighbors]
        distMatrix[j,:] = [item[1] for item in dist_neighbors]
        
    return indxMatrix, distMatrix
    


filename = 'car.csv'

df = pd.read_csv(filename, na_values="?" )
print(df.head())
print(df.shape)

#df['buying'] = df['buying'].astype('category')
#df['maint'] = df['maint'].astype('category')
#df['doors'] = df['doors'].astype('category')
#df['persons'] = df['persons'].astype('category')
#df['lug_boot'] = df['lug_boot'].astype('category')
#df['safety'] = df['safety'].astype('category')
df['class'] = df['class'].astype('category')
print(df.dtypes)
#cat_columns = df.select_dtypes(['category']).columns
#df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

df[:] = df[:].astype('category')
df[:] = df[:].apply(lambda x: x.cat.codes)
print(df.values.shape)

X = df.values[:, :6]
Y = df.values[:, 6]
print(X, '====', Y)
Nlbl = len(np.unique(Y)) # Number of classes 

print(X.ptp(0))
X_norm = (X - X.min(0)) / X.ptp(0)  # X.ptp - peak to peak range
print(X_norm)

Y_1hot = pd.get_dummies(Y)  # one hot coding
#print(Y_1hot)

distFuncs = [manhat_distance,euclidian_distance, cheb_distance]
kernFuncs = [triangular_kernel, unif_kernel, epan_kernel, quar_kernel]

F1_max = 0
for ifnc in range(len(distFuncs)): 
    for ikrn in range(len(kernFuncs)): 

        indxMatrix, distMatrix = calcDistMatrix(X, distFuncs[ifnc])
        #print('distMatrix = ', distMatrix.shape)
        print(indxMatrix)
        print(distMatrix)

        #h = distMatrix[0, 100]
        hh = h_width(distMatrix)  # hh array size (2, |D|, sqrt(|D|)
        #print('hh. size = ', hh.shape)

        Y_estim = np.zeros(len(Y))

        for ih1 in range(hh.shape[0]): 
            F1_graph = np.zeros(hh.shape[2])
            for ih2 in range(hh.shape[2]):

                for j in range(X.shape[0]):  # dataset size = N
                    up_sum = np.zeros(Y_1hot.shape[1])
                    dn_sum = 0

                    h = hh[ih1+1,j,ih2]
                    #print('h = ', h)
    
                    for i in range(1, X.shape[0]): 
                        if distMatrix[j,i] <= h:
                            kern_val = kernFuncs[ikrn](distMatrix[j,i]/h)
                            up_sum += Y_1hot.values[i,:] * kern_val  # vector
                            #print('kernval = ', kern_val, up_sum)
                            dn_sum += kern_val  # scalar
                        else:
                            continue
        
                    if dn_sum == 0:
                        continue
                    Y_1hot_estim = up_sum/dn_sum  # vector
                    Y_estim[j] = np.argmax(Y_1hot_estim)

                #Y = Y[:10]
                Y_estim = Y_estim.astype(int)
                #print('Ys = ', Y, Y_estim)

                confusion_matrix = compute_confusion_matrix(Y, Y_estim, Nlbl)
                FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
                FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                TP = np.diag(confusion_matrix)
                TP = TP.sum()
                FP = FP.sum()
                FN = FN.sum()
                #print('perfs = ', TP, FP, FN)
                # Sensitivity, hit rate, recall, or true positive rate
                TPR = TP/(TP+FN)
                # Precision or positive predictive value
                PPV = TP/(TP+FP)
                # Negative predictive value
                F1_score = 2*(TPR*PPV)/(TPR+PPV)
                F1_graph[ih2] = F1_score
                if F1_score > F1_max:
                    F1_max = F1_score
                    ifnc_max = ifnc
                    ikrn_max = ikrn
                    ih1_max = ih1
                    ih2_max = ih2
                print('F1 = ', ifnc, ikrn, ih1, ih2, F1_score)
            plt.plot(F1_graph)
            plt.show()
                
print('ifnc, ikrn, ih1, ih2, F1_max = ', ifnc_max, ikrn_max, ih1_max, ih2_max, F1_max)

