import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd
from scipy import sparse


def sparseload(filename, sep=',', dtype=float, chunksize=1000, index_col=0, droplastcolumns=0):
    with open(filename) as file:
        genes = []
        sparseblocks = []
        for i,chunk in enumerate(pd.read_csv(filename, chunksize=chunksize, sep=sep, index_col=index_col)):
            print('.', end='', flush=True)
            if i==0:
                cells = np.array(chunk.columns)
            genes.extend(list(chunk.index))
            sparseblock = sparse.csr_matrix(chunk.values.astype(dtype))
            sparseblocks.append([sparseblock])
        counts = sparse.bmat(sparseblocks)
        print(' done')

    if droplastcolumns > 0:
        end = cells.size - droplastcolumns
        cells = cells[:end]
        counts = counts[:,:end]
        
    return (counts.T, np.array(genes), cells)


def geneSelection(data, threshold=0, atleast=10, 
                  yoffset=.02, xoffset=5, decay=1.5, n=None, 
                  plot=True, markers=None, genes=None, figsize=(6,3.5),
                  markeroffsets=None, labelsize=10, alpha=1):
    
    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data>threshold).mean(axis=0)))
        A = data.multiply(data>threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:,detected].mean(axis=0))) / (1-zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data>threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:,detected]>threshold
        logs = np.zeros_like(data[:,detected]) * np.nan
        logs[mask] = np.log2(data[:,detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    lowDetection = np.array(np.sum(data>threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan
            
    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low)/2
            else:
                low = xoffset
                xoffset = (xoffset + up)/2
        print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
                
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold>0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1]+.1,.1)
        y = np.exp(-decay*(x - xoffset)) + yoffset
        if decay==1:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-x+{:.2f})+{:.2f}'.format(np.sum(selected),xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)
        else:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected),decay,xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:,None],y[:,None]),axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
        plt.gca().add_patch(t)
        
        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        if threshold==0:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of zero expression')
        else:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of near-zero expression')
        plt.tight_layout()
        
        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num,g in enumerate(markers):
                i = np.where(genes==g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = markeroffsets[num]
                plt.text(meanExpr[i]+dx+.1, zeroRate[i]+dy, g, color='k', fontsize=labelsize)
    
    return selected



# Computing the matrix of Euclidean distances
def pdist2(A,B):
    D = np.sum(A**2,axis=1,keepdims=True) + np.sum(B**2, axis=1, keepdims=True).T - 2*A@B.T
    return D

import warnings

# Computing the matrix of correlations
def corr2(A,B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    ssA = (A**2).sum(axis=1, keepdims=True)
    ssB = (B**2).sum(axis=1, keepdims=True)
    # this ignores the NaN warnings. The result can have nans!
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        C = np.dot(A, B.T) / np.sqrt(np.dot(ssA,ssB.T))
    return C

def map_to_tsne(referenceCounts, referenceGenes, newCounts, newGenes, referenceAtlas, 
                bootstrap = False, knn = 10, nrep = 100, seed = None, batchsize = 1000,
				verbose = 1):
    gg = sorted(list(set(referenceGenes) & set(newGenes)))
    if verbose > 0:
        print('Using a common set of ' + str(len(gg)) + ' genes.')
    
    newGenes = [np.where(newGenes==g)[0][0] for g in gg]
    refGenes = [np.where(referenceGenes==g)[0][0] for g in gg]
    X = newCounts[:,newGenes]
    if sparse.issparse(X):
        X = np.array(X.todense())
    X = np.log2(X + 1)
    T = referenceCounts[:,refGenes]
    if sparse.issparse(T):
        T = np.array(T.todense())
    T = np.log2(T + 1)
    
    n = X.shape[0]
    assignmentPositions = np.zeros((n, referenceAtlas.shape[1]))
    batchCount = int(np.ceil(n/batchsize))
    if (batchCount > 1) and (verbose > 0):
        print('Processing in batches', end='', flush=True) 
    for b in range(batchCount):
        if (batchCount > 1) and (verbose > 0):
            print('.', end='', flush=True) 
        batch = np.arange(b*batchsize, np.minimum((b+1)*batchsize, n))
        C = corr2(X[batch,:], T)
        ind = np.argpartition(C, -knn)[:, -knn:]
        for i in range(batch.size):
            assignmentPositions[batch[i],:] = np.median(referenceAtlas[ind[i,:],:], axis=0)
    if (batchCount > 1) and (verbose > 0):
        print(' done', flush=True) 
    
    # Note: currently bootstrapping does not support batchsize
    if bootstrap:
        if seed is not None:
            np.random.seed(seed)
        assignmentPositions_boot = np.zeros((n, referenceAtlas.shape[1], nrep))
        if verbose>0:
            print('Bootstrapping', end='', flush=True)
        for rep in range(nrep):
            if verbose>0:
                print('.', end='')
            bootgenes = np.random.choice(T.shape[1], T.shape[1], replace=True)
            C_boot = corr2(X[:,bootgenes],T[:,bootgenes])
            ind = np.argpartition(C_boot, -knn)[:, -knn:]
            for i in range(X.shape[0]):
                assignmentPositions_boot[i,:,rep] = np.median(referenceAtlas[ind[i,:],:], axis=0)
        if verbose>0:
            print(' done')      
        return (assignmentPositions, assignmentPositions_boot)
    else:
        return assignmentPositions


def map_to_clusters(referenceCounts, referenceGenes,
                    newCounts, newGenes, 
                    referenceClusters, referenceClusterNames=[], cellNames=[],
                    bootstrap = False, nrep = 100, seed = None, verbose = False, until=.95,
                    returnCmeans = False, totalClusters = None):

    gg = sorted(list(set(referenceGenes) & set(newGenes)))
    print('Using a common set of ' + str(len(gg)) + ' genes.')
    
    newGenes = [np.where(newGenes==g)[0][0] for g in gg]
    refGenes = [np.where(referenceGenes==g)[0][0] for g in gg]
    X = newCounts[:,newGenes]
    if sparse.issparse(X):
        X = np.array(X.todense())
    X = np.log2(X + 1)
    T = referenceCounts[:,refGenes]
    if sparse.issparse(T):
        T = np.array(T.todense())
    T = np.log2(T + 1)
    
    if totalClusters is not None:
        K = totalClusters
    else:
        K = np.max(referenceClusters) + 1
    means = np.zeros((K, T.shape[1]))
    for c in range(K):
        if np.sum(referenceClusters==c) > 0:
            means[c,:] = np.mean(T[referenceClusters==c,:], axis=0)

    Cmeans = corr2(X, means)
    allnans = np.sum(np.isnan(Cmeans), axis=1) == Cmeans.shape[1]
    clusterAssignment = np.zeros(Cmeans.shape[0]) * np.nan
    clusterAssignment[~allnans] = np.nanargmax(Cmeans[~allnans,:], axis=1)
    
    if bootstrap:
        if seed is not None:
            np.random.seed(seed)
            
        clusterAssignment_boot = np.zeros((X.shape[0], nrep), dtype=int)
        for rep in range(nrep):
            print('.', end='', flush=True) 
            bootgenes = np.random.choice(T.shape[1], T.shape[1], replace=True)
            Cmeans_boot = corr2(X[:,bootgenes], means[:,bootgenes])
            m = np.zeros(Cmeans.shape[0]) * np.nan
            m[~allnans] = np.nanargmax(Cmeans_boot[~allnans,:], axis=1)
            clusterAssignment_boot[:,rep] = m
        print(' done')
        
        clusterAssignment_matrix = np.zeros((X.shape[0], K))
        for cell in range(X.shape[0]):
            mapsto, mapsto_counts = np.unique(clusterAssignment_boot[cell,:], return_counts=True)
            for i,m in enumerate(mapsto):
                clusterAssignment_matrix[cell, m] = mapsto_counts[i] / nrep
    
        if verbose:
            for rownum,row in enumerate(clusterAssignment_matrix):
                ind = np.argsort(row)[::-1]
                ind = ind[:np.where(np.cumsum(row[ind]) >= until)[0][0] + 1]
                mystring = []
                for i in ind:
                    s = referenceClusterNames[i] + ' ({:.1f}%)'.format(100*row[i])
                    mystring.append(s)
                mystring = cellNames[rownum] + ': ' + ', '.join(mystring)
                print(mystring)
            
        if returnCmeans:
            return clusterAssignment, clusterAssignment_matrix, Cmeans
        else:
            return clusterAssignment, clusterAssignment_matrix
    
    else:
        if returnCmeans:
            return clusterAssignment, Cmeans
        else:
            return clusterAssignment


