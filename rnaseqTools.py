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
                  yoffset=.02, xoffset=5, decay=1, n=None, 
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
        meanExpr[detected] = np.nanmean(np.where(data[:,detected]>threshold, np.log2(data[:,detected]), np.nan), axis=0)

    lowDetection = np.array(np.sum(data>threshold, axis=0)).squeeze() < atleast
    #lowDetection = (1 - zeroRate) * data.shape[0] < atleast - .00001
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

        plt.plot(x, y, color=sns.color_palette()[2], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:,None],y[:,None]),axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color='r', alpha=.2)
        plt.gca().add_patch(t)
        
        plt.scatter(meanExpr, zeroRate, s=3, alpha=alpha, rasterized=True)
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


def scatterPlot(Z, dataset, size=4, s=3, title=None, labels_dy=2, 
                rasterized=True, alpha=.5, showlabels=True, showmeans=True,
                hideclustermeans=None, hideticklabels=True,
                hideclusterlabels=[], markers=None,
                showclusterlabels=None, clusterlabeloffsets=None,
                clusterlabelcolor='k', clusterlabelsplit=False):
    if size is not None:
        plt.figure(figsize=(size,size))

    plt.scatter(Z[:,0], Z[:,1], s=s, color=dataset.clusterColors[dataset.clusters], 
                alpha=alpha, rasterized=rasterized)

    K = dataset.clusterNames.size
    Zmeans = np.zeros((K, 2))
    for c in range(K):
        Zmeans[c,:] = np.median(Z[dataset.clusters==c, :2], axis=0)
    if hideclustermeans is not None:
        Zmeans[hideclustermeans,:] = np.nan

    if showmeans:
        if markers is None:
            markers = np.array(['o'] * K)
        else:
            markers = np.array(markers)
        for m in np.unique(markers):
            nonans = ~np.isnan(Zmeans[:,0])
            plt.scatter(Zmeans[nonans,0][markers[nonans]==m], Zmeans[nonans,1][markers[nonans]==m],
                        color=dataset.clusterColors[nonans][markers[nonans]==m], marker=m,
                        s=40, edgecolor='k', linewidth=.7);

    if showlabels:
        if showclusterlabels is not None:
            if isinstance(showclusterlabels[0], str):
                showclusterlabels = [np.where(dataset.clusterNames==c)[0][0] for c in showclusterlabels]
            else:
                showclusterlabels = list(showclusterlabels)
            hideclusterlabels = np.array([c for c in range(K) if c not in showclusterlabels])
        for c in range(K):
            if ~np.isnan(Zmeans[c,0]) and c not in hideclusterlabels:
                if clusterlabeloffsets is not None and showclusterlabels is not None:
                    dx,dy = clusterlabeloffsets[showclusterlabels.index(c)]
                else:
                    dx,dy = 0,labels_dy

                if clusterlabelcolor is not None:
                    col = clusterlabelcolor
                else:
                    col = dataset.clusterColors[c]

                if clusterlabelsplit:
                    label = '\n'.join(dataset.clusterNames[c].split(' '))
                    hor = 'left'
                else:
                    label = dataset.clusterNames[c]
                    hor = 'center'

                plt.text(Zmeans[c,0]+dx, Zmeans[c,1]+dy, label, color=col, 
                         fontsize=7, horizontalalignment=hor) 
    
    if hideticklabels:
        plt.gca().get_xaxis().set_ticklabels([])
        plt.gca().get_yaxis().set_ticklabels([])
    if title is not None:
        plt.title(title)
    if size is not None:
        plt.tight_layout()


# Computing the matrix of Euclidean distances
def pdist2(A,B):
    D = np.sum(A**2,axis=1,keepdims=True) + np.sum(B**2, axis=1, keepdims=True).T - 2*A@B.T
    return D

# Computing the matrix of correlations
def corr2(A,B):
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    ssA = (A**2).sum(axis=1, keepdims=True)
    ssB = (B**2).sum(axis=1, keepdims=True)
    C = np.dot(A, B.T) / np.sqrt(np.dot(ssA,ssB.T))
    return C

def map_to_tsne(referenceCounts, referenceGenes, newCounts, newGenes, referenceAtlas, 
                bootstrap = False, knn = 25, nrep = 100, seed = None, batchsize = 1000):
    gg = list(set(referenceGenes) & set(newGenes))
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
    if batchCount > 1:
        print('Processing in batches', end='', flush=True) 
    for b in range(batchCount):
        if batchCount > 1:
            print('.', end='', flush=True) 
        batch = np.arange(b*batchsize, np.minimum((b+1)*batchsize, n))
        C = corr2(X[batch,:], T)
        for i in range(batch.size):
            ind = np.argsort(C[i,:])[::-1][:knn]                 
            assignmentPositions[batch[i],:] = np.median(referenceAtlas[ind,:], axis=0)
    if batchCount > 1:
        print(' done', flush=True) 
    
    # Note: currently bootstrapping does not support batchsize
    if bootstrap:
        if seed is not None:
            np.random.seed(seed)
        assignmentPositions_boot = np.zeros((n, referenceAtlas.shape[1], nrep))
        print('Bootstrapping', end='', flush=True)
        for rep in range(nrep):
            print('.', end='')
            bootgenes = np.random.choice(T.shape[1], T.shape[1], replace=True)
            C_boot = corr2(X[:,bootgenes],T[:,bootgenes])
            for i in range(X.shape[0]):
                ind = np.argsort(C_boot[i,:])[::-1][:knn]                 
                assignmentPositions_boot[i,:,rep] = np.median(referenceAtlas[ind,:], axis=0)
        print(' done')      
        return (assignmentPositions, assignmentPositions_boot)
    else:
        return assignmentPositions


def map_to_clusters(referenceCounts, referenceGenes,
                    newCounts, newGenes, 
                    referenceClusters, referenceClusterNames=[], cellNames=[],
                    bootstrap = False, nrep = 100, seed = None, verbose = False, until=.95,
					returnCmeans = False):

    gg = list(set(referenceGenes) & set(newGenes))
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
    
    K = np.max(referenceClusters) + 1
    means = np.zeros((K, T.shape[1]))
    for c in range(K):
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
            for row in clusterAssignment_matrix:
                ind = np.argsort(row)[::-1]
                ind = ind[:np.where(np.cumsum(row[ind]) >= until)[0][0] + 1]
                mystring = []
                for i in ind:
                    s = referenceClusterNames[i] + ' ({:.1f}%)'.format(100*row[i])
                    mystring.append(s)
                mystring = cellNames[i] + ': ' + ', '.join(mystring)
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


