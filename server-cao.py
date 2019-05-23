import numpy as np
import pandas as pd
import scanpy.api as sc
sc.settings.verbosity = 3
import pickle

import sys
sys.path.append('/gpfs01/berens/user/dkobak/FIt-SNE')
from fast_tsne import fast_tsne


# LOAD AND PREPROCESS THE DATA
# data from https://oncoscape.v3.sttrcancer.org/atlas.gs.washington.edu.mouse.rna/downloads

from scipy.io import mmread
from scipy import sparse
import anndata
%time counts = mmread('big-data/cao/gene_count.txt').transpose()   # 38 min!!
counts = sparse.csr_matrix(counts)
adata = anndata.AnnData(counts)

sc.pp.recipe_zheng17(adata, n_top_genes=2000)  # 2 min

X = np.copy(adata.X)
X = X - X.mean(axis=0)
# Note: svd took extra ~60Gb RAM. scipy.sparse.svds would probably be better here.
# I did not want to use randomized algorithms for this paper.
%time U, s, V = np.linalg.svd(X, full_matrices=False) # 1 min 46s
U[:, np.sum(V,axis=1)<0] *= -1
X = np.dot(U, np.diag(s))
X = X[:, np.argsort(s)[::-1]][:,:50]
pickle.dump(X, open('big-pickles/cao-pca.pickle', 'wb'))

# read in cluster labels
meta = pd.read_csv('big-data/cao/cell_annotate.csv')
clusters = meta['Main_Cluster'].values.copy()
clusters[~np.isnan(clusters)] -= 1
clusters[np.isnan(clusters)] = -1
clusters = clusters.astype(int)

# cluster names
clusterNames = meta['Main_cell_type'][np.unique(clusters, return_index=True)[1][1:]].values.copy()

# t-SNE embedding from the paper
Zpaper = np.concatenate((meta['Main_cluster_tsne_1'].values[:,np.newaxis],
                         meta['Main_cluster_tsne_2'].values[:,np.newaxis]), axis=1)


# DOWNSAMPLE AND RUN t-SNE

X = pickle.load(open('big-pickles/cao-pca.pickle', 'rb'))

PCAinit = X[:,:2] / np.std(X[:,0]) * 0.0001
np.random.seed(42)
ind25k  = np.random.choice(X.shape[0], 25000, replace=False)
Z25k = fast_tsne(X[ind25k,:], perplexity_list=[30,int(25000/100)], 
                 initialization = PCAinit[ind25k,:], seed=42, 
                 learning_rate = 25000/12)

def downsampled_nn(X, Z, downsampled_ind, batchsize=1000, knn=10):
	ind_rest = np.where(~np.isin(np.arange(X.shape[0]), downsampled_ind))[0]
	steps = int(np.ceil(ind_rest.size/batchsize))
	positions = np.zeros((X.shape[0], 2))
	positions[downsampled_ind,:] = Z
	def pdist2(A,B):
		return np.sum(A**2,axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T

	for i in range(steps):
		print('.', end='', flush=True)
		if (i+1)%100==0:
			print('', flush=True)
		endind = np.min(((i+1)*batchsize, ind_rest.size))
		batch = ind_rest[i*batchsize:endind]
		D = pdist2(X[batch, :], X[downsampled_ind,:])
		ind = np.argpartition(D, knn)[:, :knn]
		for i in range(batch.size):
			positions[batch[i],:] = np.median(Z[ind[i,:],:], axis=0)		
	print('', flush=True)
	return positions

%time positions = downsampled_nn(X, Z25k, ind25k, batchsize=10000) # 14 min
pickle.dump([Z25k, positions], open("big-pickles/cao-downsampling.pickle", "wb"))


# RUN T-SNE VARIANTS ON THE FULL DATA SET

X = pickle.load(open('big-pickles/cao-pca.pickle', 'rb'))
Z25k, positions = pickle.load(open("big-pickles/cao-downsampling.pickle", "rb"))

Zs = {}

init25k = positions/np.std(positions[:,0]) * 0.0001
%time Z = fast_tsne(X, perplexity=30, initialization=init25k, late_exag_coeff=4, start_late_exag_iter=250, learning_rate=X.shape[0]/12, seed=42, load_affinities='save') # 26min
Zs['mine'] = Z

Z = fast_tsne(X, perplexity=30, initialization=init25k, learning_rate=X.shape[0]/12, seed=42, load_affinities='load') 
Zs['noexagg'] = Z

%time Z = fast_tsne(X, perplexity=30, initialization=PCAinit, late_exag_coeff=4, start_late_exag_iter=250, learning_rate=X.shape[0]/12, seed=42, load_affinities='load') 
Zs['pcainit'] = Z

Z = fast_tsne(X, perplexity=30, initialization=PCAinit, learning_rate=X.shape[0]/12, seed=42, load_affinities='load') 
Zs['noexagg-pcainit'] = Z

Z = fast_tsne(X, perplexity=30, late_exag_coeff=4, start_late_exag_iter=250,  learning_rate=X.shape[0]/12, seed=42, load_affinities='load') 
Zs['randinit'] = Z

Z = fast_tsne(X, perplexity=30, learning_rate=1000, seed=42, load_affinities='load') 
Zs['scanpy'] = Z

Z = fast_tsne(X, perplexity=30, learning_rate=X.shape[0]/12, seed=42, load_affinities='load') 
Zs['belkina'] = Z

Z = fast_tsne(X, perplexity=30, seed=42, load_affinities='load') 
Zs['default'] = Z

import umap
%time Z = umap.UMAP(random_state=1).fit_transform(X) # 2h 5min
Zs['umap'] = Z

Zs['paper'] = Zpaper

pickle.dump([Zs, clusters, clusterNames], open("big-pickles/cao-tsne.pickle", "wb"))

