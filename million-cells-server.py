# ssh dkobak@172.25.250.112 -p 60222
# ssh 172.29.0.60-63

import numpy as np
import pickle

import sys
sys.path.append('/gpfs01/berens/user/dkobak/FIt-SNE')
from fast_tsne import fast_tsne


# LOAD AND PREPROCESS THE DATA

import scanpy.api as sc
sc.settings.verbosity = 2

adata = sc.read_10x_h5('1M_neurons_filtered_gene_bc_matrices_h5.h5')
sc.pp.recipe_zheng17(adata)

X = np.copy(adata.X)
X = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X, full_matrices=False)
U[:, np.sum(V,axis=1)<0] *= -1
X = np.dot(U, np.diag(s))
X = X[:, np.argsort(s)[::-1]][:,:50]
pickle.dump(X, open('pca-scanpy.pickle', 'wb'))


# DOWNSAMPLE AND RUN t-SNE

X = pickle.load(open('pca-scanpy.pickle', 'rb')).astype(float)
PCAinit = X[:,:2] / np.std(X[:,0]) * 0.0001

np.random.seed(42)
ind25k  = np.random.choice(X.shape[0], 25000, replace=False)
Z25k = fast_tsne(X[ind25k,:], perplexity=500, initialization=PCAinit[ind25k,:], seed=42)
pickle.dump([Z25k, []], open("downsampling.pickle", "wb"))


# POSITION ALL CELLS USING K=1 NN

def pdist2(A,B):
	return np.sum(A**2,axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T

# We do this in batches of 1000 cells to avoid computing an enormous matrix of 1.3mln x 25k size
batchsize = 1000
steps = int(np.floor(X.shape[0]/batchsize) + 1)
position_id = np.zeros(X.shape[0], dtype=int)
for i in range(steps):
	print('.', end='', flush=True)
	if i>1 and i%100==0:
		print('', flush=True)
	endind = np.min(((i+1)*batchsize, X.shape[0]))
	D = pdist2(X[ind25k,:], X[i*batchsize:endind, :])
	m = np.argmin(D, axis=0)
	position_id[i*batchsize:endind] = m
print('', flush=True)
pickle.dump([Z25k, position_id], open("downsampling.pickle", "wb"))


# FINAL t-SNE

X = pickle.load(open('pca-scanpy.pickle', 'rb')).astype(float)
Z25k, position_id = pickle.load(open("downsampling.pickle", "rb"))

init25k = Z25k[position_id,:]/np.std(Z25k[position_id,0]) * 0.0001

Z = fast_tsne(X, perplexity=30, initialization=init25k, early_exag_coeff=12, 
              stop_early_exag_iter=500, late_exag_coeff=4, start_late_exag_iter=500,
              learning_rate=1000, seed=42, load_affinities='save') 


# CONTROL EXPERIMENTS

Zs = [Z]

Z = fast_tsne(X, perplexity=30, initialization=init25k, early_exag_coeff=12, 
              stop_early_exag_iter=500, learning_rate=1000, load_affinities='load') 
Zs.append(Z)

Z = fast_tsne(X, perplexity=30, initialization=PCAinit, early_exag_coeff=12, 
              stop_early_exag_iter=500, late_exag_coeff=4, start_late_exag_iter=500,
              learning_rate=1000, load_affinities='load') 
Zs.append(Z)

Z = fast_tsne(X, perplexity=30, seed=42, learning_rate=1000, 
              load_affinities='load') 
Zs.append(Z)

# Additional control: aggressive early exaggeration

Z = fast_tsne(X, perplexity=30, initialization=PCAinit, early_exag_coeff=120, 
              stop_early_exag_iter=1000, late_exag_coeff=4, start_late_exag_iter=1000,
              max_iter=2000, learning_rate=1000, load_affinities='load')
Zs.append(Z)
pickle.dump(Zs, open("tsne-results.pickle", "wb"))


# EXTRACT MARKER GENES

import collections
import scipy.sparse as sp_sparse
import tables
 
f = tables.open_file('1M_neurons_filtered_gene_bc_matrices_h5.h5', 'r')
group = f.get_node(f.root, 'mm10')
gene_ids = getattr(group, 'genes').read()
gene_names = getattr(group, 'gene_names').read().astype(str)
barcodes = getattr(group, 'barcodes').read()
data = getattr(group, 'data').read()
indices = getattr(group, 'indices').read()
indptr = getattr(group, 'indptr').read()
shape = getattr(group, 'shape').read()

matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)

markergenes = ['Snap25', 'Slc17a6', 'Slc17a7', 'Gad1', 'Gad2', 
               'Slc32a1', 'Mog', 'Aqp4', 'Pdgfra', 'Itgam', 'Flt1', 
               'Bgn', 'Olig1', 'Gja1', 'Xdh', 'Ctss', 'Myl9', 
               'Vip', 'Sst', 'Pvalb', 'Nrn1', 'S1pr1', 'Gia1',
               'Gjb6', 'Lcat', 'Acsbg1', 'Neurod6', 'Akap7', 
               'Htr3a', 'Foxp2', 'Tubb23', 'Slc1a3', 'Top2a', 
               'Stmn2', 'Meg3', 'Nrp1', 'Tac2', 'Reln', 'Pax6', 
               'Tbr2', 'Tbr1', 'Eomes', 'Pax6', 'Tac1', 'Tubb3', 
               'Stmn2', 'Sox2', 'Aldoc', 'Hes1']

markerind   = np.array([i for i,g in enumerate(gene_names) if g in markergenes])
markergenes = np.array([g for i,g in enumerate(gene_names) if g in markergenes])
markerexp   = np.array(matrix[markerind,:].todense()).T.astype('float')

pickle.dump([markergenes, markerexp], open("markers.pickle", "wb"))


# UMAP

import umap
%time Z = umap.UMAP().fit_transform(X)
pickle.dump(Z, open("umap-results.pickle", "wb"))

