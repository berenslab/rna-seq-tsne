import numpy as np
import pickle

import sys
sys.path.append('/gpfs01/berens/user/dkobak/FIt-SNE')
from fast_tsne import fast_tsne


# LOAD AND PREPROCESS THE DATA

import scanpy.api as sc
sc.settings.verbosity = 2

# Data file is from here 
# https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.3.0/1M_neurons
adata = sc.read_10x_h5('big-data/10x/1M_neurons_filtered_gene_bc_matrices_h5.h5')
sc.pp.recipe_zheng17(adata)

X = np.copy(adata.X)
X = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X, full_matrices=False)
U[:, np.sum(V,axis=1)<0] *= -1
X = np.dot(U, np.diag(s))
X = X[:, np.argsort(s)[::-1]][:,:50]
pickle.dump(X, open('big-pickles/10x-pca.pickle', 'wb'))

# load cluster labels
# https://github.com/theislab/scanpy_usage/blob/master/170522_visualizing_one_million_cells/results/louvain.csv.gz
clusters = pd.read_csv('data/10x-1mln-scanpy-louvain.csv.gz', header=None).values[:,1].astype(int)


# DOWNSAMPLE AND RUN t-SNE

X = pickle.load(open('big-pickles/10x-pca.pickle', 'rb')).astype(float)
PCAinit = X[:,:2] / np.std(X[:,0]) * 0.0001

np.random.seed(42)
ind25k  = np.random.choice(X.shape[0], 25000, replace=False)
Z25k = fast_tsne(X[ind25k,:], perplexity_list=[30,int(25000/100)], 
                 initialization = PCAinit[ind25k,:], seed=42, 
                 learning_rate = 25000/12)
pickle.dump([Z25k, []], open("big-pickles/10x-downsampling.pickle", "wb"))

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

%time positions = downsampled_nn(X, Z25k, ind25k, batchsize=10000) # 10 min
pickle.dump([Z25k, positions], open("big-pickles/10x-downsampling.pickle", "wb"))


# RUN T-SNE VARIANTS ON THE FULL DATA SET

X = pickle.load(open('big-pickles/10x-pca.pickle', 'rb'))
Z25k, positions = pickle.load(open('big-pickles/10x-downsampling.pickle', 'rb'))

Zs = {}

init25k = positions/np.std(positions[:,0]) * 0.0001
%time Z = fast_tsne(X, perplexity=30, initialization=init25k, late_exag_coeff=4, start_late_exag_iter=250, learning_rate=X.shape[0]/12, seed=42, load_affinities='save') # 13 min 37 s
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
%time Z = umap.UMAP(random_state=1).fit_transform(X) # 56 min
Zs['umap'] = Z

pickle.dump([Zs, clusters], open("big-pickles/10x-tsne.pickle", "wb"))


# EXTRACT MARKER GENES

import collections
import scipy.sparse as sp_sparse
import tables
 
f = tables.open_file('big-data/10x/1M_neurons_filtered_gene_bc_matrices_h5.h5', 'r')
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

pickle.dump([markergenes, markerexp], open("big-pickles/10x-markers.pickle", "wb"))

