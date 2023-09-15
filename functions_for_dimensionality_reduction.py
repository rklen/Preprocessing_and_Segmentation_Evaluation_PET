import numpy as np
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from scipy.ndimage import binary_dilation

# NOTE: indices given in this study

def reduce_dimensions_pca(tacs, index=np.nan, components=30):

    # Set nan's to -4 as they appear when the standard deviation is close to 0
    # (so practically when there's no activity in any time point anyway, this should not have practical relevance
    # as these voxels are background)
    tacs[np.where(np.isnan(tacs))] = -4
    dims = tacs.shape

    # Do the PCA
    if isinstance(index, float):  # if index is nan, do this
        data_flat = np.reshape(tacs, (dims[0] * dims[1] * dims[2], dims[3]))
        data_pca_flat = PCA(n_components=components).fit_transform(data_flat)
        data_pca = np.reshape(data_pca_flat, (dims[0], dims[1], dims[2], components))
    else:                         # if index is list or array, do this
        data_flat = np.array([tacs[i[0], i[1], i[2], :] for i in index])  # ok
        data_pca_flat = PCA(n_components=components).fit_transform(data_flat)

        data_pca = np.zeros(tuple((tacs.shape[0], tacs.shape[1], tacs.shape[2], components)))
        for ind in range(index.shape[0]):
            data_pca[index[ind, 0], index[ind, 1], index[ind, 2], :] = data_pca_flat[ind, :]  # ok

    return data_pca


#
def reduce_dimensions_ica(tacs, index=np.nan, components=30):

    # Set nan's to -4 as they appear when the standard deviation is close to 0
    # (so practically when there's no activity in any time point anyway, this should not have practical relevance
    # as these voxels are background)
    tacs[np.where(np.isnan(tacs))] = -4
    dims = tacs.shape

    # Do the Independent Component Analysis
    if isinstance(index, float):  # if index is nan, do this
        data_flat = np.reshape(tacs, (dims[0] * dims[1] * dims[2], dims[3]))
        data_ica_flat = FastICA(n_components=components).fit_transform(data_flat)
        data_ica = np.reshape(data_ica_flat, (dims[0], dims[1], dims[2], components))
    else:                         # if index is list or array, do this
        data_flat = np.array([tacs[i[0], i[1], i[2], :] for i in index])
        data_ica_flat = FastICA(n_components=components).fit_transform(data_flat)

        data_ica = np.zeros(tuple((tacs.shape[0], tacs.shape[1], tacs.shape[2], components)))
        for ind in range(index.shape[0]):
            data_ica[index[ind, 0], index[ind, 1], index[ind, 2], :] = data_ica_flat[ind, :]

    return data_ica


def reduce_dimensions_tsvd(tacs, index=np.nan, components=30):

    # Set nan's to -4 as they appear when the standard deviation is close to 0
    # (so practically when there's no activity in any time point anyway, this should not have practical relevance
    # as these voxels are background)
    tacs[np.where(np.isnan(tacs))] = -4
    dims = tacs.shape

    # Do the Factor Analysis
    # NOTE: i may need to extract loadings
    if isinstance(index, float):  # if index is nan, do this
        data_flat = np.reshape(tacs, (dims[0] * dims[1] * dims[2], dims[3]))
        data_svd_flat = TruncatedSVD(n_components=components, random_state=0).fit_transform(data_flat)
        data_svd = np.reshape(data_svd_flat, (dims[0], dims[1], dims[2], components))
    else:                         # if index is list or array, do this
        data_flat = np.array([tacs[i[0], i[1], i[2], :] for i in index])
        data_svd_flat = TruncatedSVD(n_components=components, random_state=0).fit_transform(data_flat)
        data_svd = np.zeros(tuple((tacs.shape[0], tacs.shape[1], tacs.shape[2], components)))
        for ind in range(index.shape[0]):
            data_svd[index[ind, 0], index[ind, 1], index[ind, 2], :] = data_svd_flat[ind, :]

    return data_svd


def reduce_dimensions_kpca(tacs, index=np.nan, components=30):

    # Set nan's to -4 as they appear when the standard deviation is close to 0
    # (so practically when there's no activity in any time point anyway, this should not have practical relevance
    # as these voxels are background)
    tacs[np.where(np.isnan(tacs))] = -4
    dims = tacs.shape

    # Do the PCA
    if isinstance(index, float):  # if index is nan, do this
        data_flat = np.reshape(tacs, (dims[0] * dims[1] * dims[2], dims[3]))
        kpca = KernelPCA(n_components=components, kernel="linear", eigen_solver="auto", random_state=1)
        data_kpca_flat = kpca.fit_transform(data_flat)
        data_kpca = np.reshape(data_kpca_flat, (dims[0], dims[1], dims[2], components))
    else:                         # if index is list or array, do this
        data_flat = np.array([tacs[i[0], i[1], i[2], :] for i in index])
        kpca = KernelPCA(n_components=components, kernel="linear", eigen_solver="auto", random_state=1)
        data_kpca_flat = kpca.fit_transform(data_flat)

        data_kpca = np.zeros(tuple((tacs.shape[0], tacs.shape[1], tacs.shape[2], components)))
        for ind in range(index.shape[0]):
            data_kpca[index[ind, 0], index[ind, 1], index[ind, 2], :] = data_kpca_flat[ind, :]

    return data_kpca


def reduce_dimensions_ppca(tacs, index=np.nan, components=30):

    # Set nan's to -4 as they appear when the standard deviation is close to 0
    # (so practically when there's no activity in any time point anyway, this should not have practical relevance
    # as these voxels are background)
    tacs[np.where(np.isnan(tacs))] = -4
    dims = tacs.shape

    # Do the PCA
    if isinstance(index, float):  # if index is nan, do this
        data_flat_full = np.reshape(tacs, (dims[0] * dims[1] * dims[2], dims[3]))
        ppca_full_flat = PCA(n_components=components).fit_transform(data_flat_full)
        data_ppca = np.reshape(ppca_full_flat, (dims[0], dims[1], dims[2], components))
    else:                         # if index is list or array, do this

        # Double the border layers in each axis (to later avoid mess with indices touching the border)
        tacs = np.concatenate((tacs[[0], :, :, :], tacs), axis=0)
        tacs = np.concatenate((tacs, tacs[[128], :, :, :]), axis=0)
        tacs = np.concatenate((tacs[:, [0], :, :], tacs), axis=1)
        tacs = np.concatenate((tacs, tacs[:, [128], :, :]), axis=1)
        tacs = np.concatenate((tacs[:, :, [0], :], tacs), axis=2)
        tacs = np.concatenate((tacs, tacs[:, :, [159], :]), axis=2)

        # Define indices for area 1 voxel wider than the original indices used for clustering
        mask_original = np.zeros((dims[0] + 2, dims[1] + 2, dims[2] + 2))
        for ind in index:
            mask_original[ind[0] + 1, ind[1] + 1, ind[2] + 1] = 1
        kernel = np.ones((3, 3, 3), dtype='uint8')
        mask_extended = binary_dilation(mask_original, kernel)
        index_extended = np.where(mask_extended == 1)

        # Do PCA for the extended mask area and convert to 4D
        data_flat_extended = np.array([tacs[index_extended[0][i], index_extended[1][i], index_extended[2][i], :] for i
                                       in range(len(index_extended[0]))])
        ppca_extended_flat = PCA(n_components=4).fit_transform(data_flat_extended)
        data_extended_ppca = np.zeros(tuple((dims[0] + 2, dims[1] + 2, dims[2] + 2, 4)))
        for ind in range(len(index_extended[0])):
            pca = ppca_extended_flat[ind, :]
            data_extended_ppca[index_extended[0][ind], index_extended[1][ind], index_extended[2][ind], :] = pca

        # Concatenate neighbourhood pca
        # NOTE: weird indexing from x to x+3 corresponds to (x-1):(x+2) in the original non-padded data
        data_ppca = np.zeros(tuple((dims[0], dims[1], dims[2], 4*27)))
        for ind in range(index.shape[0]):
            x, y, z = index[ind, 0], index[ind, 1], index[ind, 2]
            data_ppca[x, y, z, :] = np.reshape(data_extended_ppca[x:(x+3), y:(y+3), z:(z+3)], (1, 4*27))

    return data_ppca


# t-distributed Stochastic Neighbor Embedding (t-SNE)
def reduce_dimensions_tsne(tacs, index=np.nan, components=30):

    # Set nan's to -4 as they appear when the standard deviation is close to 0
    # (so practically when there's no activity in any time point anyway, this should not have practical relevance
    # as these voxels are background)
    tacs[np.where(np.isnan(tacs))] = -4
    dims = tacs.shape

    # Do the t-SNE
    if isinstance(index, float):  # if index is nan, do this
        data_flat = np.reshape(tacs, (dims[0] * dims[1] * dims[2], dims[3]))
        tsne = TSNE(n_components=components, random_state=1, learning_rate="auto", init="pca", method="exact")
        data_tsne_flat = tsne.fit_transform(data_flat)
        data_tsne = np.reshape(data_tsne_flat, (dims[0], dims[1], dims[2], components))
    else:                         # if index is list or array, do this
        data_flat = np.array([tacs[i[0], i[1], i[2], :] for i in index])
        tsne = TSNE(n_components=components, random_state=1, learning_rate="auto", init="pca", method="exact")
        data_tsne_flat = tsne.fit_transform(data_flat)
        data_tsne = np.zeros(tuple((tacs.shape[0], tacs.shape[1], tacs.shape[2], components)))
        for ind in range(index.shape[0]):
            data_tsne[index[ind, 0], index[ind, 1], index[ind, 2], :] = data_tsne_flat[ind, :]

    return data_tsne


# Multidimensional Scaling (MDS)
def reduce_dimensions_mds(tacs, index=np.nan, components=30):

    # Set nan's to -4 as they appear when the standard deviation is close to 0
    # (so practically when there's no activity in any time point anyway, this should not have practical relevance
    # as these voxels are background)
    tacs[np.where(np.isnan(tacs))] = -4
    dims = tacs.shape

    # Do the MDS
    if isinstance(index, float):  # if index is nan, do this
        data_flat = np.reshape(tacs, (dims[0] * dims[1] * dims[2], dims[3]))
        data_mds_flat = MDS(n_components=components, random_state=1, metric=True).fit_transform(data_flat)
        data_mds = np.reshape(data_mds_flat, (dims[0], dims[1], dims[2], components))
    else:                         # if index is list or array, do this
        data_flat = np.array([tacs[i[0], i[1], i[2], :] for i in index])
        data_mds_flat = MDS(n_components=components, random_state=1, metric=True).fit_transform(data_flat)
        data_mds = np.zeros(tuple((tacs.shape[0], tacs.shape[1], tacs.shape[2], components)))
        for ind in range(index.shape[0]):
            data_mds[index[ind, 0], index[ind, 1], index[ind, 2], :] = data_mds_flat[ind, :]

    return data_mds
    
    
#
def reduce_dimensions_fa(tacs, index=np.nan, components=30):

    # Set nan's to -4 as they appear when the standard deviation is close to 0
    # (so practically when there's no activity in any time point anyway, this should not have practical relevance
    # as these voxels are background)
    #tacs[np.where(np.isnan(tacs))] = -4
    dims = tacs.shape

    # Do the Factor Analysis
    # NOTE: i may need to extract loadings
    if isinstance(index, float):  # if index is nan, do this
        data_flat = np.reshape(tacs, (dims[0] * dims[1] * dims[2], dims[3]))
        fa = FactorAnalyzer(n_factors=components, svd_method="lapack", is_corr_matrix=False)
        data_fa_flat = fa.fit(data_flat)
        data_fa = np.reshape(data_fa_flat.loadings_, (dims[0], dims[1], dims[2], components))
    else:                         # if index is list or array, do this
        data_flat = np.array([tacs[i[0], i[1], i[2], :] for i in index])
        fa = FactorAnalyzer(n_factors=components, svd_method="lapack", is_corr_matrix=False)
        data_fa_flat = fa.fit(data_flat).loadings_

        data_fa = np.zeros(tuple((tacs.shape[0], tacs.shape[1], tacs.shape[2], components)))
        for ind in range(index.shape[0]):
            data_fa[index[ind, 0], index[ind, 1], index[ind, 2], :] = data_fa_flat[ind, :]

    return data_fa    


# Isometric mapping (Isomap)
def reduce_dimensions_isomap(tacs, index=np.nan, components=30):

    # Set nan's to -4 as they appear when the standard deviation is close to 0
    # (so practically when there's no activity in any time point anyway, this should not have practical relevance
    # as these voxels are background)
    tacs[np.where(np.isnan(tacs))] = -4
    dims = tacs.shape

    # Do the MDSmg,
    if isinstance(index, float):  # if index is nan, do this
        data_flat = np.reshape(tacs, (dims[0] * dims[1] * dims[2], dims[3]))
        data_isomap_flat = Isomap(n_components=components, n_neighbors=50).fit_transform(data_flat)
        data_isomap = np.reshape(data_isomap_flat, (dims[0], dims[1], dims[2], components))
    else:                         # if index is list or array, do this
        data_flat = np.array([tacs[i[0], i[1], i[2], :] for i in index])
        data_isomap_flat = Isomap(n_components=components, n_neighbors=50).fit_transform(data_flat)
        data_isomap = np.zeros(tuple((tacs.shape[0], tacs.shape[1], tacs.shape[2], components)))
        for ind in range(index.shape[0]):
            data_isomap[index[ind, 0], index[ind, 1], index[ind, 2], :] = data_isomap_flat[ind, :]

    return data_isomap
