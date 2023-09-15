import numpy as np
from scipy.ndimage import gaussian_filter
from Codes import functions_for_scaling as sf
from Codes import functions_for_dimensionality_reduction as dr
from Codes import segmentation_methods as sm

images_fdpa = ['P116012', 'P116014', 'P116019', 'P116020', 'P120196', 'P120197', 'P120198', 'P120199', 'P120992',
               'P120993', 'P120994', 'P120995', 'P120998', 'P121000', 'P121001', 'P121002', 'P121009', 'P121010',
               'P121011', 'P121012', 'P122904', 'P123182', 'P123183', 'P123185', 'P123186', 'P124016']
images_ucb = ['P117882', 'P117885', 'P117912', 'P117915', 'P117918', 'P117929', 'P117932', 'P119993', 'P120779',
              'P120782', 'P120783', 'P120795', 'P120809', 'P120813', 'P120814', 'P121118', 'P121867', 'P122817',
              'P122824', 'P122846', 'P122867', 'P122873', 'P122914', 'P122942', 'P123929', 'P123968', 'P124005',
              'P124035', 'P124999', 'P125000', 'P126197', 'P126201']
images_fdg = ['P117094', 'P117095', 'P117594', 'P117596', 'P117597', 'P117600', 'P117602', 'P117606', 'P117612',
              'P117615', 'P118121', 'P118132', 'P118271', 'P118848', 'P118849', 'P118858', 'P121613', 'P121614',
              'P121620', 'P121621', 'P121627', 'P122810', 'P122815', 'P122818', 'P122897', 'P122947', 'P123963',
              'P124010', 'P125111', 'P125113', 'P125114', 'P125915', 'P126169']

# Define paths               
path_prefix_raw_data = "path/to/load/raw/images/" 
path_prefix_indices = "path/to/load/foreground/indices/" 
path_prefix_preprocessing = "path/to/save/or/load/preprocessed/images/"        
path_prefix_results = "path/to/save/segmentation/results"      

"""
Preprocess validation images with gaussian filtering -> z-score -> 5 pca
"""

# Indices are arrays with 3 columns (x, y, and z coordinate) and rows corresponding to foreground voxels. 
# TACs are 4D arrays including the radioactivity levels of the image (last dim is time).

# FDPA
for id_fdpa in images_fdpa:
    tacs_raw_fdpa = np.load(path_prefix_raw_data + "FDPA/" + id_fdpa + "_TACs_raw.npy")  # Modify to match your file names
    index_fdpa = np.load(path_prefix_indices + "FDPA/" + id_fdpa + "_index.npy")  # Modify to match your file names
    tacs_denoised_fdpa = gaussian_filter(tacs_raw_fdpa, sigma=1)
    tacs_scaled_fdpa = sf.scale_tacs_zscore(tacs_denoised_fdpa, index_fdpa)
    tacs_reduced_fdpa = dr.reduce_dimensions_pca(tacs_scaled_fdpa, index_fdpa, 5)
    file_name = path_prefix_preprocessing + "FDPA/" + id_fdpa + "_tacs_preprocessed.npy"
    np.save(file_name, tacs_reduced_fdpa)

# UCB
for id_ucb in images_ucb:
    tacs_raw_ucb = np.load(path_prefix_raw_data + "UCB/" + id_ucb + "_TACs_raw.npy")  # Modify to match your file names
    index_ucb = np.load(path_prefix_indices + "UCB/" + id_ucb + "_index.npy")  # Modify to match your file names
    tacs_denoised_ucb = gaussian_filter(tacs_raw_ucb, sigma=1)
    tacs_scaled_ucb = sf.scale_tacs_zscore(tacs_denoised_ucb, index_ucb)
    tacs_reduced_ucb = dr.reduce_dimensions_pca(tacs_scaled_ucb, index_ucb, 5)
    file_name = path_prefix_preprocessing + "UCB/" + id_ucb + "_tacs_preprocessed.npy"
    np.save(file_name, tacs_reduced_ucb)

# FDG
for id_fdg in images_fdg:
    tacs_raw_fdg = np.load(path_prefix_raw_data + "FDG/" + id_fdg + "_TACs_raw.npy")  # Modify to match your file names
    index_fdg = np.load(path_prefix_indices + "FDG/" + id_fdg + "_index.npy")  # Modify to match your file names
    tacs_denoised_fdg = gaussian_filter(tacs_raw_fdg, sigma=1)
    tacs_scaled_fdg = sf.scale_tacs_zscore(tacs_denoised_fdg, index_fdg)
    tacs_reduced_fdg = dr.reduce_dimensions_pca(tacs_scaled_fdg, index_fdg, 5)
    file_name = path_prefix_preprocessing + "FDG/" + id_fdg + "_tacs_preprocessed.npy"
    np.save(file_name, tacs_reduced_fdg)

"""
Cluster preprocessed validation images
"""

n_gmm, n_kmeans, n_mbkmeans, n_fcmeans = 28, 26, 27, 25

# FDPA
for id_fdpa in images_fdpa:
    tacs_fdpa = np.load(path_prefix_preprocessing + "FDPA/" + id_fdpa + "_tacs_preprocessed.npy")
    indices_fdpa = np.load(path_prefix_indices + "FDPA/" + id_fdpa + "_index.npy")
    path_fdpa = path_prefix_results + "FDPA"
    file_suffix_fdpa = "_" + id_fdpa + "_validation.npy"
    label_fdpa, time_fdpa, methods_fdpa = sm.segment_image(image=tacs_fdpa, indices=indices_fdpa, res_path=path_fdpa,
                                                           file_suffix=file_suffix_fdpa, params_hierarchical=None,
                                                           params_gmm=[n_gmm], params_kmeans=[n_kmeans],
                                                           params_mbkmeans=[n_mbkmeans], params_fcmeans=[n_fcmeans],
                                                           params_hdbscan=None, params_slic=None, params_morphGAC=None,
                                                           params_morphACWE=None)

# UCB
for id_ucb in images_ucb:
    tacs_ucb = np.load(path_prefix_preprocessing + "UCB/" + id_ucb + "_tacs_preprocessed.npy")
    indices_ucb = np.load(path_prefix_indices + "UCB/" + id_ucb + "_index.npy")
    path_ucb = path_prefix_results + "UCB"
    file_suffix_ucb = "_" + id_ucb + "_validation.npy"
    label_ucb, time_ucb, methods_ucb = sm.segment_image(image=tacs_ucb, indices=indices_ucb, res_path=path_ucb,
                                                        file_suffix=file_suffix_ucb, params_hierarchical=None,
                                                        params_gmm=[n_gmm], params_kmeans=[n_kmeans],
                                                        params_mbkmeans=[n_mbkmeans], params_fcmeans=[n_fcmeans],
                                                        params_hdbscan=None, params_slic=None, params_morphGAC=None,
                                                        params_morphACWE=None)

# FDG
for id_fdg in images_fdg:
    tacs_fdg = np.load(path_prefix_preprocessing + "FDG/" + id_fdg + "_tacs_preprocessed.npy")
    indices_fdg = np.load(path_prefix_indices + "FDG/" + id_fdg + "_index.npy")
    path_fdg = path_prefix_results + "FDG"
    file_suffix_fdg = "_" + id_fdg + "_validation.npy"
    label_fdg, time_fdg, methods_fdg = sm.segment_image(image=tacs_fdg, indices=indices_fdg, res_path=path_fdg,
                                                        file_suffix=file_suffix_fdg, params_hierarchical=None,
                                                        params_gmm=[n_gmm], params_kmeans=[n_kmeans],
                                                        params_mbkmeans=[n_mbkmeans], params_fcmeans=[n_fcmeans],
                                                        params_hdbscan=None, params_slic=None, params_morphGAC=None,
                                                        params_morphACWE=None)

"""
Calculate Jaccard indices
"""
from Codes import functions_for_basic_analyses as funcs
import numpy as np
import pickle

# Read in clusters
methods = ["gmm", "kmeans", "mbkmeans", "fcmeans"]
clusters_fdpa = []
clusters_ucb = []
clusters_fdg = []
for m in methods:
    clust_fdpa = funcs.read_in_3d_clusters(images_fdpa, path_prefix_results + "FDPA", ["_" + m], "FDPA")
    clust_ucb = funcs.read_in_3d_clusters(images_ucb, path_prefix_results + "UCB", ["_" + m], "UCB")
    clust_fdg = funcs.read_in_3d_clusters(images_fdg, path_prefix_results + "FDG", ["_" + m], "FDG")
    clusters_fdpa.append(clust_fdpa)
    clusters_ucb.append(clust_ucb)
    clusters_fdg.append(clust_fdg)

# Read in manual segmentations
organs_fdpa = ["Brain", "Heart", "Lungs", "Pituitary", "Thyroids"]
organs_ucb = ["Brain", "Kidneys", "Liver"]
organs_fdg = ["brain", "heart", "kidney"]
manual_fdpa = funcs.read_manual_segmentations(images_fdpa, organs_fdpa, "FDPA")
manual_ucb = funcs.read_manual_segmentations(images_ucb, organs_ucb, "UCB")
manual_fdg = funcs.read_manual_segmentations(images_fdg, organs_fdg, "FDG")

# Calculate Jaccards
jaccards_fdpa = np.full((len(images_fdpa), len(methods), len(organs_fdpa)), np.nan)  # dims: images * methods * organs
jaccards_ucb = np.full((len(images_ucb), len(methods), len(organs_ucb)), np.nan)
jaccards_fdg = np.full((len(images_fdg), len(methods), len(organs_fdg)), np.nan)
labels_fdpa, labels_ucb, labels_fdg = [], [], []
for m in range(len(methods)):
    method_labels_fdpa, method_labels_ucb, method_labels_fdg = [], [], []
    for fdpa in range(len(images_fdpa)):
        if not isinstance(clusters_fdpa[m][fdpa], float):
            jaccard_fdpa, cluster_label = funcs.calculate_jaccard(clusters_fdpa[m][fdpa], manual_fdpa[fdpa])
            jaccards_fdpa[fdpa, m, :] = jaccard_fdpa
            method_labels_fdpa.append(cluster_label)
        else:
            method_labels_fdpa.append([-1])
    labels_fdpa.append(method_labels_fdpa)
    for ucb in range(len(images_ucb)):
        if not isinstance(clusters_ucb[m][ucb], float):
            jaccard_ucb, cluster_label = funcs.calculate_jaccard(clusters_ucb[m][ucb], manual_ucb[ucb])
            jaccards_ucb[ucb, m, :] = jaccard_ucb
            method_labels_ucb.append(cluster_label)
        else:
            method_labels_ucb.append([-1])
    labels_ucb.append(method_labels_ucb)
    for fdg in range(len(images_fdg)):
        if not isinstance(clusters_fdg[m][fdg], float):
            jaccard_fdg, cluster_label = funcs.calculate_jaccard(clusters_fdg[m][fdg], manual_fdg[fdg])
            jaccards_fdg[fdg, m, :] = jaccard_fdg
            method_labels_fdg.append(cluster_label)
        else:
            method_labels_fdg.append([-1])
    labels_fdg.append(method_labels_fdg)

# Save Jaccard indices and VOI cluster labels
np.save("path/where/to/save/the/results/AllJaccards_FDPA_validation.npy", jaccards_fdpa)
np.save("path/where/to/save/the/results/AllJaccards_UCB_validation.npy", jaccards_ucb)
np.save("path/where/to/save/the/results/AllJaccards_FDG_validation.npy", jaccards_fdg)  # images*methods*VOIs

pickle.dump(labels_fdpa, open("path/where/to/save/the/results/VOIlabels_FDPA_validation.txt", 'wb'))
pickle.dump(labels_ucb, open("path/where/to/save/the/results/VOIlabels_UCB_validation.npy", 'wb'))
pickle.dump(labels_fdg, open("path/where/to/save/the/results/VOIlabels_FDG_validation.npy", 'wb'))
# labels: FDPA=[GMM, k-means, ...], where GMM=[img1, img2, ...], where img1=[brain, heart, ...], where brain=[labels]
