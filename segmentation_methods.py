import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import AffinityPropagation
import fastcluster as fc
from sklearn.mixture import GaussianMixture
from fcmeans import FCM
import hdbscan
from scipy.cluster.hierarchy import fcluster
from time import perf_counter
from skimage.segmentation import slic
from skimage.segmentation import watershed
from skimage.segmentation import random_walker
from skimage.segmentation import morphological_geodesic_active_contour
from skimage.segmentation import morphological_chan_vese
from skimage.segmentation import inverse_gaussian_gradient


# Main function
def segment_image(image, indices, res_path, file_suffix, params_hierarchical=None, params_gmm=None, params_kmeans=None,
                  params_mbkmeans=None, params_fcmeans=None, params_hdbscan=None,  params_spectral=None,
                  params_meanshift=None, params_dbscan=None, params_birch=None, params_slic=None, params_morphGAC=None,
                  params_morphACWE=None, params_rwalker=None, params_watershed=None):

    # Extract only voxels to be clustered
    image_3d = np.sum(image, axis=3)
    tacs = np.array([image[i[0], i[1], i[2], :] for i in indices])
    file_prefix = res_path + "/Labels_"

    label_list = []  # stores the label arrays from each clustering method
    processing_time = []  # stores the processing times for each clustering
    used_methods = []

    if params_hierarchical is not None:

        labels_hierarchical, time_hierarchical = clustering_hierarchical(TACs=tacs,
                                                                         cluster_number=params_hierarchical[0])
        label_list.append(labels_hierarchical)
        processing_time.append(time_hierarchical)
        used_methods.append("hierarchical")

        file_hierarchical = file_prefix + "hierarchical" + file_suffix
        np.save(file_hierarchical, labels_hierarchical)

    if params_gmm is not None:

        labels_gmm, time_gmm = clustering_gmm(TACs=tacs, cluster_number=params_gmm[0])
        label_list.append(labels_gmm)
        processing_time.append(time_gmm)
        used_methods.append("gmm")

        if not isinstance(labels_gmm, float):
            file_gmm = file_prefix + "gmm" + file_suffix
            np.save(file_gmm, labels_gmm)

    if params_kmeans is not None:

        labels_kmeans, time_kmeans = clustering_kmeans(TACs=tacs, cluster_number=params_kmeans[0])
        label_list.append(labels_kmeans)
        processing_time.append(time_kmeans)
        used_methods.append("kmeans")

        file_kmeans = file_prefix + "kmeans" + file_suffix
        np.save(file_kmeans, labels_kmeans)

    if params_mbkmeans is not None:

        labels_mbkmeans, time_mbkmeans = clustering_MBkmeans(TACs=tacs, cluster_number=params_mbkmeans[0])
        label_list.append(labels_mbkmeans)
        processing_time.append(time_mbkmeans)
        used_methods.append("mbkmeans")

        file_mbkmeans = file_prefix + "mbkmeans" + file_suffix
        np.save(file_mbkmeans, labels_mbkmeans)

    if params_fcmeans is not None:

        labels_fcmeans, time_fcmeans = clustering_fcmeans(TACs=tacs, cluster_number=params_fcmeans[0])
        label_list.append(labels_fcmeans)
        processing_time.append(time_fcmeans)
        used_methods.append("fcmeans")

        file_fcmeans = file_prefix + "fcmeans" + file_suffix
        np.save(file_fcmeans, labels_fcmeans)

    if params_hdbscan is not None:

        labels_hdbscan, time_hdbscan = clustering_hdbscan(TACs=tacs)
        label_list.append(labels_hdbscan)
        processing_time.append(time_hdbscan)
        used_methods.append("hdbscan")

        file_hdbscan = file_prefix + "hdbscan" + file_suffix
        np.save(file_hdbscan, labels_hdbscan)

    if params_spectral is not None:

        labels_spectral, time_spectral = clustering_spectral(TACs=tacs, cluster_number=params_spectral[0])
        label_list.append(labels_spectral)
        processing_time.append(time_spectral)
        used_methods.append("spectral")

        file_spectral = file_prefix + "spectral" + file_suffix
        np.save(file_spectral, labels_spectral)

    if params_meanshift is not None:

        labels_meanshift, time_meanshift = clustering_meanshift(TACs=tacs)
        label_list.append(labels_meanshift)
        processing_time.append(time_meanshift)
        used_methods.append("meanshift")

        file_meanshift = file_prefix + "meanshift" + file_suffix
        np.save(file_meanshift, labels_meanshift)

    if params_dbscan is not None:

        labels_dbscan, time_dbscan = clustering_dbscan(TACs=tacs, min_size=params_dbscan[0], epsilon=params_dbscan[1])
        label_list.append(labels_dbscan)
        processing_time.append(time_dbscan)
        used_methods.append("dbscan")

        file_dbscan = file_prefix + "dbscan" + file_suffix
        np.save(file_dbscan, labels_dbscan)

    if params_birch is not None:

        labels_birch, time_birch = clustering_birch(TACs=tacs, cluster_number=params_birch[0])
        label_list.append(labels_birch)
        processing_time.append(time_birch)
        used_methods.append("birch")

        file_birch = file_prefix + "birch" + file_suffix
        np.save(file_birch, labels_birch)

    if params_slic is not None:

        labels_slic, time_slic = segmentation_slic(data=image_3d, indices=indices, segment_number=params_slic[0],
                                                   compact_level=params_slic[1], sigma_level=params_slic[2])
        label_list.append(labels_slic)
        processing_time.append(time_slic)
        used_methods.append("slic")

        file_slic = file_prefix + "slic" + file_suffix
        np.save(file_slic, labels_slic)

    if params_watershed is not None:

        labels_watershed, time_watershed = segmentation_watershed(data=image_3d, indices=indices)
        label_list.append(labels_watershed)
        processing_time.append(time_watershed)
        used_methods.append("watershed")

        file_watershed = file_prefix + "watershed" + file_suffix
        np.save(file_watershed, labels_watershed)

    if params_rwalker is not None:

        labels_rwalker, time_rwalker = segmentation_rwalker(data=image_3d, indices=indices, phases=params_rwalker[0],
                                                            beta_val=params_rwalker[1])
        label_list.append(labels_rwalker)
        processing_time.append(time_rwalker)
        used_methods.append("rwalker")

        file_rwalker = file_prefix + "rwalker" + file_suffix
        np.save(file_rwalker, labels_rwalker)

    if params_morphGAC is not None:

        labels_morphGAC, time_morphGAC = segmentation_morphGAC(data=image, indices=indices,
                                                               expand_strength=params_morphGAC[0],
                                                               iterations=params_morphGAC[1],
                                                               sigma_level=params_morphGAC[2])
        label_list.append(labels_morphGAC)
        processing_time.append(time_morphGAC)
        used_methods.append("morphGAC")

        file_morphGAC = file_prefix + "morphGAC" + file_suffix
        np.save(file_morphGAC, labels_morphGAC)

    if params_morphACWE is not None:

        labels_morphACWE, time_morphACWE = segmentation_morphACWE(data=image, indices=indices,
                                                                  iterations=params_morphACWE[0])
        label_list.append(labels_morphACWE)
        processing_time.append(time_morphACWE)
        used_methods.append("morphACWE")

        file_morphACWE = file_prefix + "morphACWE" + file_suffix
        np.save(file_morphACWE, labels_morphACWE)

    return label_list, processing_time, used_methods


#
#
#
def clustering_hierarchical(TACs, cluster_number):

    time_start = perf_counter()
    dendrogram = fc.linkage_vector(TACs)
    cluster_labels = fcluster(dendrogram, cluster_number, criterion='maxclust')
    time_end = perf_counter()

    processing_time = round(time_end - time_start, 0)
    print("hierarchical clustering took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time



#
#
#
def clustering_gmm(TACs, cluster_number):

    time_start = perf_counter()
    attempt = 0
    while attempt < 10:
        try:
            gmm_model = GaussianMixture(n_components=cluster_number).fit(TACs)  # Gaussian mixture model clustering
            cluster_labels = gmm_model.predict(TACs)  # extract labels from the model
            attempt = 15
        except:
            print('GMM was not successful')
            attempt = attempt + 1
    time_end = perf_counter()

    # Return nan if GMM didn't succeed in 10 tries
    if attempt == 10:
        cluster_labels = np.nan

    processing_time = round(time_end - time_start, 0)
    print("GMM took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time



#
#
#
def clustering_kmeans(TACs, cluster_number):

    time_start = perf_counter()
    cluster_labels = KMeans(n_clusters=cluster_number).fit_predict(TACs)
    time_end = perf_counter()

    processing_time = round(time_end - time_start, 0)
    print("k-means took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time



#
#
#
def clustering_optics(TACs, min_size=10, eps=0.8):

    time_start = perf_counter()
    cluster_labels = OPTICS(eps=eps, min_samples=min_size).fit_predict(TACs)
    time_end = perf_counter()

    processing_time = round(time_end - time_start, 0)
    print("OPTICS took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time



#
#
#
def clustering_spectral(TACs, cluster_number):

    time_start = perf_counter()
    cluster_labels = SpectralClustering(n_clusters=cluster_number).fit_predict(TACs)
    time_end = perf_counter()

    processing_time = round(time_end - time_start, 0)
    print("spectral clustering took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time



#
#
#
def clustering_meanshift(TACs):

    time_start = perf_counter()
    cluster_labels = MeanShift().fit_predict(TACs)
    time_end = perf_counter()

    processing_time = round(time_end - time_start, 0)
    print("mean shift clustering took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time



#
#
#
def clustering_MBkmeans(TACs, cluster_number):

    time_start = perf_counter()
    cluster_labels = MiniBatchKMeans(n_clusters=cluster_number).fit_predict(TACs)
    time_end = perf_counter()

    processing_time = round(time_end - time_start, 0)
    print("mini batch k-means took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time



# epsilon is important: too small -> plenty of non-clustered samples (-1), too big -> clusters merge together
#
#
def clustering_dbscan(TACs, min_size=5, epsilon=1.1):

    time_start = perf_counter()
    cluster_labels = DBSCAN(eps=epsilon, min_samples=min_size).fit_predict(TACs)
    time_end = perf_counter()

    processing_time = round(time_end - time_start, 0)
    print("DBSCAN took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time



#
#
#
def clustering_birch(TACs, cluster_number, cutoff=0.01):

    time_start = perf_counter()
    cluster_labels = Birch(threshold=cutoff, n_clusters=cluster_number).fit_predict(TACs)
    time_end = perf_counter()

    processing_time = round(time_end - time_start, 0)
    print("BIRCH took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time



#
#
#
def clustering_affinity(TACs, damp=0.9):

    time_start = perf_counter()
    cluster_labels = AffinityPropagation(damping=damp).fit_predict(TACs)
    time_end = perf_counter()

    processing_time = round(time_end - time_start, 0)
    print("affinity propagation took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time



#
#
#
def clustering_fcmeans(TACs, cluster_number):

    time_start = perf_counter()
    fcm = FCM(n_clusters=cluster_number)
    fcm.fit(TACs)
    cluster_labels = fcm.predict(TACs)
    time_end = perf_counter()

    processing_time = round(time_end - time_start, 0)
    print("fuzzy c-means took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time



#
#
#
def clustering_hdbscan(TACs):

    time_start = perf_counter()
    clustering = hdbscan.HDBSCAN().fit(TACs)
    cluster_labels = clustering.labels_
    time_end = perf_counter()

    processing_time = round(time_end - time_start, 0)
    print("HDBSCAN took " + str(processing_time) + " seconds" + "\n")

    return cluster_labels, processing_time


######## Other segmentation approaches


#
#
#
# NOTE: slic argument 'min_size_factor' is set to 0 because some of the VOIs may be VERY small and argument
#       'max_size_factor' to the number of segments (corresponds the whole image being one segment) as the background
#       covers most of the voxels.
def segmentation_slic(data, indices, segment_number, compact_level=10, sigma_level=0):

    # Do the slic segmentation for each time frame separately
    time_start = perf_counter()
    segments = slic(data, n_segments=segment_number, compactness=compact_level, sigma=sigma_level, convert2lab=False,
                    enforce_connectivity=True, min_size_factor=0, max_size_factor=segment_number, channel_axis=None)

    # Format the output similarly to clustering methods
    time_end = perf_counter()
    segment_labels = np.array([segments[index[0], index[1], index[2]] for index in indices])

    segmentation_time = round(time_end - time_start, 0)
    print("slic took " + str(segmentation_time) + " seconds" + "\n")

    return segment_labels, segmentation_time


#
#
#
#
def segmentation_watershed(data, indices):

    # Do the watershed segmentation for each time frame separately
    time_start = perf_counter()
    segments = watershed(data)

    # Format the output similarly to clustering methods
    time_end = perf_counter()
    segment_labels = np.array([segments[index[0], index[1], index[2]] for index in indices])

    segmentation_time = round(time_end - time_start, 0)
    print("watershed took " + str(segmentation_time) + " seconds" + "\n")

    return segment_labels, segmentation_time


#
#
#
def segmentation_rwalker(data, indices, phases=10, beta_val=130):

    # Set initial labels (0 = no label, -1 = skip this voxel, 1,2,... = new phase)
    # All segmented voxels are sorted (according to voxels' sum activity) and divided into equally sized batches.
    # Median of each bach is set as the new phase (phases somewhat associate with segments)
    initial_labels = np.full((data.shape[0], data.shape[1], data.shape[2]), -1)
    sum_data_values = []
    for k in indices:
        initial_labels[tuple(k)] = 0
        sum_data_values.append(data[tuple(k)])
    sorted_index = np.argsort(sum_data_values)
    sequence_len = len(indices) / phases
    for j in range(phases):  # Extract median of each batch and set corresponding voxel to the next integer label
        pick_index = int(np.round(sequence_len*j + sequence_len/2))
        use_index = indices[sorted_index[pick_index]]
        initial_labels[tuple(use_index)] = j + 1

    # Do the random walker segmentation for each time frame separately
    time_start = perf_counter()
    segments = random_walker(data=data, labels=initial_labels, beta=beta_val, mode='cg_mg', tol=0.001, copy=False,
                             return_full_prob=False, spacing=None, prob_tol=0.001, channel_axis=None)

    # Format the output similarly to clustering methods
    time_end = perf_counter()
    segment_labels = np.array([segments[index[0], index[1], index[2]] for index in indices])

    segmentation_time = round(time_end - time_start, 0)
    print("random walker took " + str(segmentation_time) + " seconds" + "\n")

    return segment_labels, segmentation_time


#
#
#
# NOTE: This is said to work on heavily preprocessed images, not intended for raw ones.
def segmentation_morphGAC(data, indices, expand_strength=0, iterations=50, sigma_level=5):

    # Do the contouring segmentation for each time frame separately
    time_start = perf_counter()
    segments = []
    for i in range(data.shape[3]):
        gimage = inverse_gaussian_gradient(image=data[:, :, :, i], alpha=100.0, sigma=sigma_level)
        res = morphological_geodesic_active_contour(gimage=gimage, num_iter=iterations, init_level_set='disk',
                                                    smoothing=1, threshold='auto', balloon=expand_strength)
        segments.append(res)

    # Combine segmentations in different time frames
    segments_final = combine_time_frames(segments)
    time_end = perf_counter()
    segment_labels = np.array([segments_final[index[0], index[1], index[2]] for index in indices])

    segmentation_time = round(time_end - time_start, 0)
    print("morphGAC took " + str(segmentation_time) + " seconds" + "\n")

    return segment_labels, segmentation_time

#
#
#
# NOTE: Morphological Active Contours without Edges (MorphACWE)
def segmentation_morphACWE(data, indices, iterations):

    # Do the contouring segmentation for each time frame separately
    time_start = perf_counter()
    segments = []
    for i in range(data.shape[3]):
        res = morphological_chan_vese(image=data[:, :, :, i], num_iter=iterations, init_level_set='checkerboard',
                                      smoothing=1, lambda1=1, lambda2=1)
        segments.append(res)

    # Combine segmentations in different time frames
    segments_final = combine_time_frames(segments)
    time_end = perf_counter()
    segment_labels = np.array([segments_final[index[0], index[1], index[2]] for index in indices])

    segmentation_time = round(time_end - time_start, 0)
    print("morphACWE took " + str(segmentation_time) + " seconds" + "\n")

    return segment_labels, segmentation_time
