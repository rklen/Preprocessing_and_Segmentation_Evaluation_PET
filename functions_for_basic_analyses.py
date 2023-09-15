import numpy as np
import nibabel as nib
import os
from os.path import exists


# Input: 'clusters' is a 3D array of cluster labels
#        'organs' is a list of organ specific 3D arrays (1 in voxels belonging to the organ and 0 elsewhere)
# Output: Returns two elements: list of best Jaccard indices for the organs and list (organs) of lists of cluster labels
#         providing the corresponding Jaccard. Notably, for each organ, there is a list of labels because the organ may
#         consist of several clusters (usually just one, but it is possible anyway)
def calculate_jaccard(clusters, organs):

    jaccards = []
    organ_labels = []
    for i in range(len(organs)):
        if isinstance(organs[i], float):  # numpy.nan is a float, arrays are not regardless of their content
            jaccards.append(np.nan)
            organ_labels.append([np.nan])
        else:
            organ_index = np.where(organs[i] > 0.1)
            mask_manual = np.zeros(organs[i].shape, dtype=np.int32)
            mask_manual[organ_index] = np.int32(1)
            present_labels = clusters[organ_index]
            potential_labels = np.unique(present_labels)

            # Calculate Jaccards for all labels overlapping with the organ
            organ_jaccards = []
            for j in potential_labels:
                label_index = np.where(clusters == j)
                mask_label = np.zeros(clusters.shape, dtype=np.int32)
                mask_label[label_index] = np.int32(1)
                mask_sum = mask_manual + mask_label
                jaccard = len(np.where(mask_sum == 2)[0]) / len(np.where(mask_sum > 0.1)[0])
                organ_jaccards.append(jaccard)

            # Sort jaccards and cluster labels they are associated with
            sorted_label_indices = np.argsort(-np.array(organ_jaccards))
            organ_jaccards = [organ_jaccards[n] for n in sorted_label_indices]
            potential_labels = [potential_labels[r] for r in sorted_label_indices]

            # Pick the set of clusters (usually just one) with maximum jaccard
            max_jaccard = organ_jaccards[0]
            max_label = [potential_labels[0]]
            mask_max_labels = np.zeros(clusters.shape, dtype=np.int32)
            mask_max_labels[np.where(clusters == potential_labels[0])] = np.int32(1)
            if len(potential_labels) > 1:
                k = 1
                cont = True
                while cont:
                    mask_max_labels[np.where(clusters == potential_labels[k])] = np.int32(1)
                    mask_sum_set = mask_max_labels + mask_manual
                    set_jaccard = len(np.where(mask_sum_set == 2)[0]) / len(np.where(mask_sum_set > 0.1)[0])
                    if set_jaccard > max_jaccard:
                        max_jaccard = set_jaccard
                        max_label.append(potential_labels[k])
                    else:
                        cont = False
                    k = k + 1
                    if k >= len(potential_labels):
                        cont = False
            jaccards.append(max_jaccard)
            organ_labels.append(max_label)
    return jaccards, organ_labels


# Input: 'image_ids' is a list of images, whose manual segmentations to load (e.g. ['P120997', 'P120996', 'P120999']),
#        'organs' is a list of organs, whose manual segmentations to read (written as in corresponding file names),
#        'tracer' is one of these: "FDPA", "UCB, or "FDG"
# Output: Returns a list (images) of lists (organs) of 3D arrays including the manual segmentation
# NOTE: All image ids should be from the same tracer and all given organs should be segmented for all given images
def read_manual_segmentations(image_ids, organs, tracer):

    # Set the path to manual segmentations based on tracer
    path_prefix = "path/to/manual/segmentations/"

    # Read in the manual segments for each organ one image at time
    manual_segments = []
    for p in range(len(image_ids)):
        image_segments = []
        for o in organs:
            file_name = path_prefix + image_ids[p] + "/" + o + ".img"  # Modify this to match your file naming protocol

            if exists(file_name):
                organ_nifti = nib.load(file_name).get_fdata()
                organ_nifti = np.transpose(np.flip(organ_nifti, 2), (1, 0, 2))
            else:
                organ_nifti = np.nan

            image_segments.append(organ_nifti)
        manual_segments.append(image_segments)

    return manual_segments


# Input: 'image_ids' is a list of images, whose manual segmentations to load (e.g. ['P120997', 'P120996', 'P120999']),
#        'path' indicates where to find the clustering results
#        'tags' is a list of identification terms for the files to read in case 'path' contains different clusterings
#        'tracer' is one of these: "FDPA", "UCB", or "FDG"
# Output: Returns a list of 3D arrays including the cluster labels. If the clustering is not available for some of the
#         given images, the list includes numpy nan in those places.
# NOTE: All image ids should be from the same tracer and the clustering results
def read_in_3d_clusters(image_ids, path, tags, tracer):

    path_index = "path/to/foreground/indices/" + tracer + "/"  # Modify this to match your file naming protocol

    clusters = []
    for i in range(len(image_ids)):

        # Initialise the 3D array
        labels = np.zeros((128, 128, 159))

        # Read in voxel indices
        file_index = path_index + image_ids[i] + "_index.npy"
        index = np.load(file_index)

        # Read in clustering results
        files = os.listdir(path)
        for t in tags:
            files = [accepted for accepted in files if t in accepted]
        files = [accepted for accepted in files if image_ids[i] in accepted]  # Use image id as additional tag
        if len(files) > 1:
            raise SyntaxError('File is not unique, define more tags for image ' + image_ids[i])
        if len(files) == 0:
            clusters.append(np.nan)
        else:
            file_results = path + "/" + files[0]
            clustering_results = np.load(file_results)

            # Fill in the clustering results
            for ind in range(len(index)):
                labels[tuple(index[ind])] = clustering_results[ind] + 1

            clusters.append(labels)

    return clusters
