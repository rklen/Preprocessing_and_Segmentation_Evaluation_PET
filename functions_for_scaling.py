import numpy as np
import scipy.stats as stats

# NOTE: foreground indices used in this study (argument 'index' is given instead of default nan)

#
def scale_tacs_zscore(tacs, index=np.nan):

    if isinstance(index, float):
        scaled_image = stats.zscore(tacs, axis=3)
    else:
        flat_tacs = np.array([tacs[i[0], i[1], i[2], :] for i in index])
        flat_tacs_scaled = stats.zscore(flat_tacs, axis=0)
        scaled_image = np.full(tacs.shape, np.nan)
        for ind in range(index.shape[0]):
            scaled_image[index[ind, 0], index[ind, 1], index[ind, 2], :] = flat_tacs_scaled[ind, :]

    return scaled_image


# Output: 4D array similar to 'tacs', but each time point is scaled with logistic function so that upper limit is 2,
#         median (unique) value get value 1 and the steepness of the logistic curve is defined so that 0.9 quantile
#         of unique values gets value 1.8 (0.9*2).
def scale_tacs_logistic(tacs, index=np.nan):

    # Define the key parameters
    max_val = 2
    scaled_image = np.full(tacs.shape, np.nan)

    # Do the logistic scaling for each time point separately
    if isinstance(index, float):
        for i in range(tacs.shape[3]):
            time_data = tacs[:, :, :, i]
            unique_vals = np.unique(time_data)
            value_90 = np.quantile(unique_vals, 0.9)
            value_50 = np.median(unique_vals)
            steepness = -1 * (np.log(2 / 1.8 - 1) / (value_90 - value_50))
            scaled_time = max_val / (1 + np.exp(-1 * steepness * (time_data - value_50)))
            scaled_image[:, :, :, i] = scaled_time
    else:
        flat_tacs = np.array([tacs[i[0], i[1], i[2], :] for i in index])
        flat_tacs_scaled = np.full(flat_tacs.shape, np.nan)
        for i in range(tacs.shape[3]):
            time_data = flat_tacs[:, i]
            unique_vals = np.unique(time_data)
            value_90 = np.quantile(unique_vals, 0.9)
            value_50 = np.median(unique_vals)
            steepness = -1 * (np.log(2 / 1.8 - 1) / (value_90 - value_50))
            scaled_time = max_val / (1 + np.exp(-1 * steepness * (time_data - value_50)))
            flat_tacs_scaled[:, i] = scaled_time
        for ind in range(index.shape[0]):
            scaled_image[index[ind, 0], index[ind, 1], index[ind, 2], :] = flat_tacs_scaled[ind, :]

    return scaled_image


# Input: 'tacs' ia a 4D array of voxel activities over time (time has to be the last dimension)
# Output:
def scale_tacs_sum1(tacs):

    # Initialise denoised image array
    dims = tacs.shape
    scaled_image = np.empty(dims)

    # Calculate the scaling factor for each voxel
    voxel_sums = np.sum(tacs, axis=3)

    # Do the scaling
    for r in range(dims[0]):
        for c in range(dims[1]):
            for d in range(dims[2]):
                voxel_sum = voxel_sums[r, c, d]
                if voxel_sum > 0:
                    for t in range(dims[3]):
                        scaled_image[r, c, d, t] = tacs[r, c, d, t] / voxel_sum
                else:
                    for t in range(dims[3]):
                        scaled_image[r, c, d, t] = 0

    return scaled_image
    