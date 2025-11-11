
# -*- coding: utf-8 -*-


import sys
from cmbfscnn.utils import load_pkl
from cmbfscnn.CMBFS import CMBFSCNN
import numpy as np
import os

# Firstly, simulate the data of the sky signal.
# Set random configuration files for foreground and cosmological parameters
# example: 'syn_spectralindex_random':(0.05, 'one') means that random spectral index of synchrotron radiation is required.
# The random size of the spectral index is 5%. The random pattern is "one", indicating random pixel dependently.
# If the random pattern is 'multi',  meaning random pixel independently. Please refer to Table 2 in our paper for details.
data_config_random_one_05 = {'syn_spectralindex_random':(0.05, 'one'), 'syn_amplitude_random':(0.1, 'one'),
                           'dust_spectralindex_random':(0.05, 'one'), 'dust_amplitude_random':(0.1, 'one'),
                           'dust_temp_random':(0.05, 'multi'),
                               'ame_amplitude_random':(0.1, 'one'), 'Random_types_of_cosmological_parameters':'Uniform'}



freqs_LiteBIRD = np.array([50,78,100,119,140,166,195,235,280,337])
Beams_LiteBIRD = np.array([56.8,36.9,30.2,26.3,23.7,28.9,28.0,24.7,22.5,20.9])
Sens_LiteBIRD = np.array([32.78,18.59,12.93,9.79,9.55,5.81,7.12,15.16,17.98,24.99])
output_beam_LiteBIRD =  28.9
output_freq_LiteBIRD =  166

nside = 128

save_data_dir = 'data/DATA_128_rand_s5_d10/'
save_result_dir = 'data/DATA_results_128_rand_s5_d10/'

# The number of samples for simulating sky maps

# The sample sizes of sky map for the training set, validation set，and test set are 1000, 300, and 300, respectively.
N_sky_maps = [1200, 400, 400]
# N_sky_maps = [1, 1, 1]

# The sample sizes of noise map for the training set, validation set，and test set are 1000, 300, and 300, respectively.
N_noise_maps = [1200, 400, 400]
# N_noise_maps = [1, 1, 1]

# Do you use 'half-split maps' for testing? Our paper uses the 'half-split maps'.
is_half_split_map = True

# Do you use a full-sky map for testing？As a tutorial, we use partial-sky ('block_-1') for testing
is_fullsky = True
# Training with a full-sky map requires a significant amount of GPU memory (>24GB), and it is recommended to use multiple GPUs for training.

# For CMB experiments with high white noise levels, such as LiteBIRD, we set the expected output of CNN to use CMB+ILC noise. Thus, we need calculate ILC noise using ILC method.
# Please refer to Section 3.2 in our paper for details.
# Readers can use the following six lines of code to calculate ILC noise.
using_ilc_cmbmap = False # Is ILC noise used as the expected noise output.

is_polarization_data = True # Is polarization data used for testing? If false, the simulated data includes temperature and polarization.
block_n = 'block_0' # The sky is divided into 12 blocks, and block 0 is selected for testing. Only valid for parameter is_fullsky = True
dataset_type = ['training_set', 'validation_set', 'testing_set'] # The dataset includes training, validation, and testing sets
padding = False

# if len(sys.argv) > 0:
#     config_file = sys.argv[1]
# else:
#     print("Please enter the configuration file")
#     sys.exit(1)
#
# config = load_pkl(config_file)

Data_parameters = {
    'data_config_random': data_config_random_one_05,
    'freqs': freqs_LiteBIRD,
    "output_freq": output_freq_LiteBIRD,
    "beams": Beams_LiteBIRD,
    "output_beam": output_beam_LiteBIRD,
    "sens": Sens_LiteBIRD,
    "nside": nside,
    "save_data_dir": save_data_dir,
    "save_result_dir": save_result_dir,
    "is_half_split_map": is_half_split_map,
    "is_polarization_data": is_polarization_data,
    'is_fullsky': is_fullsky,
    "block_n": block_n,
    'padding': padding,
    "dataset_type": dataset_type,
    "N_sky_maps": N_sky_maps,
    "N_noise_maps": N_noise_maps,
    "N_threads_preprocessing": 1,
    "component": 'QU'
}

if is_fullsky:
    ilc_mask = np.ones(12*nside**2)
else:
    ilc_mask = get_mask_for_block(block_n,nside)

ILC_parameters = {
    "using_ilc_cmbmap": using_ilc_cmbmap,
    "ilc_mask": ilc_mask,
    "ILC_N_threads": 1
}

Training_CNN_parameters = {
    "iteration": 3e4,
    "batch_size": 1,
    "learning_rate": 0.01,
    "device_ids": [0],
    "CNN_model": 'CMBFSCNN_level3'
}

conf_paras = {**Data_parameters, **ILC_parameters, **Training_CNN_parameters}

if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)

# save_pkl(conf_paras, save_data_dir+'config')

cmbfcnn = CMBFSCNN(conf_paras)
cmbfcnn.run_CMBFSCNN()
