import os
import numpy as np

def load_simulated_data(data_directory, logger):

    data_path = os.path.join(data_directory, 'curves1k.npz')
    logger.info(f"Loading simulated data from {data_path}")
    data_array = np.load(data_path)['curves']
    
    conditions = [(1e4, 5e10, 15e4, 0), (5e10, 1e4, 15e4, 0), (1e5, 1e5, 15e4, 5e5)]
    num_strains, num_conditions, _, seq_length = data_array.shape
    all_n_values = np.empty((num_strains, num_conditions * seq_length))
    
    for strain_index in range(num_strains):
        concatenated_values = np.concatenate(
            [data_array[strain_index, condition_index, 2, :] for condition_index in range(num_conditions)]
        )
        all_n_values[strain_index, :] = concatenated_values

    logger.info(f"Simulated data shape: {all_n_values.shape}")
    return all_n_values