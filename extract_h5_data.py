import h5py
import numpy as np
import pandas as pd

def extract_3d_array_from_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        if 'return_data' in f:
            return_data = f['return_data']
            if isinstance(return_data, h5py.Dataset):
                return return_data[:]
            else:
                print("Return data is not a dataset.")
                return None
        else:
            print("No 'return_data' key found.")
            return None

def save_3d_array_as_csv_with_xyz(filename, output_filename, num_elements):
    data_3d_array = extract_3d_array_from_hdf5(filename)
    if data_3d_array is not None:
        x_dim, y_dim, z_dim = data_3d_array.shape
        # Create XYZ coordinates
        x_coords, y_coords, z_coords = np.meshgrid(range(x_dim), range(y_dim), range(z_dim), indexing='ij')
        x_coords = x_coords.flatten()[:num_elements]
        y_coords = y_coords.flatten()[:num_elements]
        z_coords = z_coords.flatten()[:num_elements]
        # Reshape data to 1D array
        data_1d_array = data_3d_array.reshape(-1)[:num_elements]
        # Create DataFrame with XYZ coordinates and data
        df = pd.DataFrame({'x': x_coords, 'y': y_coords, 'z': z_coords, 'value': data_1d_array})
        # Save DataFrame to CSV
        df.to_csv(output_filename, index=False)

# DO all
# def save_3d_array_as_csv_with_xyz(filename, output_filename):
#     data_3d_array = extract_3d_array_from_hdf5(filename)
#     if data_3d_array is not None:
#         x_dim, y_dim, z_dim = data_3d_array.shape
#         # Create XYZ coordinates
#         x_coords, y_coords, z_coords = np.meshgrid(range(x_dim), range(y_dim), range(z_dim), indexing='ij')
#         x_coords = x_coords.flatten()
#         y_coords = y_coords.flatten()
#         z_coords = z_coords.flatten()
#         # Reshape data to 1D array
#         data_1d_array = data_3d_array.reshape(-1)
#         # Create DataFrame with XYZ coordinates and data
#         df = pd.DataFrame({'x': x_coords, 'y': y_coords, 'z': z_coords, 'value': data_1d_array})
#         # Save DataFrame to CSV
#         df.to_csv(output_filename, index=False)

filename_hdf = '/media/alex/phd-data/cob-thesis/alex-example-processing/test1/voxrs_test_19_052_vox.h5'
output_csv_filename = '/media/alex/phd-data/cob-thesis/alex-example-processing/test1/voxrs_test_19_052_vox_return_data_1mil.csv'

num_elements = 1000000
save_3d_array_as_csv_with_xyz(filename_hdf, output_csv_filename, num_elements)
