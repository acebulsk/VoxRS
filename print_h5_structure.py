import h5py

def print_hdf5_structure(f, key, indent=0):
    print(' ' * indent + key)
    if isinstance(f[key], h5py.Group):
        for subkey in f[key].keys():
            print_hdf5_structure(f[key], subkey, indent + 4)
    elif isinstance(f[key], h5py.Dataset):
        dataset = f[key]
        print(' ' * (indent + 4) + 'Dataset shape:', dataset.shape)

#filename_hdf = '/media/alex/phd-data/cob-thesis/alex-example-processing/test1/voxrs_test_19_052_las_traj.h5'
filename_hdf = '/media/alex/phd-data/cob-thesis/alex-example-processing/test1/voxrs_test_19_052_vox.h5'
h5_data = h5py.File(filename_hdf, 'r')

for key in h5_data.keys():
    print_hdf5_structure(h5_data, key)