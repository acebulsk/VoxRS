import sys
import numpy as np
import pandas as pd
import yaml
import os
import src.las_ray_sampling as lrs
import tools.lrs_footprint_products as tools

def main(config_file):
    """
    Configuration file for generating synthetic hemispheres of expected lidar returns by voxel ray samping of lidar
    (eg. to estimate light transmittance across the hemisphere at a given point, Staines thesis figure 3.3).
        batch_dir: directory for all batch outputs
        pts_in: coordinates and elevations at which to calculate hemispheres
        
    :param config_file: Path to the configuration file, if not provided defaults to yaml in this dir
    :return:
    """

    # Read YAML config file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    config_id = config["config_id"]
    working_dir = os.path.normpath(config["working_dir"])

    # Initiate voxel obj
    vox = lrs.VoxelObj()
    vox.vox_hdf5 = os.path.join(working_dir, "voxrs_" + config_id + '_vox.h5')  # (str) file path to vox file (.hdf5)

    # # LOAD VOX
    # load existing vox files, using vox file path
    vox = lrs.load_vox(vox.vox_hdf5, load_post=True)

    # create outputs folder if not exists
    outputs_dir = os.path.join(working_dir, "outputs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    runtag = config["runtag"]
    if runtag is None:
        runtag = ''
    elif runtag != '':
        runtag = "_" + runtag + "_"

    # # VOLUMETRIC RESAMPLING
    if config["resample_las"]:

        # create volumetric_resampling folder if not exists
        vol_dir = os.path.join(outputs_dir, 'volumetric_resampling')
        if not os.path.exists(vol_dir):
            os.makedirs(vol_dir)     

        samps_per_vox = config["samps_per_vox"]  # (int) volumetric resample rate [samples per voxel]
        sample_threshold = config["sample_threshold"]  # (int) noise filter will drop all returns from voxels where total returns <= sample_threshold           
        las_out = os.path.join(vol_dir, "vol_" + config_id + runtag + "_resampled.las")  # (str) output .las file path

        # GENERATE RESAMPLED POINT CLOUD
        lrs.vox_to_las(vox.vox_hdf5, las_out, samps_per_vox, sample_threshold)

    # # GRID RESAMPLING
    if config["resample_grid"]:

        # create grid_resampling folder if not exists
        grid_dir = os.path.join(outputs_dir, 'grid_resampling')
        if not os.path.exists(grid_dir):
            os.makedirs(grid_dir)

        phi_from = config["phi"][0]
        phi_to = config["phi"][1]
        phi_by = config["phi"][2]

        phi_list = list(range(phi_from, phi_to + 1, phi_by))

        if theta_offset is None and theta_step is None:
            print('Using theta sequence specified in the yaml file.')

            theta_from = config["theta"][0]
            theta_to = config["theta"][1]
            theta_by = config["theta"][2]

        else:
            print("Overriding the yaml file theta sequence and using argument values: ")
            print("theta-offset: ", theta_offset)
            print("theta-step: ", theta_step)

            theta_from = theta_offset*theta_step
            theta_to = theta_from + (theta_step-1)
            theta_by = 1

            runtag = runtag + "t" + str(theta_from) + "_" + str(theta_to)

        theta_list = list(range(theta_from, theta_to + 1, theta_by))

        # create two lists of the same length each element of the first list is repeated across each element of the second
        phi_theta = []

        # Iterate over each element of the first list
        for x in phi_list:
            # Repeat the current element of the first list for each element of the second list
            for y in theta_list:
                # Append the pair (x, y) to the result list
                phi_theta.append((x, y))

        # Extract first elements into a separate list
        phi_list = [pair[0] for pair in phi_theta]

        # Extract second elements into a separate list
        theta_list = [pair[1] for pair in phi_theta]

        phi_theta_df = {'phi_deg': phi_list, 'theta_deg': theta_list}

        # Create a DataFrame from the dictionary
        phi_theta_df = pd.DataFrame(phi_theta_df)
        phi_theta_df['phi_radians'] = np.deg2rad(phi_theta_df['phi_deg'])
        phi_theta_df['theta_radians'] = np.deg2rad(phi_theta_df['theta_deg'])
        print(phi_theta_df)
        phi_theta_df.to_csv(grid_dir + '/' + config_id + runtag + "grid_resampling_phi_theta_dict.csv", index=False)
        
        # define VoxRS grid metadata object
        rsgmeta = lrs.RaySampleGridMetaObj()
        rsgmeta.file_dir = grid_dir
        rsgmeta.lookup_db = 'posterior'  # (str) db lookup, default 'posterior'
        rsgmeta.config_id = config_id
        rsgmeta.agg_sample_length = vox.sample_length  # (num) ray resample length (default to same as sample length)
        rsgmeta.agg_method = 'single_ray_agg'  # (str) aggregation method, default 'single_ray_agg'
        
        rsgmeta.src_ras_file = config["dem_in"]  # (str) complete path to raster (geotif) file with coordinates and elevations at which to calculate hemispheres, masked to points of interest
        rsgmeta.mask_file = rsgmeta.src_ras_file  # (str) complete path to raster (geotif) file with mask of points of interest (use src_ras_file by default)
        rsgmeta.phi_d = np.array(phi_list) # zenith angle of rays (deg)
        rsgmeta.theta_d = np.array(theta_list) # azimuth angle of rays (clockwise from north, from above looking down, in deg)
        rsgmeta.phi = np.array(phi_list) * np.pi / 180  # zenith angle of rays (radians)
        rsgmeta.theta = np.array(theta_list) * np.pi / 180  # azimuth angle of rays (clockwise from north, from above looking down, in radians)
        rsgmeta.max_distance = config["max_distance"]  # maximum distance [m] to sample ray (balance computation time with accuracy at distance)
        rsgmeta.min_distance = config["min_distance"]  # minimum distance [m] to sample ray (default 0, increase to avoid "lens occlusion" within dense voxels)
        rsgmeta.id = 0

        if type(rsgmeta.phi) is not np.ndarray:
            rsgmeta.phi = [rsgmeta.phi]
            rsgmeta.theta = [rsgmeta.theta]
        
        rsgmeta.file_name = ["grid_" + rsgmeta.config_id + runtag + "_p{:.4f}_t{:.4f}.tif".format(rsgmeta.phi[ii], rsgmeta.theta[ii]) for ii in range(0, np.size(rsgmeta.phi))]
        rsgm = lrs.rs_gridgen(rsgmeta, vox, runtag, initial_index=0)  # run grid resampling. Can take single or list of runs.

    # # HEMISPHERE RESAMPLING
        
    if config["resample_hemi"]:

        # create hemi_dir (with error handling)
        hemi_dir = os.path.join(outputs_dir, 'hemisphere_resampling', runtag)
        lrs.create_dir(hemi_dir, desc='hemi')

        # define VoxRS hemisphere metadata object
        rshmeta = lrs.RaySampleGridMetaObj()
        rshmeta.file_dir = hemi_dir
        rshmeta.config_id = config_id
        rshmeta.agg_sample_length = vox.sample_length  # (num) ray resample length (default to same as sample length)
        rshmeta.lookup_db = 'posterior'  # (str) db lookup, default 'posterior'
        rshmeta.agg_method = 'single_ray_agg'  # (str) aggregation method, default 'single_ray_agg'

        # HEMISPHERE RAY GEOMETRY PARAMETERS
        rshmeta.img_size = config["img_size"]  # angular resolution (square) ~ samples / pi

        # phi_step = (np.pi / 2) / (180 * 2)  # alternative way of defining, based on angular step
        # rshmeta.max_phi_rad = phi_step * rshmeta.img_size

        rshmeta.max_phi_rad = config["max_phi_deg"] * np.pi/180  # maximum zenith angle to sample  (pi/2 samples to horizon)
        hemi_m_above_ground = config["hemi_m_above_ground"]  # height [m] above ground points at which to generate hemispheres
        rshmeta.max_distance = config["max_distance"]  # maximum distance [m] to sample ray (balance computation time with accuracy at distance)
        rshmeta.min_distance = config["min_distance"]  # minimum distance [m] to sample ray (default 0, increase to avoid "lens occlusion" within dense voxels)

        # PROCESSING PARAMETERS
        tile_count_1d = 5  # (int) number of square tiles along one side (total # of tiles = tile_count_1d^2)
        n_cores = config["n_cores"]  # (int) number of processing cores
        
        # PTS CONFIGURATION
        pts_in = config["pts_in"]  # (str) path to .csv file with coordinates and elevations at which to calculate hemispheres
        #            pts file must include the following header labels: id, easting_m, northing_m, elev_m
        pts = pd.read_csv(pts_in)

        rshmeta.id = pts.id
        rshmeta.origin = np.array([pts.easting_m,
                                   pts.northing_m,
                                   pts.elev_m + hemi_m_above_ground]).swapaxes(0, 1)

        rshmeta.file_name = ["hemi_" + rshmeta.config_id + "_" + str(idd) + ".tif" for idd in pts.id]

        rshm = lrs.rs_hemigen(rshmeta, vox, tile_count_1d, n_cores)

        snow_on_coef = 0.37181197  # python tx drop 5

        # Loop through all .tif files in the directory and creat png outputs
        # for idx, row in enumerate(pts.iterrows()):
        #     tools.hemi_view(hemi_dir, idx, snow_on_coef)


# add config for sample from grid

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        config_file = sys.argv[1]
    else:
        print("Usage: python script_name.py <config_file> [theta-offset] [theta-step]")
        print("Using default configuration file: config_2_resampling.yml")
        config_file = "config_2_resampling.yml"
    
    # Setting default values for theta-offset and theta-step
    theta_offset = None
    theta_step = None
    
    # Check if theta-offset and theta-step arguments are provided
    if len(sys.argv) >= 3:
        theta_offset = int(sys.argv[2])
    if len(sys.argv) >= 4:
        theta_step = int(sys.argv[3])

    # Check if theta_offset is provided but theta_step is not
    if theta_offset is not None and theta_step is None:
        print("Error: Theta-step not provided.")
        sys.exit(1)

    main(config_file)

# preliminary visualization - same as hemi_view function in tools

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import tifffile as tif
# ii = 0
# peace = tif.imread(rshm.file_dir[ii] + rshm.file_name[ii])
# plt.imshow(peace[:, :, 2], interpolation='nearest')