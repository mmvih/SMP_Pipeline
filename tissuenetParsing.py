import os, copy
import logging, argparse

import matplotlib.pyplot as plt
import numpy as np

from skimage import morphology
from skimage import segmentation

import bfio
from bfio import BioWriter

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger("ftl")
logger.setLevel("DEBUG")

def convert_to_boundarymaps(groundtruth_boundary_array):
    anno = morphology.label(groundtruth_boundary_array)

    boundaries = segmentation.find_boundaries(anno)
    dilated_boundaries = morphology.binary_dilation(boundaries)

    label_binary = np.zeros((anno.shape + (3,)))
    label_binary[(anno == 0) & (boundaries == 0), 0] = 1
    label_binary[(anno != 0) & (boundaries == 0), 1] = 1
    label_binary[boundaries == 1, 2] = 1
    
    dilated_label_binary = np.zeros((anno.shape + (3,)))
    dilated_label_binary[(anno == 0) & (dilated_boundaries == 0), 0] = 1
    dilated_label_binary[(anno != 0) & (dilated_boundaries == 0), 1] = 1
    dilated_label_binary[dilated_boundaries == 1, 2] = 1

    # erode away the center
    gt_cntrbnry_2pxlerro_arr = np.zeros(anno.shape) 
    gt_cntrbnry_2pxlerro_arr[label_binary[:, :, 1] == 1] = 1  # center values are set to 1
    # print("NUMBER OF 1s in EROSION: ", np.sum(gt_cntrbnry_2pxlerro_arr))


    # three pixel values in this output
    groundtruth_multiclass_array = np.zeros(anno.shape)
    groundtruth_multiclass_array[dilated_label_binary[:, :, 2] == 1] = 1  # border values are set to 1
    groundtruth_multiclass_array[dilated_label_binary[:, :, 1] == 1] = 2  # center values are set to 2
    count_twos = np.count_nonzero((groundtruth_multiclass_array == 2).all)
    if count_twos == 0:
        groundtruth_multiclass_array[groundtruth_multiclass_array == 0] = 2

    # only the border of the cyst is defined with a pixel value of 1
    groundtruth_borderbinary_array = np.zeros(anno.shape)
    groundtruth_borderbinary_array[groundtruth_multiclass_array == 1] = 1

    # only the center of the cyst is defined with a pixel value of 1
    gt_cntrbnry_3pxlerro_arr = np.zeros(anno.shape)
    gt_cntrbnry_3pxlerro_arr[groundtruth_multiclass_array == 2] = 1

    return groundtruth_multiclass_array, groundtruth_borderbinary_array, gt_cntrbnry_3pxlerro_arr, gt_cntrbnry_2pxlerro_arr


def save_file(npz_location, classofdata, typeofdata, npz_dir, output_dir, suffix):
    # mapping out all the outputs
    diction = np.load(os.path.join(npz_dir, npz_location))
    X, y = diction['X'], diction['y']
    tissue_list, platform_list = diction['tissue_list'], diction['platform_list']

    class_directory = os.path.join(output_dir, classofdata)
    if not os.path.isdir(class_directory):
        os.mkdir(class_directory)

    output_directory = os.path.join(class_directory, typeofdata)
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # sanity check
    num_examples = len(tissue_list)
    assert len(tissue_list) == len(platform_list)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == len(tissue_list)

    # saving images in one directory and different kinds of ground truth in other directories.
    image_directory = os.path.join(output_directory, "image")
    if not os.path.isdir(image_directory):
        os.mkdir(image_directory)
    # We have four different groundtruths being created
    groundtruth_directory = os.path.join(output_directory, "groundtruth")
    if not os.path.isdir(groundtruth_directory):
        os.mkdir(groundtruth_directory)
    groundtruth_multiclass_directory = os.path.join(output_directory, "groundtruth_multiclass")
    if not os.path.isdir(groundtruth_multiclass_directory):
        os.mkdir(groundtruth_multiclass_directory)
    groundtruth_borderbinary_directory = os.path.join(output_directory, "groundtruth_borderbinary")
    if not os.path.isdir(groundtruth_borderbinary_directory):
        os.mkdir(groundtruth_borderbinary_directory)
    groundtruth_centerbinary3_directory = os.path.join(output_directory, "groundtruth_centerbinary_2pixelsmaller")
    if not os.path.isdir(groundtruth_centerbinary3_directory):
        os.mkdir(groundtruth_centerbinary3_directory)
    groundtruth_centerbinary2_directory = os.path.join(output_directory, "groundtruth_centerbinary_1pixelsmaller")
    if not os.path.isdir(groundtruth_centerbinary2_directory):
        os.mkdir(groundtruth_centerbinary2_directory)

    for ex in range(num_examples):

        # separate out the nuclear data from the cell's data
        if classofdata == "nuclear":
            image_array = X[ex, :, :, 0].squeeze()
            groundtruth_array = y[ex, :, :, 1].squeeze()
        else:
            image_array = X[ex, :, :, 1].squeeze()
            groundtruth_array = y[ex, :, :, 0].squeeze()
        # Other ground truth 
        groundtruth_multiclass_array = copy.deepcopy(groundtruth_array)
        groundtruth_multiclass_array, groundtruth_borderbinary_array, \
            gt_cntrbnry_3pxlerro_arr, gt_cntrbnry_2pxlerro_arr = convert_to_boundarymaps(groundtruth_multiclass_array)

        # all output (images and groundtruths will have the same name, just saved in different directories)
        file_outputname = "{0}_{1}_{2}.{3}".format(classofdata, typeofdata, ex, suffix)

        # the images range from 0 to 1, therefore multiply by 255 if need to view the image in matplotlib
        image_file = os.path.join(image_directory, file_outputname)
        groundtruth_file = os.path.join(groundtruth_directory, file_outputname)
        groundtruth_multiclass_file = os.path.join(groundtruth_multiclass_directory, file_outputname)
        groundtruth_borderbinary_file = os.path.join(groundtruth_borderbinary_directory, file_outputname)
        groundtruth_centerbinary3_file = os.path.join(groundtruth_centerbinary3_directory, file_outputname)
        groundtruth_centerbinary2_file = os.path.join(groundtruth_centerbinary2_directory, file_outputname)

        # Save the images
        with BioWriter(image_file, X=image_array.shape[1], 
                                   Y=image_array.shape[0],
                                   Z=1, C=1, T=1, dtype=image_array.dtype) as bw:
            bw[:] = image_array
        with BioWriter(groundtruth_file, X=groundtruth_array.shape[1],
                                         Y=groundtruth_array.shape[0],
                                         Z=1, C=1, T=1, dtype=groundtruth_array.dtype) as bw:
            bw[:] = groundtruth_array
        with BioWriter(groundtruth_multiclass_file, X = groundtruth_multiclass_array.shape[1],
                                                    Y = groundtruth_multiclass_array.shape[0],
                                                    Z = 1, C = 1, T = 1, dtype=groundtruth_multiclass_array.dtype) as bw:
            bw[:] =  groundtruth_multiclass_array
        with BioWriter(groundtruth_borderbinary_file, X = groundtruth_borderbinary_array.shape[1],
                                                      Y = groundtruth_borderbinary_array.shape[0],
                                                      Z = 1, C = 1, T = 1, dtype=groundtruth_borderbinary_array.dtype) as bw:
            bw[:] =  groundtruth_borderbinary_array
        with BioWriter(groundtruth_centerbinary3_file, X = gt_cntrbnry_3pxlerro_arr.shape[1],
                                                       Y = gt_cntrbnry_3pxlerro_arr.shape[0],
                                                       Z = 1, C = 1, T = 1, dtype=gt_cntrbnry_3pxlerro_arr.dtype) as bw:
            bw[:] =  gt_cntrbnry_3pxlerro_arr
        with BioWriter(groundtruth_centerbinary2_file, X = gt_cntrbnry_2pxlerro_arr.shape[1],
                                                       Y = gt_cntrbnry_2pxlerro_arr.shape[0],
                                                       Z = 1, C = 1, T = 1, dtype=gt_cntrbnry_2pxlerro_arr.dtype) as bw:
            bw[:] =  gt_cntrbnry_2pxlerro_arr        

        if ex == 0:  # checking the first image qualitatively
            fig, axes = plt.subplots(3, 2, figsize=(16, 24))
            axes[0, 0].imshow(image_array)
            axes[0, 1].imshow(groundtruth_array)
            axes[1, 0].imshow(groundtruth_multiclass_array)
            axes[1, 1].imshow(gt_cntrbnry_2pxlerro_arr)
            axes[2, 0].imshow(groundtruth_borderbinary_array)
            axes[2, 1].imshow(gt_cntrbnry_3pxlerro_arr)
            axes[0, 0].set_title("Image")
            axes[0, 1].set_title(f"Ground Truth: {np.min(groundtruth_array)}-{np.max(groundtruth_array)}")
            axes[1, 1].set_title(f"Ground Truth Binary Centers 1 pixel errosion: {np.min(gt_cntrbnry_2pxlerro_arr)}-{np.max(gt_cntrbnry_2pxlerro_arr)}")
            axes[1, 0].set_title(f"Ground Truth Multiclass: {np.min(groundtruth_multiclass_array)}-{np.max(groundtruth_multiclass_array)}")
            axes[2, 0].set_title(f"Ground Truth Binary Borders: {np.min(groundtruth_borderbinary_array)}-{np.max(groundtruth_borderbinary_array)}")
            axes[2, 1].set_title(f"Ground Truth Binary Centers 2 pixel errosion: {np.min(gt_cntrbnry_3pxlerro_arr)}-{np.max(gt_cntrbnry_3pxlerro_arr)}")

            fig.suptitle("Example Plot for {}'s Data ({})".format(classofdata, typeofdata))
            plot_name = "{}_{}.jpg".format(classofdata, typeofdata)
            plt.savefig(os.path.join(output_dir, plot_name))

        logger.info("Saved {}'s {} data ({}/{})".format(classofdata, typeofdata, ex, num_examples - 1))
    logger.info("Saved all of {}'s {} data".format(classofdata, typeofdata))

def main():

    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

    parser.add_argument('--npzDir', dest='npzDir', type=str, required=True, \
                        help='Path to Directory containing Input Predictions for all Models')
    parser.add_argument('--outputDir', dest='outputDir', type=str, required=True,
                        help='Path to Groundtruth Directory that match the inputPredictions')
    parser.add_argument('--suffix', dest='suffix', type=str, required=True,
                        help='Path to where the Output Labels will be saved')

    args = parser.parse_args()
    npz_dir = args.npzDir
    output_dir = args.outputDir
    suffix = args.suffix

    logger.info(f"NPZ Directory : {npz_dir}")
    logger.info(f"Output Tissuenet Directory : {output_dir}")
    logger.info(f"Suffix : {suffix}")

    if not os.path.isdir(output_dir):
        raise ValueError("Output Directory ({}) does not exist; please create manually".format(output_dir))

    # run the functions for different parameters
    save_file(os.path.join(npz_dir, "tissuenet_v1.0_test.npz"), "nuclear", "test",  npz_dir, output_dir, suffix)
    save_file(os.path.join(npz_dir, "tissuenet_v1.0_train.npz"), "nuclear", "train",  npz_dir, output_dir, suffix)
    save_file(os.path.join(npz_dir, "tissuenet_v1.0_val.npz"), "nuclear", "validation",  npz_dir, output_dir, suffix)

    save_file(os.path.join(npz_dir, "tissuenet_v1.0_test.npz"), "cell", "test",  npz_dir, output_dir, suffix)
    save_file(os.path.join(npz_dir, "tissuenet_v1.0_train.npz"), "cell", "train",  npz_dir, output_dir, suffix)
    save_file(os.path.join(npz_dir, "tissuenet_v1.0_val.npz"), "cell", "validation",  npz_dir, output_dir, suffix)

main()
