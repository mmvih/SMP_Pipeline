import os, sys, json
import logging, argparse

import numpy as np

import bfio
from bfio import BioReader

from multiprocessing import Queue 
import subprocess
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("inputScript")
logger.setLevel("DEBUG")

def figurePlot(images_dirpath,
               groundtruth_dirpath,
               predictions_dirpath,
               plot_path,
               curr_smp_model):
    
    predictions_list = os.listdir(predictions_dirpath)
    fig, ax = plt.subplots(1, 3, figsize=(4*3, 4*1))
    ax[0].set_title("Original Image") 
    ax[1].set_title("Groundtruth")
    ax[2].set_title("Predictions")
    
    for prediction in predictions_list:
        
        logger.debug(f"{curr_smp_model} - {prediction}")
        prediction_path  = os.path.join(predictions_dirpath, prediction)
        image_path       = os.path.join(images_dirpath, prediction)
        groundtruth_path = os.path.join(groundtruth_dirpath, prediction)
        
        with BioReader(image_path) as br_image:
            image_arr = br_image[:].squeeze()
        with BioReader(groundtruth_path) as br_groundtruth:
            groundtruth_arr = br_groundtruth[:].squeeze()
        with BioReader(prediction_path) as br_prediction:
            prediction_arr = br_prediction[:].squeeze()
            
        groundtruth_mask = np.ma.masked_where(groundtruth_arr==0, groundtruth_arr)
        prediction_mask  = np.ma.masked_where(prediction_arr==0, prediction_arr)
        
        fig.suptitle(f"{curr_smp_model}\n{prediction}")
        fig.tight_layout()
                
        ax[0].imshow(image_arr, cmap="gray")
        
        ax[1].imshow(image_arr, cmap="gray", alpha=.95)
        if len(np.unique(groundtruth_arr)) < 3:
            ax[1].imshow(groundtruth_mask, cmap="Wistia", alpha=1)
        else:
            ax[1].imshow(groundtruth_mask, alpha=1)
        
        ax[2].imshow(image_arr, cmap="gray", alpha=.95)
        if len(np.unique(prediction_arr)) < 3:
            ax[2].imshow(prediction_mask, cmap="Wistia", alpha=1)
        else:
            ax[2].imshow(prediction_mask, alpha=1)           
        
        if prediction.endswith(".ome.tif"):
            output_path = os.path.join(plot_path, f"{prediction[:-8]}.jpg")
        else:
            output_path = os.path.join(plot_path, f"{prediction[:-4]}.jpg")
        
        plt.savefig(output_path)
        plt.cla()

def main():
    
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Creating Plots from Images with Masked Groundtruth and Predictions')
    
    parser.add_argument('--inputImages', dest='imagesDir', type=str, required=True, \
                        help='Path to Directory containing images')
    parser.add_argument('--inputPredictions', dest='predictionsDir', type=str, required=True, \
                        help='Path to Directory containing model predictions')
    parser.add_argument('--modelPredictionsbase', dest='modelPredictionsbase', type=str, required=True, \
                        help='Either ftl or predictions')
    parser.add_argument('--inputGroundtruth', dest='groundtruthDir', type=str, required=True, \
                        help='Path to Directory containing groundtruth')
    parser.add_argument('--inputMetrics', dest='metricDir', type=str, required=False, \
                        help='Path to Directory containing model metrics')
    parser.add_argument('--outputPlots', dest='outputPlots', type=str, required=True, \
                        help="Path to the Directory containing output models")
    
    args = parser.parse_args()
    images_dirpath = args.imagesDir
    input_predictions_dirpath = args.predictionsDir
    groundtruth_dirpath = args.groundtruthDir
    metric_dirpath = args.metricDir
    model_predictionsbase = args.modelPredictionsbase
    output_plots_dirpath = args.outputPlots
    
    
    if not os.path.exists(output_plots_dirpath):
        raise ValueError(f"Output Directory - {output_plots_dirpath} - does not exist")
    
    logger.info(f"Images Directory : {os.path.abspath(images_dirpath)}")
    logger.info(f"Groundtruth Directory : {os.path.abspath(groundtruth_dirpath)}")
    logger.info(f"Predictions Directory : {os.path.abspath(input_predictions_dirpath)}")
    logger.info(F"Metric Directory : {os.path.abspath(metric_dirpath)}")
    
    input_predictions_list = os.listdir(input_predictions_dirpath)
    num_models = len(input_predictions_list)
    
    counter = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()-10) as executor:
        for curr_smp_model in input_predictions_list:
            
            counter += 1
            logger.info(f"\n{counter}/{num_models}. {curr_smp_model}")
            
            input_prediction_dirpath = os.path.join(input_predictions_dirpath, curr_smp_model)
            
            if not os.path.isdir(input_prediction_dirpath):
                # ignore the json files
                logger.debug(f"Not Running {input_prediction_dirpath} - not a directory")
                continue
            
            output_plot_dirpath      = os.path.join(output_plots_dirpath, curr_smp_model)
            
            predictions_dirpath = os.path.join(input_prediction_dirpath, model_predictionsbase)
            plot_dirpath = os.path.join(output_plot_dirpath, model_predictionsbase)

            if os.path.exists(plot_dirpath):
                num_figures = len(os.listdir(plot_dirpath))
                if len(os.listdir(plot_dirpath)) == len(os.listdir(predictions_dirpath)):
                    logger.debug(f"Not Running {curr_smp_model} - figures already exist with {num_figures}")
                    continue
            
            if not os.path.exists(output_plot_dirpath):
                os.mkdir(output_plot_dirpath)
            
            plot_path = os.path.join(output_plot_dirpath, model_predictionsbase+"_plots")
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)
                
            executor.submit(figurePlot,
                            images_dirpath,
                            groundtruth_dirpath,
                            predictions_dirpath,
                            plot_path,
                            curr_smp_model)
            
main()
