import os,sys
import logging,argparse

import numpy as np

import bfio
from bfio import BioWriter

from filepattern import FilePattern 

import torch
import torchvision

from multiprocessing import Queue 
import subprocess
import concurrent.futures

import time

polus_smp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polus-plugins/segmentation/polus-smp-training-plugin/")
sys.path.append(polus_smp_dir)

from src.utils import get_labels_mapping
from src.utils import Dataset
from src.training import MultiEpochsDataLoader

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("savepredictions")
logger.setLevel("DEBUG")


def getLoader(images_Dir, 
              labels_Dir,
              file_pattern):
    
    images_fp = FilePattern(images_Dir, file_pattern)
    labels_fp = FilePattern(labels_Dir, file_pattern)

    image_array, label_array, names = get_labels_mapping(images_fp(), labels_fp(), provide_names=True)

    testing_dataset = Dataset(images=image_array,
                              labels=label_array)
    testing_loader = MultiEpochsDataLoader(testing_dataset, num_workers=4, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)
    
    return testing_dataset, images_fp, labels_fp, names

"""MAKING PREDICTIONS"""
queue = Queue()
def evaluation(cuda_num : str,
               smp_input_path : str,
               smp_output_path : str,
               test_loader,
               names):

    
    try:
        
        modelpth_path = os.path.join(smp_input_path, "model.pth")
        
        if not os.path.exists(smp_output_path):
            os.mkdir(smp_output_path)
        
        tor_device = torch.device(f"cuda:{cuda_num}")
        model = torch.load(modelpth_path, map_location=tor_device)
        
        predictions_path = os.path.join(smp_output_path, "predictions") # this is where images get saved to
  
        if not os.path.exists(predictions_path):
            os.mkdir(predictions_path)
        
        img_count = 0
        logger.info(f"Saving predictions at {predictions_path}")
        for im, gt in test_loader:
            im_tensor = torch.from_numpy(im).to(tor_device).unsqueeze(0)
            pr_tensor = model.predict(im_tensor)

            pr = pr_tensor.cpu().detach().numpy().squeeze()[..., None, None, None]
            pr[pr >= .50] = 1
            pr[pr < .50] = 0
            
            
            filename = names[img_count][:-4] + ".ome.tif"
            pr_filename = os.path.join(predictions_path, filename)
            
            with BioWriter(pr_filename, Y=pr.shape[0],
                                        X=pr.shape[1],
                                        Z=1,
                                        C=1,
                                        T=1,
                                        dtype=pr.dtype) as bw_pr:
                bw_pr[:] = pr
            
            img_count = img_count + 1
            
        queue.put(cuda_num)
            
    except Exception as e:
        queue.put(cuda_num)
        logger.info(f"ERROR: {e}")

def main():
    
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

    parser.add_argument('--inputModels', dest='inputModels', type=str, required=True, \
                        help='Path to Input Models')
    parser.add_argument('--outputPredictions', dest='outputPredictions', type=str, required=True, \
                        help='Path to Output Predictions')
    parser.add_argument('--imagesTestDir', dest='imagesTestDir', type=str, required=True, \
                        help='Path to Images that are for testing')
    parser.add_argument('--labelsTestDir', dest='labelsTestDir', type=str, required=True, \
                        help='Path to Labels that are for testing')
    parser.add_argument('--filePattern', dest='filePattern', type=str, required=False,
                        default=".*", help="Pattern of Images for creating predictions")
    
    args = parser.parse_args()
    input_models_dirpath =  args.inputModels
    output_predictions_dirpath = args.outputPredictions
    logger.info(f"Input Models Directory : {input_models_dirpath}")
    logger.info(f"Output Predictions Directory : {output_predictions_dirpath}")
    
    testing_images_dirpath = args.imagesTestDir
    testing_labels_dirpath = args.labelsTestDir
    logger.info(f"Testing Images Directory : {testing_images_dirpath}")
    logger.info(f"Testing Labels Directory : {testing_labels_dirpath}")
    
    file_pattern = args.filePattern
    logger.info(f"File Pattern : {file_pattern}")
    
    input_models_list = os.listdir(input_models_dirpath)
    num_models = len(input_models_list)
    logger.info(f"There are {num_models} to generate predictions for")
    
    NUM_GPUS = torch.cuda.device_count()
    NUM_PROCESSES = num_models

    logger.info(f"Queuing up {NUM_GPUS} GPUs ...")
    for gpu_ids in (range(NUM_GPUS)):
        queue.put(str(gpu_ids))

    iter_input_models_list = iter(input_models_list)

    logger.info("Getting Loaders ...")
    test_loader, _, _, names = getLoader(images_Dir=testing_images_dirpath,
                                         labels_Dir=testing_labels_dirpath,
                                         file_pattern=file_pattern)
    num_examples = len(test_loader)
    logger.info(f"Each model will be generating {num_examples} predictions")

    counter = 0
    sleeping_in = 0    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_GPUS+(NUM_GPUS/2)) as executor:
        while counter != num_models-1:
            if not queue.empty():
                sleeping_in = 0
                
                curr_smp_model = next(iter_input_models_list)
                logger.info(f"LOOKING at {curr_smp_model}")
                
                # looking at only ONE input model and ONE output location
                input_model_dirpath = os.path.join(input_models_dirpath, curr_smp_model)
                output_prediction_dirpath = os.path.join(output_predictions_dirpath, curr_smp_model)
                
                ERROR_path = os.path.join(input_model_dirpath, "ERROR")
                if os.path.exists(ERROR_path):
                    logger.debug(f"Not Running - ERROR Exists {input_model_dirpath}")
                    counter += 1
                    continue
                
                modelpth_path = os.path.join(input_model_dirpath, "model.pth")
                if not os.path.exists(modelpth_path):
                    counter += 1
                    logger.debug(f"Not Running - model.pth does not Exists {input_model_dirpath}")
                    continue
                
                predictions_path = os.path.join(output_prediction_dirpath, "predictions") # this is where images get saved to
                if os.path.exists(predictions_path):
                    num_predictions = len(os.listdir(predictions_path))
                    if num_predictions == num_examples:
                        counter += 1
                        logger.debug(f"Not Running - already has outputs {output_prediction_dirpath}")
                        continue
    
                
                executor.submit(evaluation, queue.get(), input_model_dirpath, output_prediction_dirpath, test_loader, names)
                counter = counter + 1
                logger.info(f"analyzed {counter}/{num_models-1} models\n")

            else:
                sleeping_in += 1
                logger.info(f"sleeping in : x{sleeping_in}")
                time.sleep(5)
                continue

main()