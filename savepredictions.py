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
logger.setLevel("INFO")


def getLoader(images_Dir, 
              labels_Dir):
    
    filepattern = ".*"
    images_fp = FilePattern(images_Dir, filepattern)
    labels_fp = FilePattern(labels_Dir, filepattern)

    image_array, label_array, names = get_labels_mapping(images_fp(), labels_fp(), provide_names=True)

    testing_dataset = Dataset(images=image_array,
                              labels=label_array)
    testing_loader = MultiEpochsDataLoader(testing_dataset, num_workers=4, batch_size=10, shuffle=True, pin_memory=True, drop_last=True)
    
    return testing_dataset, images_fp, labels_fp, names

"""MAKING PREDICTIONS"""

queue = Queue()
def evaluation(smp_model : str, 
               cuda_num : str,
               smp_inputs_path : str,
               smp_outputs_path : str,
               test_loader,
               names):

    
    try:
        smp_model_dirpath = os.path.join(smp_inputs_path, smp_model)
        logger.info("LOOKING AT: ", smp_model_dirpath)
        
        modelpth_path = os.path.join(smp_model_dirpath, "model.pth")
        
        output_path = os.path.join(smp_outputs_path, smp_model)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        tor_device = torch.device(f"cuda:{cuda_num}")
        model = torch.load(modelpth_path, map_location=tor_device)
        
        pr_collection = os.path.join(output_path, "predictions") # this is where images get saved to
  
        if not os.path.exists(pr_collection):
            os.mkdir(pr_collection)
        
        img_count = 0
        logger.info(f"Generating predictions for {smp_model} : saving in {output_path}")
        for im, gt in test_loader:
            im_tensor = torch.from_numpy(im).to(tor_device).unsqueeze(0)
            pr_tensor = model.predict(im_tensor)

            pr = pr_tensor.cpu().detach().numpy().squeeze()[..., None, None, None]
            pr[pr >= .50] = 1
            pr[pr < .50] = 0
            
            
            filename = names[img_count][:-4] + ".ome.tif"
            pr_filename = os.path.join(pr_collection, filename)
            
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
    parser.add_argument('--imagesTestDir', dest='imagesTrainDir', type=str, required=True, \
                        help='Path to Images that are for testing')
    parser.add_argument('--labelsTestDir', dest='labelsTrainDir', type=str, required=True, \
                        help='Path to Labels that are for testing')
    
    args = parser.parse_args()
    smp_inputs_path =  args.inputModels
    smp_outputs_path = args.outputPredictions
    
    testing_images = args.imagesTestDir
    testing_labels = args.labelsTestDir
    
    smp_inputs_list = os.listdir(smp_inputs_path)
    
    NUM_GPUS = 8
    NUM_PROCESSES = len(smp_inputs_list)
    PROC_PER_GPU = int(np.ceil(NUM_PROCESSES/NUM_GPUS))

    logger.info("Queuing up GPUs ...")
    for gpu_ids in (range(NUM_GPUS)):
        queue.put(str(gpu_ids))

    iter_smp_inputs_list = iter(smp_inputs_list)

    num_models = len(smp_inputs_list)
 
    
    logger.info("Getting Loaders ...")
    test_loader, _, _, names = getLoader(images_Dir=testing_images,
                                         labels_Dir=testing_labels)

    counter = 0
    sleeping_in = 0    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        while counter != len(smp_inputs_list)-1:
            if not queue.empty():
                sleeping_in = 0
                
                curr_smp_model = next(iter_smp_inputs_list)
                
                smp_model_dirpath = os.path.join(smp_inputs_path, curr_smp_model)
                
                ERROR_path = os.path.join(smp_model_dirpath, "ERROR")
                if os.path.exists(ERROR_path):
                    logger.debug(f"Not Running - ERROR Exists {smp_model_dirpath}")
                    counter += 1
                    continue
                
                modelpth_path = os.path.join(smp_model_dirpath, "model.pth")
                if not os.path.exists(modelpth_path):
                    counter += 1
                    logger.debug(f"Not Running - model.pth does not Exists {smp_model_dirpath}")
                    continue
                
                
                output_path = os.path.join(smp_outputs_path, curr_smp_model)
                pr_collection = os.path.join(output_path, "predictions") # this is where images get saved to
    
                if os.path.exists(pr_collection):
                    if len(os.listdir(pr_collection)) == 1249:
                        counter += 1
                        logger.debug(f"Not Running - already has outputs")
                        continue
    
                
                executor.submit(evaluation, curr_smp_model, queue.get(), smp_inputs_path, smp_outputs_path, test_loader, names)
                counter = counter + 1
                logger.info(f"analyzed {counter}/{num_models-1} models")

            else:
                sleeping_in += 1
                logger.info(f"sleeping in : x{sleeping_in}")
                time.sleep(5)
                continue

main()