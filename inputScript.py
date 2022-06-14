import os, sys
import shutil

import csv

import logging, argparse

import itertools
from itertools import repeat

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch

from multiprocessing import current_process, Queue 
from concurrent.futures import ThreadPoolExecutor
import subprocess

import time

polus_smp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polus-plugins/segmentation/polus-smp-training-plugin/")
sys.path.append(polus_smp_dir)

from src.utils import ENCODERS

epochs = 500
batchSize = 32

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("pipeline")
logger.setLevel("INFO")

NUM_GPUS = torch.cuda.device_count()
QUEUE = Queue()
encodervariant_dictionary = {variant:base for base in ENCODERS for variant in ENCODERS[base]}

def csv_rowprocess(model_parameters, headers, gpu_id, **kwargs):
    ident = current_process().ident
    try:
        model_dirname = "-".join(model_parameters)
        logger.info(f'{ident}: starting process on GPU {gpu_id} for {model_dirname}')
        
        model_dirpath = os.path.join(kwargs["output_workdir"], model_dirname)
        modelpth_path = os.path.join(model_dirpath, "model.pth")
        logpath = os.path.join(model_dirpath, "logs.log")

        num_arguments = len(headers)
        assert len(headers) == len(model_parameters)

        process_env = os.environ.copy()
        process_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        python_command =  f"time python {os.path.join(polus_smp_dir, "src/main.py")}" + \
                            f" --imagesTrainDir " + kwargs["imagesTrainDir"] + \
                            f" --labelsTrainDir " + kwargs["labelsTrainDir"] + \
                            f" --imagesValidDir " + kwargs["imagesValidDir"] + \
                            f" --labelsValidDir " + kwargs["labelsValidDir"] + \
                            f" --maxEpochs {epochs}" + \
                            f" --patience 30" + \
                            f" --batchSize {batchSize}" + \
                            f" --outputDir {model_dirpath}" + \
                            f" --device cuda:0" + \
                            f" --checkpointFrequency 5" + \
                            f" --minDelta .0001"
                            
        # additional arguments for the python command
        for argument_idx in range(num_arguments):
            if model_parameters[argument_idx] != "NA":
                python_command = python_command + f" --{headers[argument_idx]} {model_parameters[argument_idx]}"
                if headers[argument_idx] == "encoderVariant":
                    python_command = python_command + f" --encoderBase {encodervariant_dictionary[model_parameters[argument_idx]]}"
            else:
                continue
        
        if not os.path.exists(model_dirpath): # start it up
            os.mkdir(model_dirpath)
            logfile = open(logpath, 'w')
            logger.info(f"Starting up Process: {python_command}")
            subprocess.call(python_command, shell=True, stdout=logfile, stderr=logfile, env=process_env)
            
            # if the model did not get trained
            if not os.path.exists(modelpth_path):
                ErrorFile = os.path.join(model_dirpath, "ERROR")
                with open(ErrorFile, 'w') as errorfile:
                    pass
            else: # if the model did get trained
                # MOVE CHECKPOINT_FINAL AND MODEL_FINAL and delete all the checkpoints if subprocess run was successful
                checkpoint_dirpath = os.path.join(model_dirpath, "checkpoints")
                checkpoint_finalpath = os.path.join(checkpoint_dirpath, "checkpoint_final.pth")
                model_finalpath      = os.path.join(checkpoint_dirpath, "model_final.pth")
                if os.path.exists(checkpoint_finalpath) and os.path.exists(model_finalpath):
                    shutil.move(checkpoint_finalpath, model_dirpath)
                    shutil.move(model_finalpath, model_dirpath)
                if (os.path.exists(os.path.join(model_dirpath,"checkpoint_final.pth")) and \
                    os.path.exists(os.path.join(model_dirpath,"model_final.pth"))):
                    shutil.rmtree(checkpoint_dirpath)

        else:            
            if os.path.exists(modelpth_path):
                return 0
            checkpointpth_path = os.path.join(model_dirpath, "checkpoint.pth")
            if os.path.exists(checkpointpth_path):
                python_command = python_command + f" --pretrainedModel {model_dirpath}"
                logfile = open(logpath, 'w')
                subprocess.call(python_command, shell=True, stdout=logfile, stderr=logfile, env=process_env)
            else:
                logger.info(f"Trying again: {python_command}")
                logfile = open(logpath, 'w')
                subprocess.call(python_command, shell=True, stdout=logfile, stderr=logfile, env=process_env)
                
        logger.info(f"{ident}: ending process on GPU {gpu_id}")
        
    except Exception as e:
        logger.info(f"ERROR {e}")
    
    finally:
        QUEUE.put(gpu_id)

def main():

    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

    parser.add_argument('--csvFile', dest='csvFile', type=str, required=True, \
                        help='Path to csv File')
    parser.add_argument('--imagesTrainDir', dest='imagesTrainDir', type=str, required=True, \
                        help='Path to Images that are Trained')
    parser.add_argument('--labelsTrainDir', dest='labelsTrainDir', type=str, required=True, \
                        help='Path to Labels that are Trained')
    parser.add_argument('--imagesValidDir', dest='imagesValidDir', type=str, required=True, \
                        help='Path Images that are Validated')
    parser.add_argument('--labelsValidDir', dest='labelsValidDir', type=str, required=True, \
                        help='Path Labels that are Validated')
    parser.add_argument('--outputModels', dest='outputModels', type=str, required=True, \
                        help='Path to Output Directory')

    args = parser.parse_args()
    csv_path: str = args.csvFile
    main_file_path = args.mainFile
    images_training_dirpath = args.imagesTrainDir
    labels_training_dirpath = args.labelsTrainDir
    images_validation_dirpath = args.imagesValidDir
    labels_validation_dirpath = args.labelsValidDir
    output_models_dirpath = args.outputModels
    
    if not os.path.exists(output_models_dirpath):
        raise ValueError(f"Output Directory ({output_models_dirpath}) does not exist")
    
    logger.info(f"Input CSV Path : {csv_path}")
    logger.info(f"Images Training Directory : {images_training_dirpath}")
    logger.info(f"Labels Training Directory : {labels_training_dirpath}")
    logger.info(f"Images Validation Directory : {images_validation_dirpath}")
    logger.info(f"Labels Validation Directory : {labels_validation_dirpath}")
    logger.info(f"Output Models Directory : {output_models_dirpath}")
    
    input_kwargs = {"output_workdir" : output_models_dirpath,
                    "imagesTrainDir" : images_training_dirpath,
                    "labelsTrainDir" : labels_training_dirpath,
                    "imagesValidDir" : images_validation_dirpath,
                    "labelsValidDir" : labels_validation_dirpath}

    logger.info(f"\nOpening CSV file")
    csv_file = open(csv_path)
    csv_reader = csv.reader(csv_file)

    NUM_PROCESSES = len(list(csv_reader)) - 1

    logger.info(f"\nQueuing up {NUM_GPUS} GPUs ...")
    for gpu_ids in (range(NUM_GPUS)):
        logger.debug(f"queuing device {gpu_ids} - {torch.cuda.get_device_name(gpu_ids)}")
        QUEUE.put(str(gpu_ids)) # pass in what environments will be available for each subprocess

    csv_file.seek(0)
    headers = next(csv_reader)

    counter = 0
    with ThreadPoolExecutor(max_workers=NUM_GPUS+(NUM_GPUS/2)) as executor:
        for model_parameters in csv_reader:

            counter += 1
            
            model_dirname = "-".join(model_parameters)
            logger.info(f"\n{counter}. {model_dirname}")
            
            model_dirpath = os.path.join(input_kwargs["output_workdir"], model_dirname)
            modelpth_path = os.path.join(model_dirpath, "model.pth")
            if os.path.exists(modelpth_path):
                logger.debug(f"Not Running ({counter}/{NUM_PROCESSES}) - output already exists at {modelpth_path}")
            
            sleeping_in = 0
            while QUEUE.empty():
                sleeping_in += 1
                time.sleep(30)
                logger.debug(f"There are currently no available GPUS to use - sleeping in x{sleeping_in}")
        
            if not QUEUE.empty():
                executor.submit(csv_rowprocess, model_parameters, headers, QUEUE.get(), **input_kwargs)
            
            logger.info(f"analyzed {counter}/{NUM_PROCESSES} models")
        logger.info(f"DONE ANALYZING ALL MODELS!")

    
main()
