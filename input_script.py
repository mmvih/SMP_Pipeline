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
import concurrent.futures
import subprocess

import time

polus_smp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polus-plugins/segmentation/polus-smp-training-plugin/")
sys.path.append(polus_smp_dir)

from src.utils import ENCODERS

# available_gpus
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

epochs = 500
batchSize = 32

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("pipeline")
logger.setLevel("INFO")

encodervariant_dictionary = {variant:base for base in ENCODERS for variant in ENCODERS[base]}

queue = Queue()

def create_plots(outputdirectory, traincsv, validcsv):
    
    score_dictionary = {}
    with open(traincsv, 'r') as train:
        with open(validcsv, 'r') as valid:
            for t_line, v_line in zip(train, valid):
                split_t = t_line.rstrip("\n").split(",")
                split_v = v_line.rstrip("\n").split(",")
                for scores_t, scores_v in zip(split_t, split_v):
                    Train_name, Train_value = scores_t.split(":")
                    Valid_name, Valid_value = scores_v.split(":")
                    Train_name = Train_name.strip()
                    Valid_name = Valid_name.strip()
                    assert Train_name == Valid_name
                    
                    if Train_name in score_dictionary:
                        score_dictionary[Train_name][0].append(float(Train_value))
                        score_dictionary[Valid_name][1].append(float(Valid_value))
                    else:
                        score_dictionary[Train_name] = \
                            [[float(Train_value)], [float(Valid_value)]]

    score_keys = score_dictionary.keys()
    fig, ax = plt.subplots(2, 3, figsize=(15,12))
    fig.tight_layout(pad=3)
    i = 0
    j = 0
    fig.suptitle(os.path.basename(outputdirectory))
    for score_name in score_dictionary.keys():
        ax[i,j].set_ylim([0,1])
        ax[i,j].set_xlabel("EPOCHS")
        ax[i,j].set_ylabel(score_name)
        ax[i,j].plot(score_dictionary[score_name][0], label="Train")
        ax[i,j].plot(score_dictionary[score_name][1], label="Validation")
        i = i + 1
        if i == 2:
            i = 0
            j = j + 1
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.savefig(os.path.join(outputdirectory, f"Logs.jpg"))
    plt.clf()

def csv_rowprocess(row, headers, **kwargs):
    gpu_id = queue.get()
    ident = current_process().ident
    try:
        logger.info('{}: starting process on GPU {}'.format(ident, gpu_id))
        model_directory = f"{row[0]}"
        for parameter in row[1:]:         
            if parameter == "NA":
                model_directory = model_directory + "-" + "NA"
            else:
                model_directory = model_directory + "-" + parameter
        
        if not os.path.exists(kwargs["output_workdir"]):
            exit() # output directory does not exist
        newdirectory_formodel = os.path.join(kwargs["output_workdir"], model_directory)
        logpath = os.path.join(newdirectory_formodel, "logs.log")

        process_env = os.environ.copy()
        process_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        docker_container =  f"time python " + kwargs["python_main"] + \
                            f" --imagesTrainDir " + kwargs["imagesTrainDir"] + \
                            f" --labelsTrainDir " + kwargs["labelsTrainDir"] + \
                            f" --imagesValidDir " + kwargs["imagesValidDir"] + \
                            f" --labelsValidDir " + kwargs["labelsValidDir"] + \
                            f" --maxEpochs {epochs}" + \
                            f" --patience 30" + \
                            f" --batchSize {batchSize}" + \
                            f" --outputDir {newdirectory_formodel}" + \
                            f" --device cuda:0" + \
                            f" --checkpointFrequency 5" + \
                            f" --minDelta .0001"
                            # " --trainPattern cell_train_6{xx}.tif" + \
                            # " --validPattern cell_validation_6{xx}.tif"
        num_arguments = len(headers)
        model_file = os.path.join(newdirectory_formodel, "model.pth")
        assert len(headers) == len(row)
        for argument_idx in range(num_arguments):
            if row[argument_idx] != "NA":
                docker_container = docker_container + f" --{headers[argument_idx]} {row[argument_idx]}"
                if headers[argument_idx] == "encoderVariant":
                    docker_container = docker_container + f" --encoderBase {encodervariant_dictionary[row[argument_idx]]}"
            else:
                continue
        if not os.path.exists(newdirectory_formodel):
            os.makedirs(newdirectory_formodel)
            logfile = open(logpath, 'a')
            logger.info(f"Starting up Process: {docker_container}", flush=True)
            subprocess.call(docker_container, shell=True, stdout=logfile, stderr=logfile, env=process_env)
            if not os.path.exists(model_file):
                ErrorFile = os.path.join(newdirectory_formodel, "ERROR")
                with open(ErrorFile, 'w') as errorfile:
                    pass
            else:
                # MOVE CHECKPOINT_FINAL AND MODEL_FINAL and delete all the checkpoints if subprocess run was successful
                checkpoint_dirpath = os.path.join(newdirectory_formodel, "checkpoints")
                checkpoint_finalpath = os.path.join(checkpoint_dirpath, "checkpoint_final.pth")
                model_finalpath      = os.path.join(checkpoint_dirpath, "model_final.pth")
                if os.path.exists(checkpoint_finalpath) and \
                    os.path.exists(model_finalpath):
                    shutil.move(checkpoint_finalpath, newdirectory_formodel)
                    shutil.move(model_finalpath, newdirectory_formodel)
                if (os.path.exists(os.path.join(newdirectory_formodel,"checkpoint_final.pth")) and \
                    os.path.exists(os.path.join(newdirectory_formodel,"model_final.pth"))):
                    shutil.rmtree(checkpoint_dirpath)
        else:            
            if os.path.exists(model_file):
                return 0
            checkpoint_file = os.path.join(newdirectory_formodel, "checkpoint.pth")
            if os.path.exists(checkpoint_file):
                docker_container = docker_container + f" --pretrainedModel {newdirectory_formodel}"
                logfile = open(logpath, 'a')
                subprocess.call(docker_container, shell=True, stdout=logfile, stderr=logfile, env=process_env)
            else:
                logger.info(f"Trying again: {docker_container}", flush=True)
                logfile = open(logpath, 'a')
                subprocess.call(docker_container, shell=True, stdout=logfile, stderr=logfile, env=process_env)
                
        # trainlogs = os.path.join(newdirectory_formodel, "trainlogs.csv")
        # validlogs = os.path.join(newdirectory_formodel, "validlogs.csv")
        # if os.path.isfile(trainlogs) and os.path.isfile(validlogs):
        #     create_plots(newdirectory_formodel, trainlogs, validlogs)
        logger.info('{}: ending process on GPU {}'.format(ident, gpu_id))
    except Exception as e:
        logger.info(f"ERROR {e}")
    finally:
        queue.put(gpu_id)

track_changes = True
def main():
    try:
        """ Argument parsing """
        logger.info("Parsing arguments...")
        parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

        parser.add_argument('--csvFile', dest='csvFile', type=str, required=True, \
                            help='Path to csv File')
        parser.add_argument('--outputModels', dest='outputModels', type=str, required=True, \
                            help='Path to Output Directory')
        parser.add_argument('--MainFile', dest='mainFile', type=str, required=True, \
                            help='Path to Training SMP file')
        parser.add_argument('--imagesTrainDir', dest='imagesTrainDir', type=str, required=True, \
                            help='Path to Images that are Trained')
        parser.add_argument('--labelsTrainDir', dest='labelsTrainDir', type=str, required=True, \
                            help='Path to Labels that are Trained')
        parser.add_argument('--imagesValidDir', dest='imagesValidDir', type=str, required=True, \
                            help='Path Images that are Validated')
        parser.add_argument('--labelsValidDir', dest='labelsValidDir', type=str, required=True, \
                            help='Path Labels that are Validated')

        args = parser.parse_args()
        csvFile: str = args.csvFile
        
        input_kwargs = {"output_workdir" : args.outputModels,
                        "python_main"    : args.mainFile,
                        "imagesTrainDir" : args.imagesTrainDir,
                        "labelsTrainDir" : args.labelsTrainDir,
                        "imagesValidDir" : args.imagesValidDir,
                        "labelsValidDir" : args.labelsValidDir}


        csv_file = open(csvFile)
        csv_reader = csv.reader(csv_file)

        NUM_GPUS = len(available_gpus)
        NUM_PROCESSES = int(len(list(csv_reader))) - 1
        PROC_PER_GPU = int(np.ceil(NUM_PROCESSES/NUM_GPUS))

        for gpu_ids in tqdm(range(5)):
            queue.put(str(gpu_ids))

        csv_file.seek(0)
        headers = next(csv_reader)

        counter = 0
        sleeping_in = 0     
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            while counter != NUM_PROCESSES:
                if not queue.empty():
                    sleeping_in = 0
                    row = next(csv_reader)
                    
                    model_dirname = "-".join(row)
                    model_dir     = os.path.join(input_kwargs["output_workdir"], model_dirname)
                    modelpth_path = os.path.join(model_dir, "model.pth")

                    if os.path.exists(modelpth_path):
                        counter = counter + 1
                        continue

                    executor.submit(csv_rowprocess, row, headers, **input_kwargs)
                    counter = counter + 1
                else:
                    sleeping_in += 1
                    time.sleep(45)
                    continue


    except Exception as e:
        print(f"ERROR: {e}")
    

main()
