import os, sys
import csv
import logging, argparse
# import pandas
import itertools
from itertools import repeat

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from multiprocessing import Pool, current_process, Queue 
import torch
import subprocess

import concurrent.futures


available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

epochs = 15
batchSize = 4


logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("pipeline")
logger.setLevel("INFO")


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
    for score_name in score_dictionary.keys():
        plt.plot(score_dictionary[score_name][0])
        plt.plot(score_dictionary[score_name][1])
        plt.legend(["Train", "Validation"])
        plt.title(score_name)
        plt.xlabel("EPOCHS")
        plt.ylabel(score_name)
        plt.savefig(os.path.join(outputdirectory, f"{score_name}.jpg"))
        plt.clf()
                    
def csv_rowprocess(row, headers, **kwargs):
    print("INPUT ROW: ", row)
    print(kwargs)
    gpu_id = queue.get()
    ident = current_process().ident
    try:
        print('{}: starting process on GPU {}'.format(ident, gpu_id))
        model_directory = f"{row[0]}"
        for parameter in row[1:]:         
            if parameter == "NA":
                # newdirectory_formodel = os.path.join(newdirectory_formodel, "NA")
                model_directory = model_directory + "-" + "NA"
            else:
            #     newdirectory_formodel = os.path.join(newdirectory_formodel, parameter)
                model_directory = model_directory + "-" + parameter
        newdirectory_formodel = os.path.join(kwargs["output_workdir"], model_directory)
        docker_container =  f"python " + kwargs["python_main"] + \
                            f" --imagesTrainDir " + kwargs["imagesTrainDir"] + \
                            f" --labelsTrainDir " + kwargs["labelsTrainDir"] + \
                            f" --imagesValidDir " + kwargs["imagesValidDir"] + \
                            f" --labelsValidDir " + kwargs["labelsValidDir"] + \
                            f" --maxEpochs {epochs}" + \
                            f" --batchSize {batchSize}" + \
                            f" --outputDir {newdirectory_formodel}" + \
                            f" --device cuda:{gpu_id}"
        num_arguments = len(headers)
        assert len(headers) == len(row)
        for argument_idx in range(num_arguments):
            if row[argument_idx] != "NA":
                docker_container = docker_container + f" --{headers[argument_idx]} {row[argument_idx]}"
            else:
                continue
        if not os.path.exists(newdirectory_formodel):
            os.makedirs(newdirectory_formodel)
            subprocess.call(docker_container, shell=True)
        else:
            model_file = os.path.join(newdirectory_formodel, "model.pth")
            if os.path.exists(model_file):
                return 0
            checkpoint_file = os.path.join(newdirectory_formodel, "checkpoint.pth")
            if os.path.exists(checkpoint_file):
                docker_container = docker_container + f" --pretrainedModel {newdirectory_formodel}"
                subprocess.call(docker_container, shell=True)
        trainlogs = os.path.join(newdirectory_formodel, "trainlogs.csv")
        validlogs = os.path.join(newdirectory_formodel, "validlogs.csv")
        if os.path.isfile(trainlogs) and os.path.isfile(validlogs):
            create_plots(newdirectory_formodel, trainlogs, validlogs)
        print('{}: ending process on GPU {}'.format(ident, gpu_id))
    except Exception as e:
        print(e)
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
        print("Done GPU calcs...")

        for gpu_ids in tqdm(range(NUM_GPUS)):
            queue.put(gpu_ids)

        csv_file.seek(0)
        headers = next(csv_reader)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for row in csv_reader:
                executor.submit(csv_rowprocess, row, headers, **input_kwargs)

    except Exception as e:
        print(e)

main()