import os, sys
import csv
import logging, argparse
# import pandas
import itertools
from itertools import repeat

import shutil

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from multiprocessing import Pool, current_process, Queue 
import torch
import subprocess

import concurrent.futures
import time

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

epochs = 500
batchSize = 32


logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("pipeline")
logger.setLevel("INFO")

encodervariant_dictionary = {
 'resnet18': 'ResNet', 'resnet34': 'ResNet', 'resnet50': 'ResNet', 
 'resnet101': 'ResNet', 'resnet152': 'ResNet', 
 
 'resnext50_32x4d': 'ResNeXt', 'resnext101_32x4d': 'ResNeXt', 'resnext101_32x8d': 'ResNeXt', 
 'resnext101_32x16d': 'ResNeXt', 'resnext101_32x32d': 'ResNeXt', 'resnext101_32x48d': 'ResNeXt',
 'timm-res2net50_26w_4s': 'Res2NeXt', 'timm-res2net101_26w_4s': 'Res2NeXt', 'timm-res2net50_26w_6s': 'Res2NeXt', 
 'timm-res2net50_26w_8s': 'Res2NeXt', 'timm-res2net50_48w_2s': 'Res2NeXt', 'timm-res2net50_14w_8s': 'Res2NeXt', 'timm-res2next50': 'Res2NeXt',
 
 'timm-resnest14d': 'ResNeSt', 'timm-resnest26d': 'ResNeSt', 'timm-resnest50d': 'ResNeSt', 
 'timm-resnest101e': 'ResNeSt', 'timm-resnest200e': 'ResNeSt', 'timm-resnest269e': 'ResNeSt', 
 'timm-resnest50d_4s2x40d': 'ResNeSt', 'timm-resnest50d_1s4x24d': 'ResNeSt', 
 
 'timm-regnetx_002': 'RegNetx/y', 'timm-regnetx_004': 'RegNetx/y', 'timm-regnetx_006': 'RegNetx/y', 'timm-regnetx_008': 'RegNetx/y', 
 'timm-regnetx_016': 'RegNetx/y', 'timm-regnetx_032': 'RegNetx/y', 'timm-regnetx_040': 'RegNetx/y', 'timm-regnetx_064': 'RegNetx/y', 
 'timm-regnetx_080': 'RegNetx/y', 'timm-regnetx_120': 'RegNetx/y', 'timm-regnetx_160': 'RegNetx/y', 'timm-regnetx_320': 'RegNetx/y', 
 'timm-regnety_002': 'RegNetx/y', 'timm-regnety_004': 'RegNetx/y', 'timm-regnety_006': 'RegNetx/y', 'timm-regnety_008': 'RegNetx/y', 
 'timm-regnety_016': 'RegNetx/y', 'timm-regnety_032': 'RegNetx/y', 'timm-regnety_040': 'RegNetx/y', 'timm-regnety_064': 'RegNetx/y', 
 'timm-regnety_080': 'RegNetx/y', 'timm-regnety_120': 'RegNetx/y', 'timm-regnety_160': 'RegNetx/y', 'timm-regnety_320': 'RegNetx/y', 
 
 'timm-gernet_s': 'GERNet', 'timm-gernet_m': 'GERNet', 'timm-gernet_l': 'GERNet', 
 
 'senet154': 'SE-Net', 'se_resnet50': 'SE-Net', 'se_resnet101': 'SE-Net', 'se_resnet152': 'SE-Net', 'se_resnext50_32x4d': 'SE-Net', 
 'se_resnext101_32x4d': 'SE-Net', 
 
 'timm-skresnet18': 'SK-ResNeXt', 'timm-skresnet34': 'SK-ResNeXt', 'timm-skresnext50_32x4d': 'SK-ResNeXt', 
 
 'densenet121': 'DenseNet', 'densenet169': 'DenseNet', 'densenet201': 'DenseNet', 'densenet161': 'DenseNet', 
 
 'inceptionresnetv2': 'Inception', 'inceptionv4': 'Inception', 'xception': 'Inception', 
 
 'efficientnet-b0': 'EfficientNet', 'efficientnet-b1': 'EfficientNet', 'efficientnet-b2': 'EfficientNet', 'efficientnet-b3': 'EfficientNet', 
 'efficientnet-b4': 'EfficientNet', 'efficientnet-b5': 'EfficientNet', 'efficientnet-b6': 'EfficientNet', 'efficientnet-b7': 'EfficientNet', 
 'timm-efficientnet-b0': 'EfficientNet', 'timm-efficientnet-b1': 'EfficientNet', 'timm-efficientnet-b2': 'EfficientNet', 
 'timm-efficientnet-b3': 'EfficientNet', 'timm-efficientnet-b4': 'EfficientNet', 'timm-efficientnet-b5': 'EfficientNet', 
 'timm-efficientnet-b6': 'EfficientNet', 'timm-efficientnet-b7': 'EfficientNet', 'timm-efficientnet-b8': 'EfficientNet', 
 'timm-efficientnet-l2': 'EfficientNet', 
 'timm-tf_efficientnet_lite0': 'EfficientNet', 'timm-tf_efficientnet_lite1': 'EfficientNet', 
 'timm-tf_efficientnet_lite2': 'EfficientNet', 'timm-tf_efficientnet_lite3': 'EfficientNet', 
 'timm-tf_efficientnet_lite4': 'EfficientNet', 
 
 'mobilenet_v2': 'MobileNet', 
 'timm-mobilenetv3_large_075': 'MobileNet', 'timm-mobilenetv3_large_100': 'MobileNet', 'timm-mobilenetv3_large_minimal_100': 'MobileNet', 
 'timm-mobilenetv3_small_075': 'MobileNet', 'timm-mobilenetv3_small_100': 'MobileNet', 'timm-mobilenetv3_small_minimal_100': 'MobileNet', 
 
 'dpn68': 'DPN', 'dpn68b': 'DPN', 'dpn92': 'DPN', 'dpn98': 'DPN', 'dpn107': 'DPN', 'dpn131': 'DPN', 
 
 'vgg11': 'VGG', 'vgg11_bn': 'VGG', 'vgg13': 'VGG', 'vgg13_bn': 'VGG', 
 'vgg16': 'VGG', 'vgg16_bn': 'VGG', 'vgg19': 'VGG', 'vgg19_bn': 'VGG',
 
"efficientnetv2_l" : 'TIMM',
"efficientnetv2_l_in21k_tf": 'TIMM',
"efficientnetv2_m": 'TIMM',
"efficientnetv2_m_in21k_tf": 'TIMM',
"efficientnetv2_s": 'TIMM',
"efficientnetv2_s_in21k_tf" : 'TIMM',
"mixnet_l": 'TIMM',
"mixnet_m": 'TIMM',
"mixnet_s": 'TIMM',
"mixnet_xl": 'TIMM',
"mixnet_xxl": 'TIMM',
"nfnet_f0": 'TIMM',
"nfnet_f1": 'TIMM',
"nfnet_f2": 'TIMM',
"nfnet_f3": 'TIMM',
"nfnet_f4": 'TIMM',
"nfnet_f5": 'TIMM',
"nfnet_f6": 'TIMM',
"nfnet_f7": 'TIMM',
"nfnet_l0_eca": 'TIMM',
"nfnet_l1_eca": 'TIMM',
"nfnet_l2_eca": 'TIMM',
"nfnet_l3_eca": 'TIMM',
"swinnet26t_256": 'TIMM',
"swinnet50ts_256": 'TIMM',
"swinnext26ts_256_eca": 'TIMM'}

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
                    # docker_container = docker_container + f" --encoderBase TIMM"
                    docker_container = docker_container + f" --encoderBase {encodervariant_dictionary[row[argument_idx]]}"
            else:
                continue
        if not os.path.exists(newdirectory_formodel):
            os.makedirs(newdirectory_formodel)
            logfile = open(logpath, 'a')
            print(f"Starting up Process: {docker_container}", flush=True)
            subprocess.call(docker_container, shell=True, stdout=logfile, stderr=logfile, env=process_env)
            if not os.path.exists(model_file):
                ErrorFile = os.path.join(newdirectory_formodel, "ERROR")
                with open(ErrorFile, 'w') as errorfile:
                    pass
            else:
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
                print(f"Trying again: {docker_container}", flush=True)
                logfile = open(logpath, 'a')
                subprocess.call(docker_container, shell=True, stdout=logfile, stderr=logfile, env=process_env)
                
        # trainlogs = os.path.join(newdirectory_formodel, "trainlogs.csv")
        # validlogs = os.path.join(newdirectory_formodel, "validlogs.csv")
        # if os.path.isfile(trainlogs) and os.path.isfile(validlogs):
        #     create_plots(newdirectory_formodel, trainlogs, validlogs)
        logger.info('{}: ending process on GPU {}'.format(ident, gpu_id))
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
        print(e)
    

main()
