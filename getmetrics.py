import os,sys,json
import copy
# os.environ["CUDA_VISIBLE_DEVICES"] = "0", "1", "2", "3", "4", "6", "7"
import shutil

import numpy as np

from filepattern import FilePattern 

import tempfile

import torch
import torchvision

from multiprocessing import Pool, current_process, Queue 
import subprocess
import concurrent.futures

import bfio
from bfio import BioWriter

polus_dir = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/polus-plugins/segmentation/polus-smp-training-plugin/"
sys.path.append(polus_dir)

from src.utils import Dataset
from src.utils import get_labels_mapping
from src.training import initialize_dataloader
from src.training import MultiEpochsDataLoader
from src.utils import METRICS
from src.utils import LOSSES

import time

"""INPUT PARAMETERS"""

# checkme : outputpath
labelled_groundtruth = "/home/vihanimm/SegmentationModelToolkit/Data/ometif_data/nuclear/test/groundtruth/"
groundtruth_basepath = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ftl"
prediction_basepath  = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ftl"
ftl_basepath         = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/ftl_outputs"

# groundtruth_dir = "groundtruths"
# predictions_dir = "predictions"
# ftl_dir         = "ftl"

output_basepath = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_metrics"

"""MAKING PREDICTIONS"""

queue = Queue()
def evaluation(pr_collection,
               gt_collection,
               ftl_collection,
               pixel_output_model_path,
               cell_output_model_path):
    
    try:

        celllogfile = os.path.join(cell_output_model_path, "logs.log")
        cellresultfile = os.path.join(cell_output_model_path, "result.csv")
        total_stats_cell = os.path.join(cell_output_model_path, "total_stats_result.csv")
        # if os.path.exists(celllogfile):
        #     os.remove(celllogfile)
        # if os.path.exists(cellresultfile):
        #     os.remove(cellresultfile)

        
        if not (os.path.exists(total_stats_cell) and os.path.exists(cellresultfile)):
            pythoncell_command = "python /home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/polus-plugins/features/polus-cellular-evaluation-plugin/src/main.py" + \
                            f" --GTDir {labelled_groundtruth}" + \
                            f" --PredDir {ftl_collection}" + \
                            f" --inputClasses 1" + \
                            f" --totalStats true" + \
                            f" --totalSummary true" + \
                            f" --outDir {cell_output_model_path}" 

            celllogfile = open(celllogfile, 'a')
            subprocess.call(pythoncell_command, shell=True, stdout=celllogfile, stderr=celllogfile)
        
        
        pixellogfile = os.path.join(pixel_output_model_path, "logs.log")
        pixelresultfile = os.path.join(pixel_output_model_path, "result.csv")
        total_stats_pixel = os.path.join(pixel_output_model_path, "total_stats_result.csv")
        # if os.path.exists(pixellogfile):
        #     os.remove(pixellogfile)
        # if os.path.exists(pixelresultfile):
        #     os.remove(pixelresultfile)
        
        if not (os.path.exists(total_stats_pixel) and os.path.exists(pixelresultfile)):
            pythonpixel_command = "python /home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/polus-plugins/features/polus-pixelwise-evaluation-plugin/src/main.py" + \
                            f" --GTDir {gt_collection}" + \
                            f" --PredDir {pr_collection}" + \
                            f" --inputClasses 1" + \
                            f" --individualStats False" + \
                            f" --totalStats True" + \
                            f" --outDir {pixel_output_model_path}"
        
            pixellogfile = open(pixellogfile, 'a')
            subprocess.call(pythonpixel_command, shell=True, stdout=pixellogfile, stderr=pixellogfile)
        

        print(f"SAVED: {pixel_output_model_path} , {cell_output_model_path}")
        
    except Exception as e:
        print(e)

counter = 1
sleeping_in = 0


models_list = os.listdir("/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP")
num_models = len(models_list)
with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-10) as executor:

    for smp_model in models_list:
        print(f"analyzing {smp_model} : {counter}/{num_models}")
        model_gt_dir  = os.path.join(groundtruth_basepath, smp_model)
        model_pr_dir  = os.path.join(prediction_basepath, smp_model)
        model_ftl_dir = os.path.join(ftl_basepath, smp_model)
    
        pr_collection = os.path.join(model_pr_dir, "predictions") # this is where images get saved to
        gt_collection = os.path.join(model_gt_dir, "groundtruths")
        ftl_collection = os.path.join(model_ftl_dir, "ftl")
    
        # print(pr_collection)
        # print(gt_collection)
        # print(ftl_collection)
        
        output_model_metrics = os.path.join(output_basepath, smp_model)
        if not os.path.exists(output_model_metrics):
            os.mkdir(output_model_metrics)
        
        pixel_output_model_path = os.path.join(output_model_metrics, "pixeleval")
        if not os.path.exists(pixel_output_model_path):
            os.mkdir(pixel_output_model_path)
            
        cell_output_model_path  = os.path.join(output_model_metrics, "celleval")
        if not os.path.exists(cell_output_model_path):
            os.mkdir(cell_output_model_path)
        
        total_stats_cell = os.path.join(cell_output_model_path, "total_stats_result.csv")
        total_stats_pixel = os.path.join(pixel_output_model_path, "total_stats_result.csv")
        if os.path.exists(total_stats_cell) and os.path.exists(total_stats_pixel):
            print(f"already have outputs saved in {total_stats_cell} , {total_stats_pixel}")
            counter = counter + 1
            continue

        if not os.path.exists(pixel_output_model_path):
            os.makedirs(pixel_output_model_path)
        
        if not os.path.exists(cell_output_model_path):
            os.makedirs(cell_output_model_path)

        
        skip_model = False
        test_dirs = [pr_collection, gt_collection, ftl_collection]
        for test_dir in test_dirs:
            if not os.path.exists(test_dir):
                print(f"\t{test_dir} does not exist")
                skip_model = True
                break
            if len(os.listdir(test_dir)) < 1249:
                print(f"\t{test_dir} has only {len(os.listdir(test_dir))} images")
                skip_model = True
                break
            
        if skip_model:
            counter += 1
            continue
        
        print(f"SUBMIT: {pixel_output_model_path} , {cell_output_model_path}")
        executor.submit(evaluation, 
                        pr_collection = pr_collection, 
                        gt_collection = gt_collection, 
                        ftl_collection = ftl_collection, 
                        pixel_output_model_path = pixel_output_model_path,
                        cell_output_model_path = cell_output_model_path)
        counter = counter + 1
        print("\n")
print("DONE WITH ALL THE MODELS :)")