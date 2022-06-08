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

base_dir = "/home/vihanimm/SegmentationModels/SMP_Pipeline"
polus_dir = f"{base_dir}/polus-plugins/segmentation/polus-smp-training-plugin/"
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
smp_inputs_path =  f"{base_dir}/cytoplasm_models"
smp_outputs_path = f"{base_dir}/cytoplasm_models_testoutputs"

testing_images = "/home/vihanimm/Data/tissuenet/tif_data/cell/test/image/"
testing_labels = "/home/vihanimm/Data/tissuenet/tif_data/cell/test/groundtruth_centerbinary_2pixelsmaller/"

smp_inputs_list = os.listdir(smp_inputs_path)


def getLoader(images_Dir, 
              labels_Dir):
    
    # checkme : filepattern
    filepattern = ".*"
    # filepattern = "nuclear_test_60{x}.tif"
    images_fp = FilePattern(testing_images, filepattern)
    labels_fp = FilePattern(testing_labels, filepattern)

    image_array, label_array, names = get_labels_mapping(images_fp(), labels_fp(), provide_names=True)

    testing_dataset = Dataset(images=image_array,
                              labels=label_array)
    testing_loader = MultiEpochsDataLoader(testing_dataset, num_workers=4, batch_size=10, shuffle=True, pin_memory=True, drop_last=True)
    
    testing_dataset_vis = Dataset(images=image_array,
                                    labels=label_array,
                                    preprocessing=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    testing_loader_vis = MultiEpochsDataLoader(testing_dataset_vis, num_workers=4, batch_size=10, shuffle=True, pin_memory=True, drop_last=True)
        
    return testing_dataset, testing_dataset_vis, images_fp, labels_fp, names


print("Getting Loaders", flush=True)
test_loader, test_loader_vis, images_fp, labels_fp, names = getLoader(images_Dir=testing_images,
                                            labels_Dir=testing_labels)
print("Done with Loaders", flush=True)
test_loader_len = torch.tensor(len(test_loader))


"""MAKING PREDICTIONS"""

queue = Queue()
def evaluation(smp_model : str, 
               cuda_num : str):
    
    try:
        smp_model_dirpath = os.path.join(smp_inputs_path, smp_model)
        print("LOOKING AT: ", smp_model_dirpath, flush=True)
        
        modelpth_path = os.path.join(smp_model_dirpath, "model.pth")
        
        output_path = os.path.join(smp_outputs_path, smp_model)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        tor_device = torch.device(f"cuda:{cuda_num}")
        model = torch.load(modelpth_path, map_location=tor_device)
        
        pr_collection = os.path.join(output_path, "predictions") # this is where images get saved to
        # gt_collection = os.path.join(output_path, "groundtruths")
        
        
        if not os.path.exists(pr_collection):
            os.mkdir(pr_collection)
        # if not os.path.exists(gt_collection):
        #     os.mkdir(gt_collection)
        
        img_count = 0
        print(f"Generating predictions for {smp_model} : saving in {output_path}", flush=True)
        for im, gt in test_loader:
            im_tensor = torch.from_numpy(im).to(tor_device).unsqueeze(0)
            pr_tensor = model.predict(im_tensor)
            
            gt = gt.squeeze()[..., None, None, None]

            pr = pr_tensor.cpu().detach().numpy().squeeze()[..., None, None, None]
            pr[pr >= .50] = 1
            pr[pr < .50] = 0
            
            assert gt.shape == pr.shape
            
            filename = names[img_count][:-4] + ".ome.tif"
            pr_filename = os.path.join(pr_collection, filename)
            # gt_filename = os.path.join(gt_collection, filename)
            
            with BioWriter(pr_filename, Y=pr.shape[0],
                                        X=pr.shape[1],
                                        Z=1,
                                        C=1,
                                        T=1,
                                        dtype=pr.dtype) as bw_pr:
                bw_pr[:] = pr
                
            # with BioWriter(gt_filename, Y=gt.shape[0],
            #                             X=gt.shape[1],
            #                             Z=1,
            #                             C=1,
            #                             T=1,
            #                             dtype=gt.dtype) as bw_gt:
            #     bw_gt[:] = gt
            
            img_count = img_count + 1
            
        
        queue.put(cuda_num)
            
    except Exception as e:
        queue.put(cuda_num)
        print(e)
            
NUM_GPUS = 8
NUM_PROCESSES = len(smp_inputs_list)
PROC_PER_GPU = int(np.ceil(NUM_PROCESSES/NUM_GPUS))


for gpu_ids in (range(8)):
    print(gpu_ids, flush=True)
    print(gpu_ids)
    queue.put(str(gpu_ids))
print("Queued up GPUs", flush=True)

iter_smp_inputs_list = iter(smp_inputs_list)

num_models = len(smp_inputs_list)
counter = 0
sleeping_in = 0     
# checkme : max workers
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    while counter != len(smp_inputs_list)-1:
        if not queue.empty():
            sleeping_in = 0
            
            curr_smp_model = next(iter_smp_inputs_list)
            
            smp_model_dirpath = os.path.join(smp_inputs_path, curr_smp_model)
            
            ERROR_path = os.path.join(smp_model_dirpath, "ERROR")
            if os.path.exists(ERROR_path):
                print(f"Not Running - ERROR Exists {smp_model_dirpath}", flush=True)
                counter += 1
                continue
            
            modelpth_path = os.path.join(smp_model_dirpath, "model.pth")
            if not os.path.exists(modelpth_path):
                counter += 1
                print(f"Not Running - model.pth does not Exists {smp_model_dirpath}", flush=True)
                continue
            
            
            output_path = os.path.join(smp_outputs_path, curr_smp_model)
            pr_collection = os.path.join(output_path, "predictions") # this is where images get saved to
            gt_collection = os.path.join(output_path, "groundtruths")
 
            if os.path.exists(pr_collection) and os.path.exists(gt_collection):
                if len(os.listdir(pr_collection)) == 1249 and len(os.listdir(gt_collection)) == 1249:
                    counter += 1
                    print(f"Not Running - already has outputs")
                    continue
 
            
            executor.submit(evaluation, curr_smp_model, queue.get())
            counter = counter + 1
            print(f"analyzed {counter}/{num_models-1} models", flush=True)

        else:
            sleeping_in += 1
            print(f"sleeping in : x{sleeping_in}", flush=True)
            time.sleep(5)
            continue
