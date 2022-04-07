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

"""INPUT PARAMETERS"""

smp_inputs_path =  "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP/"
smp_outputs_path = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_pixeleval/"

testing_images = "/home/vihanimm/SegmentationModelToolkit/Data/tif_data/nuclear/test/image/"
testing_labels = "/home/vihanimm/SegmentationModelToolkit/Data/tif_data/nuclear/test/groundtruth_centerbinary_2pixelsmaller/"

smp_inputs_list = os.listdir(smp_inputs_path)

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]       

def getLoader(images_Dir, 
              labels_Dir):
    
    filepattern = ".*"
    images_fp = FilePattern(testing_images, filepattern)
    labels_fp = FilePattern(testing_labels, filepattern)

    image_array, label_array = get_labels_mapping(images_fp(), labels_fp())

    testing_dataset = Dataset(images=image_array,
                              labels=label_array)
    testing_loader = MultiEpochsDataLoader(testing_dataset, num_workers=4, batch_size=10, shuffle=True, pin_memory=True, drop_last=True)
    
    testing_dataset_vis = Dataset(images=image_array,
                                    labels=label_array,
                                    preprocessing=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor()]))
    testing_loader_vis = MultiEpochsDataLoader(testing_dataset_vis, num_workers=4, batch_size=10, shuffle=True, pin_memory=True, drop_last=True)
        
    return testing_dataset, testing_dataset_vis, images_fp, labels_fp



"""MAKING PREDICTIONS"""

queue = Queue()
def evaluation(smp_model : str, 
               cuda_num : str):
    
    print("Getting Loaders")
    test_loader, test_loader_vis, images_fp, labels_fp = getLoader(images_Dir=testing_images,
                                                labels_Dir=testing_labels)
    print("Done with Loaders")
    test_loader_len = torch.tensor(len(test_loader))
    
    tor_device = torch.device(f"cuda:{cuda_num}")
    smp_model_dirpath = os.path.join(smp_inputs_path, smp_model)
    
    ERROR_path = os.path.join(smp_model_dirpath, "ERROR")
    if os.path.exists(ERROR_path):
        return
    
    modelpth_path = os.path.join(smp_model_dirpath, "model.pth")
    if not os.path.exists(modelpth_path):
        return
    model = torch.load(modelpth_path, map_location=tor_device)
    
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        pr_collection = os.path.join(temp_dir, "predictions") # this is where images get saved to
        gt_collection = os.path.join(temp_dir, "groundtruths")
        output_path = os.path.join(smp_outputs_path, smp_model)
        
        if not os.path.exists(pr_collection):
            os.mkdir(pr_collection)
        if not os.path.exists(gt_collection):
            os.mkdir(gt_collection)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        img_count = 0
        for im, gt in test_loader:
            im_tensor = torch.from_numpy(im).to(tor_device).unsqueeze(0)
            pr_tensor = model.predict(im_tensor)
            
            gt = gt.squeeze()[..., None, None, None]
            pr = pr_tensor.cpu().detach().numpy().squeeze()[..., None, None, None]
            
            assert gt.shape == pr.shape
            
            filename = f"Image_{img_count}.tif"
            pr_filename = os.path.join(pr_collection, filename)
            gt_filename = os.path.join(gt_collection, filename)
            
            with BioWriter(pr_filename, Y=pr.shape[0],
                                        X=pr.shape[1],
                                        Z=1,
                                        C=1,
                                        T=1,
                                        dtype=pr.dtype) as bw_pr:
                bw_pr[:] = pr
                
            with BioWriter(gt_filename, Y=gt.shape[0],
                                        X=gt.shape[1],
                                        Z=1,
                                        C=1,
                                        T=1,
                                        dtype=gt.dtype) as bw_gt:
                bw_gt[:] = gt
                
            img_count = img_count + 1
        
        queue.put(cuda_num)
        python_command = "python /home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/polus-plugins/features/polus-pixelwise-evaluation-plugin/src/main.py" + \
                        f" --GTDir {gt_collection}" + \
                        f" --PredDir {pr_collection}" + \
                        f" --inputClasses 1" + \
                        f" --individualStats False" + \
                        f" --totalStats True" + \
                        f" --outDir {output_path}"
                            
        print(python_command)
            
        logfile = open(os.path.join(output_path, "logs.log"), 'a')
        subprocess.call(python_command, shell=True, stdout=logfile, stderr=logfile)
                                
            
NUM_GPUS = len(available_gpus)
NUM_PROCESSES = len(smp_inputs_list)
PROC_PER_GPU = int(np.ceil(NUM_PROCESSES/NUM_GPUS))


for gpu_ids in (range(8)):
    queue.put(str(gpu_ids))
            
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            # i = 0
    for smp_input in smp_inputs_list:
        executor.submit(evaluation, smp_input, queue.get())
            
            
            
            