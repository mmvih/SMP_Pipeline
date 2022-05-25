import os,sys,json
import copy
# os.environ["CUDA_VISIBLE_DEVICES"] = "0", "1", "2", "3", "4", "6", "7"
import shutil

import numpy as np

from filepattern import FilePattern 

from matplotlib import pyplot as plt

from itertools import cycle, islice
from itertools import repeat

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue

import torch
import torchvision

import bfio 
from bfio import BioWriter

import time


polus_dir = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/polus-plugins/segmentation/polus-smp-training-plugin/"
sys.path.append(polus_dir)

from src.utils import Dataset
from src.utils import get_labels_mapping
from src.training import initialize_dataloader
from src.training import MultiEpochsDataLoader
from src.utils import METRICS
from src.utils import LOSSES

import math

smp_inputs_path =  "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP"
smp_outputs_path = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_thresanalysis"
if not os.path.exists(smp_outputs_path): os.mkdir(smp_outputs_path)

testing_images = "/home/vihanimm/SegmentationModelToolkit/Data/tif_data/nuclear/test/image/"
testing_labels = "/home/vihanimm/SegmentationModelToolkit/Data/tif_data/nuclear/test/groundtruth_centerbinary_2pixelsmaller/"

smp_inputs_list = os.listdir(smp_inputs_path)
print(len(smp_inputs_list))
for smp_input in smp_inputs_list:
    smp_input_model_dirpath = os.path.join(smp_inputs_path, smp_input)
    smp_input_model_path = os.path.join(os.path.join(smp_input_model_dirpath, "model.pth"))
    smp_input_ERROR_path = os.path.join(os.path.join(smp_input_model_dirpath, "ERROR"))
    if not os.path.exists(smp_input_model_path):
        smp_inputs_list.remove(smp_input)

def getLoader(images_Dir, 
              labels_Dir):
        
    # nuclear_test_61{x}.tif for only ten images
    my_fp = ".*"
    images_fp = FilePattern(images_Dir, my_fp)
    labels_fp = FilePattern(labels_Dir, my_fp)
    
    image_array, label_array = get_labels_mapping(images_fp(), labels_fp())

    testing_dataset = Dataset(images=image_array,
                              labels=label_array)
    testing_loader = MultiEpochsDataLoader(testing_dataset, num_workers=4, batch_size=8, shuffle=True, pin_memory=True, drop_last=True)
    
    testing_dataset_vis = Dataset(images=image_array,
                                    labels=label_array,
                                    preprocessing=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor()]))
    testing_loader_vis = MultiEpochsDataLoader(testing_dataset_vis, num_workers=4, batch_size=8, shuffle=True, pin_memory=True, drop_last=True)
        
    return testing_dataset, testing_dataset_vis, images_fp, labels_fp

def add_to_axis(prediction, threshold, axis=None):
   
    new_img = copy.deepcopy(prediction)
    new_img[new_img < threshold]  = 0
    new_img[new_img >= threshold] = 1

    if axis != None:
        axis.imshow(new_img.cpu())
        axis.set_title(f"Threshold: {threshold}")

def getPlots(dictionary, metric_name, type, smp_output_path, fig=None, ax=None):

    counter = 0
    for item in dictionary.keys():
        image       = dictionary[item]["image"]
        groundtruth = dictionary[item]["groundtruth"]
        prediction  = dictionary[item]["prediction"]
        
        ax[0].imshow(image.cpu())
        ax[0].set_title("Image")
        ax[1].imshow(groundtruth.cpu())
        ax[1].set_title("GroundTruth")
        ax[2].imshow(prediction.cpu())
        ax[2].set_title("Prediction")
        
        # threshold = 0.1
        # for i in range(1, 4):
        #     for j in range(3):
        #         add_to_axis(prediction=prediction, threshold=threshold, axis=ax[i,j])
        #         threshold += 0.1
        #         threshold = round(threshold,1)
        

        
        if metric_name == 'MCCLoss' and type == 'worst':
            type = "best"
        if metric_name == 'MCCLoss' and type == 'best':
            type = "worst"

        saveto = os.path.join(smp_output_path, type)
        if not os.path.exists(saveto):
            os.mkdir(saveto)
        fig.suptitle(f"{os.path.basename(smp_output_path)}: {type}{counter} {metric_name}", fontsize="14")
        plot_name = os.path.join(saveto, f"{type}_{counter}_{metric_name}.jpg")
        fig.subplots_adjust(wspace=None, hspace=None)
        fig.savefig(plot_name, bbox_inches="tight")
        assert os.path.exists(plot_name)
        plt.cla()
        counter += 1


loss = 'MCCLoss'
metrics = list(metric() for metric in METRICS.values())
metric_loss = LOSSES[loss]()
metric_loss.__name__ = loss
metrics.append(metric_loss)

avg_model_metric_comparison = {metric.__name__ : {} for metric in metrics}
std_model_metric_comparison = {metric.__name__ : {} for metric in metrics}
model_times                 = {}

fig, ax = plt.subplots(1, 3, figsize = (12, 4))

test_loader, test_loader_vis, images_fp, labels_fp = getLoader(images_Dir=testing_images,
                                            labels_Dir=testing_labels)
test_loader_len = torch.tensor(len(test_loader))

example_names = {}
example_names_list = []
for image,label in zip(images_fp(), labels_fp()):
    basename = os.path.basename(str(image[0]['file']))
    assert basename == os.path.basename(str(label[0]['file']))
    example_names[basename] = None
    example_names_list.append(basename)

def forloop(smp_input_file, 
            smp_inputs_path,
            smp_outputs_path,
            device):
    
    try:
        print(smp_input_file, flush=True)
        
        smp_input_path = os.path.join(smp_inputs_path, smp_input_file)
        plot_path = os.path.join(smp_input_path, "Logs.jpg")
        model_path = os.path.join(smp_input_path, "model.pth")
        config_path = os.path.join(smp_input_path, "config.json")
        ERROR_path = os.path.join(smp_input_path, "ERROR")
        
        # if os.path.exists(ERROR_path):
        #     print("ERROR")
        #     queue.put(device)
        #     return

        smp_output_path = os.path.join(smp_outputs_path, smp_input_file)
        if not os.path.exists(smp_output_path):
            os.mkdir(smp_output_path)
        
        starttime = time.time()  
        if os.path.exists(model_path) and os.path.exists(config_path):
            
            model = torch.load(model_path,map_location=device)
            
            configObj     = open(config_path, 'r')
            configDict    = json.load(configObj)

            metric_outputs = {}
            best_metric_outputs  = {}
            worst_metric_outputs = {}      
            for metric in metrics:
                metric_outputs[metric.__name__] = {'model': smp_input_file, 'metric': metric.__name__, 'avg': torch.tensor(0).to(device).float(), 'maxi': torch.tensor(0), 'mini': torch.tensor(1), 
                                                    'stddev' : torch.tensor(0), 'all': example_names.copy(), 'time_allmetrics': 0}
                best_metric_outputs[metric.__name__] = {}
                worst_metric_outputs[metric.__name__] = {}
            
            i = 0
            for test in test_loader:
                test0 = test[0]
                test1 = test[1]
                xtensor = torch.from_numpy(test0).to(device).unsqueeze(0)
                ytensor = torch.from_numpy(test1).to(device).unsqueeze(0)
                ztensor = torch.from_numpy(test_loader_vis[i][0]).to(device).unsqueeze(0)
                pr_mask = model.predict(xtensor)
                pr_mask[pr_mask >= .50] = 1
                pr_mask[pr_mask < .50] = 0 

                for metric in metrics:
                    try:
                        metric_value = (METRICS[metric.__name__].forward(self=metric, y_pr=pr_mask, y_gt=ytensor))
                    except:
                        metric_value = (LOSSES[metric.__name__].forward(self=metric, y_pred=pr_mask, y_true=ytensor))
                    
                    metric_outputs[metric.__name__]['avg'] += torch.divide(metric_value, test_loader_len)
                    metric_outputs[metric.__name__]['all'][example_names_list[i]] = metric_value.item()
                    metric_outputs[metric.__name__]['mini'] = torch.minimum(metric_value, metric_outputs[metric.__name__]['mini'])
                    metric_outputs[metric.__name__]['maxi'] = torch.maximum(metric_value, metric_outputs[metric.__name__]['maxi'])
                
                i += 1 
            
            metric_names = [metric.__name__ for metric in metrics]
            metric_values = [metric_outputs[metric]['avg'].item() for metric in metric_names]

            totaltime = time.time()-starttime
            model_times[smp_input_file] = totaltime
            
            for metric in metrics:
                # getPlots(best_metric_outputs[metric.__name__], metric_name=metric.__name__, smp_output_path, type="best", fig=fig, ax=ax)
                # getPlots(worst_metric_outputs[metric.__name__], metric_name=metric.__name__, smp_output_path, type="worst", fig=fig, ax=ax)
                metric_outputs[metric.__name__]['stddev'] = np.std(list(metric_outputs[metric.__name__]['all'].values()))
                metric_outputs[metric.__name__]['avg'] = metric_outputs[metric.__name__]['avg'].item()
                metric_outputs[metric.__name__]['mini'] = metric_outputs[metric.__name__]['mini'].item()
                metric_outputs[metric.__name__]['maxi'] = metric_outputs[metric.__name__]['maxi'].item()
                metric_outputs[metric.__name__]['all']  = {k: v for k, v in sorted(metric_outputs[metric.__name__]['all'].items(), 
                                                                        key=lambda item: item[1])}
                metric_outputs[metric.__name__]['time_allmetrics'] = totaltime
                
                avg_model_metric_comparison[metric.__name__][smp_input_file] = metric_outputs[metric.__name__]['avg']
                std_model_metric_comparison[metric.__name__][smp_input_file] = metric_outputs[metric.__name__]['stddev']
                metrics_json = os.path.join(smp_output_path, f"metrics_{metric.__name__}.json")
                with open(metrics_json, 'w') as metric_json:
                    json.dump(metric_outputs[metric.__name__], metric_json, indent=4)

            print("TOTAL TIME: ", totaltime, flush=True)
            print(" ")
    except:
        print(f"{smp_input_file} threw error")
    finally:
        queue.put(device)

gpu_list = [0, 1, 2, 3, 5, 6, 7]
gpu_list = list(islice(cycle(gpu_list), len(smp_inputs_list)))
gpu_list = [torch.device(f"cuda:{gpu}") for gpu in gpu_list]

queue = Queue()
for i in range(8):
    queue.put(torch.device(f"cuda:{i}"))

counter = 0
sleeping_in = 0
with ThreadPoolExecutor(max_workers=20) as executor:
    for smp_input in smp_inputs_list:
        
        smp_input_path = os.path.join(smp_inputs_path, smp_input)
        plot_path = os.path.join(smp_input_path, "Logs.jpg")
        model_path = os.path.join(smp_input_path, "model.pth")
        config_path = os.path.join(smp_input_path, "config.json")
        ERROR_path = os.path.join(smp_input_path, "ERROR")
        
        if os.path.exists(ERROR_path):
            continue
        
        if not os.path.exists(model_path):
            continue
    
        while queue.empty():
            sleeping_in += 1
            time.sleep(30)
        
        if not queue.empty():
            executor.submit(forloop, smp_input, smp_inputs_path, smp_outputs_path, queue.get())

    # executor.map(forloop, smp_inputs_list, repeat(smp_inputs_path), repeat(smp_outputs_path), gpu_list)

    
for metric in metrics:    
    model_json_metric_path = os.path.join(smp_outputs_path, metric.__name__ + "_models.json")
    avg_model_metric_comparison[metric.__name__] = {k: str(v) + "-" + str(std_model_metric_comparison[metric.__name__][k]) for k, v in sorted(avg_model_metric_comparison[metric.__name__].items(), 
                                                                    key=lambda item: item[1])}
    print(avg_model_metric_comparison[metric.__name__])
    with open(model_json_metric_path, 'w') as config_file:
        json.dump(avg_model_metric_comparison[metric.__name__], config_file, indent=4)
model_json_time_path = os.path.join(smp_outputs_path, "time_models.json")
model_times = {k: v for k, v in sorted(model_times.items(), 
                                        key=lambda item: item[1])}
with open(model_json_time_path, 'w') as time_file:
    json.dump(model_times, time_file, indent=4)
        
        