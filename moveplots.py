import os,sys,json
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import shutil

import numpy as np

from filepattern import FilePattern 

from matplotlib import pyplot as plt

import torch
import torchvision

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
smp_outputs_path = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_testing123"

testing_images = "/home/vihanimm/SegmentationModelToolkit/Data/tif_data/nuclear/test/image/"
testing_labels = "/home/vihanimm/SegmentationModelToolkit/Data/tif_data/nuclear/test/groundtruth_centerbinary_2pixelsmaller/"

smp_inputs_list = os.listdir(smp_inputs_path)

def getLoader(images_Dir, 
              labels_Dir):
        
    images_fp = FilePattern(images_Dir, "nuclear_test_61{x}.tif")
    labels_fp = FilePattern(labels_Dir, "nuclear_test_61{x}.tif")
    
    image_array, label_array = get_labels_mapping(images_fp(), labels_fp())
    
    testing_dataset = Dataset(images=image_array,
                              labels=label_array)
    testing_loader = MultiEpochsDataLoader(testing_dataset, num_workers=4, batch_size=8, shuffle=True, pin_memory=True, drop_last=True)
    
    testing_dataset_vis = Dataset(images=image_array,
                                    labels=label_array,
                                    preprocessing=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor()]))
    testing_loader_vis = MultiEpochsDataLoader(testing_dataset_vis, num_workers=4, batch_size=8, shuffle=True, pin_memory=True, drop_last=True)
        
    return testing_dataset, testing_dataset_vis

def add_to_axis(prediction, threshold, axis=None):
   
    new_img = copy.deepcopy(prediction)
    new_img[new_img < threshold]  = 0
    new_img[new_img >= threshold] = 1

    if axis != None:
        axis.imshow(new_img.cpu())
        axis.set_title(f"Threshold: {threshold}")

def getPlots(dictionary, metric_name, type, fig=None, ax=None):

    counter = 0
    for item in dictionary.keys():
        image       = dictionary[item]["image"]
        groundtruth = dictionary[item]["groundtruth"]
        prediction  = dictionary[item]["prediction"]
        
        ax[0,0].imshow(image.cpu())
        ax[0,0].set_title("Image")
        ax[0,1].imshow(groundtruth.cpu())
        ax[0,1].set_title("GroundTruth")
        ax[0,2].imshow(prediction.cpu())
        ax[0,2].set_title("Prediction")
        
        threshold = 0.1
        for i in range(1, 4):
            for j in range(3):
                add_to_axis(prediction=prediction, threshold=threshold, axis=ax[i,j])
                threshold += 0.1
                threshold = round(threshold,1)
        
        saveto = os.path.join(smp_output_path, type)
        if not os.path.exists(saveto):
            os.mkdir(saveto)
        
        fig.suptitle(f"{os.path.basename(smp_output_path)}: {type}{counter} {metric_name}", fontsize="14")
        plot_name = os.path.join(saveto, f"{type}_{counter}_{metric_name}.jpg")
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.savefig(plot_name, bbox_inches="tight")
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

fig, ax = plt.subplots(4, 3, figsize = (12, 16))

test_loader, test_loader_vis = getLoader(images_Dir=testing_images,
                                            labels_Dir=testing_labels)
test_loader_len = torch.tensor(len(test_loader))

for smp_input_file in smp_inputs_list:
    
    smp_input_path = os.path.join(smp_inputs_path, smp_input_file)
    plot_path = os.path.join(smp_input_path, "Logs.jpg")
    model_path = os.path.join(smp_input_path, "model.pth")
    config_path = os.path.join(smp_input_path, "config.json")
    ERROR_path = os.path.join(smp_input_path, "ERROR")
    
    if os.path.exists(ERROR_path):
        print("ERROR")
        continue

    smp_output_path = os.path.join(smp_outputs_path, smp_input_file)
    if not os.path.exists(smp_output_path):
        os.mkdir(smp_output_path)
        
    if os.path.exists(plot_path):
        shutil.copy(plot_path, os.path.join(smp_output_path, "scores.jpg"))
        
    if os.path.exists(model_path) and os.path.exists(config_path):
        
        model = torch.load(model_path,map_location=torch.device('cuda:0'))
        
        configObj     = open(config_path, 'r')
        configDict    = json.load(configObj)


        metric_outputs = {}
        best_metric_outputs  = {}
        worst_metric_outputs = {}      
        for metric in metrics:
            metric_outputs[metric.__name__] = {'avg': torch.tensor(0).to(torch.device("cuda:0")).float(), 'maxi': torch.tensor(0), 'mini': torch.tensor(1), 
                                                'all': [], 'stddev' : torch.tensor(0)}
            best_metric_outputs[metric.__name__] = {}
            worst_metric_outputs[metric.__name__] = {}
            
        i = 0
        for test in test_loader:
            test0 = test[0]
            test1 = test[1]
            xtensor = torch.from_numpy(test0).to(torch.device('cuda:0')).unsqueeze(0)
            ytensor = torch.from_numpy(test1).to(torch.device('cuda:0')).unsqueeze(0)
            ztensor = torch.from_numpy(test_loader_vis[i][0]).to(torch.device('cuda:0')).unsqueeze(0)
            pr_mask = model.predict(xtensor)

            for metric in metrics:
                try:
                    metric_value = (METRICS[metric.__name__].forward(self=metric, y_pr=pr_mask, y_gt=ytensor))
                except:
                    metric_value = (LOSSES[metric.__name__].forward(self=metric, y_pred=pr_mask, y_true=ytensor))
                
                metric_outputs[metric.__name__]['avg'] += torch.divide(metric_value, test_loader_len)
                metric_outputs[metric.__name__]['all'].append(metric_value)
                metric_outputs[metric.__name__]['mini'] = torch.minimum(metric_value, metric_outputs[metric.__name__]['mini'])
                metric_outputs[metric.__name__]['maxi'] = torch.maximum(metric_value, metric_outputs[metric.__name__]['maxi'])
            
                if i < 2:
                    best_metric_outputs[metric.__name__][float(metric_value)] = {"image"       : torch.squeeze(ztensor),
                                                                                 "groundtruth" : torch.squeeze(ytensor),
                                                                                 "prediction"  : torch.squeeze(pr_mask)}
                
                    worst_metric_outputs[metric.__name__][float(metric_value)] = {"image"      : torch.squeeze(ztensor),
                                                                                  "groundtruth" : torch.squeeze(ytensor),
                                                                                  "prediction"  : torch.squeeze(pr_mask)}
                else:
                    worstof_bestvalues = min(best_metric_outputs[metric.__name__].keys())
                    bestof_worstvalues = max(worst_metric_outputs[metric.__name__].keys())

                    if metric_value > worstof_bestvalues:
                        del best_metric_outputs[metric.__name__][worstof_bestvalues]
                        best_metric_outputs[metric.__name__][float(metric_value)] = {"image"       : torch.squeeze(ztensor),
                                                                                     "groundtruth" : torch.squeeze(ytensor),
                                                                                     "prediction"  : torch.squeeze(pr_mask)}
                    
                    if metric_value < bestof_worstvalues:
                        del worst_metric_outputs[metric.__name__][bestof_worstvalues]
                        worst_metric_outputs[metric.__name__][float(metric_value)] = {"image"       : torch.squeeze(ztensor),
                                                                                     "groundtruth" : torch.squeeze(ytensor),
                                                                                     "prediction"  : torch.squeeze(pr_mask)}
            i += 1  
        
        
        
        metric_names = [metric.__name__ for metric in metrics]
        metric_values = [metric_outputs[metric]['avg'] for metric in metric_names]
        print("STARTING PLOTS")
        for metric in metrics:
            getPlots(best_metric_outputs[metric.__name__], metric_name=metric.__name__, type="best", fig=fig, ax=ax)
            getPlots(worst_metric_outputs[metric.__name__], metric_name=metric.__name__, type="worst", fig=fig, ax=ax)
            metric_outputs[metric.__name__]['stddev'] = torch.std(torch.stack(metric_outputs[metric.__name__]['all']))
            avg_model_metric_comparison[metric.__name__][smp_input_file] = metric_outputs[metric.__name__]['avg'].item()
            std_model_metric_comparison[metric.__name__][smp_input_file] = metric_outputs[metric.__name__]['stddev'].item()
        print("DONE GENERATING PLOTS")

        # plt.clf()
        # plt.cla()
        # plt.rcParams["figure.figsize"] = (8,8)
        # bars = plt.bar(metric_names, metric_values, color=['green','blue','purple','brown','teal','pink'])
        # for bar in bars:
        #     yval = bar.get_height()
        #     plt.text(bar.get_x(), yval + .005, "{0:0.5f}".format(yval))
        # plt.title(f"Average Scores of 1249 Tissuenet Testing Images for {smp_input_file}")
        # plt.savefig(os.path.join(smp_output_path, "avg_scores.jpg"))
        # plt.clf()
        
        with open(os.path.join(smp_output_path, "avg_scores.txt"), "w") as textfile:
            textfile.write(str(metric_names))
            textfile.write("\n")
            textfile.write(str(metric_values))
            
        print(metric_outputs)
        print(" ")

for metric in metrics:    
    model_json_metric_path = os.path.join(smp_outputs_path, metric.__name__ + "_models.json")
    avg_model_metric_comparison[metric.__name__] = {k: str(v) + "-" + str(std_model_metric_comparison[metric.__name__][k]) for k, v in sorted(avg_model_metric_comparison[metric.__name__].items(), 
                                                                    key=lambda item: item[1])}
    print(avg_model_metric_comparison[metric.__name__])
    with open(model_json_metric_path, 'w') as config_file:
        json.dump(avg_model_metric_comparison[metric.__name__], config_file, indent=4)
        
        
        