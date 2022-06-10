import os,sys,json,copy
import logging,argparse

import numpy as np

from filepattern import FilePattern 

from matplotlib import pyplot as plt

from itertools import cycle, islice
from itertools import repeat

from multiprocessing import Queue 
import subprocess
from concurrent.futures import ThreadPoolExecutor

import torch
import torchvision

import time

polus_smp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polus-plugins/segmentation/polus-smp-training-plugin/")
sys.path.append(polus_smp_dir)

from src.utils import Dataset
from src.utils import get_labels_mapping
from src.training import initialize_dataloader
from src.training import MultiEpochsDataLoader
from src.utils import METRICS
from src.utils import LOSSES

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("savepredictions")
logger.setLevel("DEBUG")

NUM_GPUS = torch.cuda.device_count()
QUEUE = Queue()

def getLoader(images_Dir, 
              labels_Dir,
              file_pattern):
    
    images_fp = FilePattern(images_Dir, file_pattern)
    labels_fp = FilePattern(labels_Dir, file_pattern)

    image_array, label_array, names = get_labels_mapping(images_fp(), labels_fp(), provide_names=True)

    testing_dataset = Dataset(images=image_array,
                              labels=label_array)
    testing_loader = MultiEpochsDataLoader(testing_dataset, num_workers=4, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)
    
    return testing_dataset, images_fp, labels_fp, names

def evaluation(tor_device : torch.cuda.device,
               input_model_dirpath : str,
               output_metric_dirpath : str,
               test_loader : Dataset,
               names : list,
               metrics : list):
    
    try:
        print(smp_input_file, flush=True)
        
        smp_input_path = os.path.join(smp_inputs_path, smp_input_file)
        plot_path = os.path.join(smp_input_path, "Logs.jpg")
        model_path = os.path.join(smp_input_path, "model.pth")
        config_path = os.path.join(smp_input_path, "config.json")
        ERROR_path = os.path.join(smp_input_path, "ERROR")

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
        QUEUE.put(device)
        
def main():
    
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

    parser.add_argument('--inputModels', dest='inputModels', type=str, required=True,
                        help='Path to Input Predictions')
    parser.add_argument('--outputMetrics', dest='outputMetrics', type=str, required=True,
                        help='Path to where the Output Metrics will be saved')
    parser.add_argument('--imagesTestDir', dest='imagesTestDir', type=str, required=True, \
                        help='Path to Images that are for testing')
    parser.add_argument('--labelsTestDir', dest='labelsTestDir', type=str, required=True, \
                        help='Path to Labels that are for testing')
    parser.add_argument('--filePattern', dest='filePattern', type=str, required=False,
                        default=".*", help="Pattern of Images for creating predictions")
    
    args = parser.parse_args()
    input_models_dirpath = args.inputModels
    output_metrics_dirpath = args.outputMetrics
    logger.info(f"Input Models Directory : {input_models_dirpath}")
    logger.info(f"Output Predictions Directory : {output_metrics_dirpath}")
    
    images_testing_dirpath = args.imagesTestDir
    labels_testing_dirpath = args.labelsTestDir
    logger.info(f"Testing Images Directory : {images_testing_dirpath}")
    logger.info(f"Testing Labels Directory : {labels_testing_dirpath}")
    
    file_pattern = args.filePattern
    logger.info(f"File Pattern : {file_pattern}")
    
    logger.info(f"\nQueuing up {NUM_GPUS} GPUs ...")
    for gpu_ids in (range(NUM_GPUS)):
        logger.debug(f"queuing device {gpu_ids} - {torch.cuda.get_device_name(gpu_ids)}")
        QUEUE.put(torch.cuda.device(gpu_ids))
    
    logger.info("\nGetting Loaders ...")
    test_loader, _, _, names = getLoader(images_Dir=images_testing_dirpath,
                                         labels_Dir=labels_testing_dirpath,
                                         file_pattern=file_pattern)
    num_examples = len(test_loader)
    
    input_models_list = os.listdir(input_models_dirpath)
    num_models = len(input_models_list)

    counter = 0
    logger.info(f"\nIterating through {num_models} models ...")
    logger.info(f"Each model will be generating {num_examples} predictions")
    
    metrics = list(metric() for metric in METRICS.values())
    avg_model_metric_comparison = {metric.__name__ : {} for metric in metrics}
    std_model_metric_comparison = {metric.__name__ : {} for metric in metrics}
    model_times                 = {}
    
    with ThreadPoolExecutor(max_workers=NUM_GPUS+(NUM_GPUS/2)) as executor:
        
        for curr_smp_model in input_models_list:
            
            counter += 1
            logger.info(f"\n{counter}. {curr_smp_model}")
            
            input_model_dirpath = os.path.join(input_models_dirpath, curr_smp_model)
            output_metric_dirpath = os.path.join(output_metrics_dirpath, curr_smp_model)
            logger.debug(f"Input Predictions Path : {input_model_dirpath}")
            logger.debug(f"Output Labels Path : {output_metric_dirpath}")
            
            ERROR_path = os.path.join(input_model_dirpath, "ERROR")
            if os.path.exists(ERROR_path):
                logger.debug(f"Not Running ({counter}/{num_models}) - ERROR exists {ERROR_path}")
                continue
            
            modelpth_path = os.path.join(input_model_dirpath, "model.pth")
            if not os.path.exists(modelpth_path):
                logger.debug(f"Not Running ({counter}/{num_models}) - model.pth does not exist {input_model_dirpath}")
                continue
        
            configjson_path = os.path.join(input_model_dirpath, "config.json")
            if not os.path.exists(configjson_path):
                logger.debug(f"Not Running ({counter}/{num_models}) - config.json does not exist {input_model_dirpath}")
                continue
            with open(configjson_path) as json_file:
                config_dict = json.load(json_file)
                loss = config_dict['lossName']
                metric_loss = LOSSES[loss]()
                metric_loss.__name__ = loss
                metrics.append(metric_loss)
        
            sleeping_in = 0
            while QUEUE.empty():
                sleeping_in += 1
                time.sleep(30)
                logger.debug(f"There are currently no available GPUS to use - sleeping in x{sleeping_in}")
            
            if not os.path.exists(output_metric_dirpath):
                os.mkdir(output_metric_dirpath)
            
            if not QUEUE.empty():
                executor.submit(evaluation, QUEUE.get(), input_model_dirpath, output_metric_dirpath, test_loader, names, metrics)

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

main()