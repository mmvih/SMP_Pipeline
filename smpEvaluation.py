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

from src.training import initialize_dataloader
from src.training import MultiEpochsDataLoader
from src.utils import Dataset
from src.utils import get_labels_mapping
from src.utils import METRICS
from src.utils import LOSSES
from src.utils import MODELS

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("smpEvaluation")
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

def evaluation(tor_device,
               input_model_dirpath : str,
               output_metric_dirpath : str,
               test_loader : Dataset,
               names : list,
               metrics : list,
               models_metrics : dict):
    
    try:

        checkpointpth_path = os.path.join(input_model_dirpath, "checkpoint.pth")
        checkpoint = torch.load(checkpointpth_path)
        
        # using checkpoint.pth is more reliable than model.pth
        model = MODELS[checkpoint["model_name"]](encoder_name=checkpoint["encoder_variant"],
                                                 encoder_weights=checkpoint["encoder_weights"],
                                                 in_channels=1,
                                                 activation="sigmoid")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(tor_device)
        logger.debug(f"Done Loading the Model")

        test_loader_metrics = {metrics[metric].__name__: {} for metric in metrics}
        
        num_testing = torch.tensor(len(test_loader)).to(tor_device).float()
        
        start_time = time.time()
        for (img_arr, gt_arr), name in zip(test_loader, names):
            
            xtensor = torch.from_numpy(img_arr).to(tor_device).unsqueeze(0)
            ytensor = torch.from_numpy(gt_arr).to(tor_device).unsqueeze(0)

            pr_mask = model.predict(xtensor)
            pr_mask[pr_mask >= .50] = 1
            pr_mask[pr_mask < .50] = 0 

            for metric in metrics:
                metric_name = metrics[metric].__name__
                if metric_name in METRICS:
                    metric_value = (METRICS[metric_name].forward(self=metrics[metric], y_pr=pr_mask, y_gt=ytensor))
                else:
                    metric_value = (LOSSES[metric_name].forward(self=metrics[metric], y_pred=pr_mask, y_true=ytensor))
                
                test_loader_metrics[metric_name][name] = metric_value.item()
                
        total_time = time.time() - start_time
        logger.debug(f"Done Iterating throgh the testing images - time took {total_time} seconds")
        
        model_name = os.path.basename(input_model_dirpath)
        models_metrics["time"][model_name] = total_time
        
        for metric in metrics:
            
            metric_name = metrics[metric].__name__
            
            test_loader_metric = test_loader_metrics[metric_name]
            test_loader_metric = {k:v for k,v in sorted(test_loader_metric.items(), key=lambda item:item[1])}
            
            test_loader_metric_values = list(test_loader_metric.values())
            
            metric_summary = {"model" : model_name,
                              "metric" : metric_name,
                              "average" : np.average(test_loader_metric_values),
                              "standard_deviation" : np.std(test_loader_metric_values),
                              "maximum" : np.max(test_loader_metric_values),
                              "minimum" : np.min(test_loader_metric_values),
                              "time" : total_time}
            logger.debug(f"\n{metric_summary}")
            
            models_metrics[metric][model_name] = f"{metric_summary['average']} - {metric_summary['standard_deviation']}"
            
            metric_summary["testing_images"] = test_loader_metric
            
            metrics_json_path = os.path.join(output_metric_dirpath, f"{metric_name}.json")
            with open(metrics_json_path, 'w') as metric_json_file:
                json.dump(metric_summary, metric_json_file, indent=4)
            logger.debug(f"Saved to {metrics_json_path}")

        return 0

    except Exception as e:
        logger.info(f"ERROR: {e}")
        
    finally:
        QUEUE.put(tor_device)
        
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
    
    if not os.path.exists(output_metrics_dirpath):
        raise ValueError(f"Output Directory ({output_metrics_dirpath}) does not exist")
    
    images_testing_dirpath = args.imagesTestDir
    labels_testing_dirpath = args.labelsTestDir
    logger.info(f"Testing Images Directory : {images_testing_dirpath}")
    logger.info(f"Testing Labels Directory : {labels_testing_dirpath}")
    
    file_pattern = args.filePattern
    logger.info(f"File Pattern : {file_pattern}")
    
    logger.info(f"\nQueuing up {NUM_GPUS} GPUs ...")
    for gpu_ids in (range(NUM_GPUS)):
        logger.debug(f"queuing device {gpu_ids} - {torch.cuda.get_device_name(gpu_ids)}")
        QUEUE.put(torch.device(f"cuda:{gpu_ids}"))
    
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
    
    metrics = {metric.__name__ : metric() for metric in METRICS.values()}
    
    models_metrics = {metric.__name__ : {} for metric in METRICS.values()}
    models_metrics["time"] = {}
    
    with ThreadPoolExecutor(max_workers=NUM_GPUS+(NUM_GPUS/2)) as executor:
        
        for curr_smp_model in input_models_list[0:5]:
            
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
            
            json_file = open(configjson_path, 'r')
            config_dict = json.load(json_file)
            
            loss = config_dict['lossName']
            metric_loss = LOSSES[loss]()
            metric_loss.__name__ = loss
            if metric_loss.__name__ not in metrics:
                metrics[metric_loss.__name__] = metric_loss
                models_metrics[metric_loss.__name__] = {}
        
            sleeping_in = 0
            while QUEUE.empty():
                sleeping_in += 1
                time.sleep(30)
                logger.debug(f"There are currently no available GPUS to use - sleeping in x{sleeping_in}")
            
            if not os.path.exists(output_metrics_dirpath):
                os.mkdir(output_metrics_dirpath)
            
            if not os.path.exists(output_metric_dirpath):
                os.mkdir(output_metric_dirpath)
            
            if not QUEUE.empty():
                executor.submit(evaluation, QUEUE.get(), input_model_dirpath, output_metric_dirpath, test_loader, names, metrics, models_metrics)

    logger.info(f"Summarizing Metrics across all the Models")
    for models_metric in models_metrics:
        
        models_metric_output_path = os.path.join(output_metrics_dirpath, f"{models_metric}.json")
        models_metric_dict = models_metrics[models_metric]
        
        with open(models_metric_output_path, 'w') as models_metric_output_json_file:
            json.dump(models_metric_dict, models_metric_output_json_file, indent=4)
            
    logger.info(f"Done Summarizing!")
    

main()