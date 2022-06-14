import os, sys
import csv
import logging, argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("lossPlots")
logger.setLevel("DEBUG")

def main():
    
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

    parser.add_argument('--inputModels', dest='inputModels', type=str, required=True,
                        help='Path to Input Models Directory')
    parser.add_argument('--outputLosses', dest='outputLosses', type=str, required=True, \
                        help='Path to Output Directory')
    
    args = parser.parse_args()
    input_models_dirpath  = args.inputModels
    output_losses_dirpath = args.outputLosses

    if not os.path.exists(output_losses_dirpath):
        raise ValueError(f"Output Directory ({output_losses_dirpath}) does not exist")
    
    logger.info(f"Input Models Directory : {input_models_dirpath}")
    logger.info(f"Output Models Directory : {output_losses_dirpath}")
    
    input_models_list = os.listdir(input_models_dirpath)
    num_models = len(input_models_list)
    
    
    counter = 0
    fig, axes = plt.subplots(2, 3, figsize=(15,12))
    fig.tight_layout(pad=3)
    logger.info(f"Iterating through {num_models} models ...")
    for curr_smp_model in input_models_list:
        
        counter += 1
        logger.info(f"\n{counter}. {curr_smp_model}")
        
        input_model_dirpath = os.path.join(input_models_dirpath, curr_smp_model)
        if not os.path.exists(input_model_dirpath):
            logger.debug(f"Not Running ({counter}/{num_models}) - input Model directory does not exist ({output_model_dirpath})")
        
        output_loss_dirpath = os.path.join(output_losses_dirpath, curr_smp_model)
        if not os.path.exists(output_loss_dirpath):
            os.mkdir(output_loss_dirpath)
        
        trainlogscsv_path = os.path.join(input_model_dirpath, "trainlogs.csv")
        validlogscsv_path = os.path.join(input_model_dirpath, "validlogs.csv")
        
        if not os.path.exists(trainlogscsv_path):
            logger.debug(f"Not Running ({counter}/{num_models}) - {trainlogscsv_path} does not exist")
            continue
    
        if not os.path.exists(validlogscsv_path):
            logger.debug(f"Not Running ({counter}/{num_models}) - {validlogscsv_path} does not exist")
        
        logger.debug(f"Path for Training Logs   : {trainlogscsv_path}")
        logger.debug(f"Path for Validation Logs : {validlogscsv_path}")
        
        trainlogs_df = pd.read_csv(trainlogscsv_path, header=None)
        validlogs_df = pd.read_csv(validlogscsv_path, header=None)
        
        fig.suptitle(curr_smp_model)
        for (train_name, train_data), (valid_name, valid_data), ax in \
            zip(trainlogs_df.iteritems(), validlogs_df.iteritems(), axes.flat):
            
            ax.set_ylim([0,1])
            ax.set_xlabel("EPOCHS")
            
            train_data_split = train_data.str.split(":")
            valid_data_split = valid_data.str.split(":")
            
            new_train_name = train_data_split[0][0]
            new_valid_name = valid_data_split[0][0]
            
            assert new_train_name == new_valid_name, \
                f"{new_train_name} and {new_valid_name} are not the same; " + \
                f"Order of trainlogs.csv and validlogs.csv is not the same"
            
            ax.set_ylabel(new_train_name)
            
            trainlogs_df[train_name] = train_data_split.str[1].astype(float)
            validlogs_df[valid_name] = valid_data_split.str[1].astype(float)
            
            ax.plot(train_data_split.str[1].astype(float), label="Train")
            ax.plot(valid_data_split.str[1].astype(float), label="Validation")
            
            trainlogs_df.rename(columns = {train_name : train_data_split[0][0]}, inplace=True)
            validlogs_df.rename(columns = {valid_name : valid_data_split[0][0]})
    
            ax.legend(["Training", "Validation"])
        
        output_logpng_path = os.path.join(output_loss_dirpath, "epochLogs.png")
        plt.savefig(output_logpng_path)
        logger.debug(f"Saved Output Plots to {output_logpng_path}") #override any existing plots
        logger.debug(f"analyzed {counter}/{num_models} models")
        
        for ax in axes.flat:
            ax.clear()
            
    logger.info("Done Iterating through all {num_models} models!")
                
            
            
main()
            
    