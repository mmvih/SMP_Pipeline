from ast import Continue
import os, sys, json
import logging, argparse

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams['figure.dpi'] = 300

import numpy as np
import pandas as pd

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("savepredictions")
logger.setLevel("DEBUG")


def create_boxplots(dataframe, output_directory, show_howmany, metrics):
    
    plt.rcParams["figure.figsize"] = (20,30)
    colors = ['pink']*show_howmany + ['lightblue']*show_howmany + ['lightgreen']*show_howmany
    pink_patch = mpatches.Patch(color='pink', label="Lowest Scoring Models")
    lightblue_patch = mpatches.Patch(color='lightblue', label='(Random) Middle Scoring Models')
    lightgreen_patch = mpatches.Patch(color='lightgreen', label='Highest Scoring Models')

    for columnName in dataframe.keys():
        
        if columnName == "Image_Name":
            continue
        
        column_filename = columnName.replace(" ", "_").replace("/","-")
        model_data = dataframe[[columnName]]
        
        model_data_sorted = model_data.sort_values(by=[columnName])

        randgraph_idx = np.sort(np.random.randint(show_howmany+1, 
                                len(model_data)-show_howmany+1, 
                                show_howmany).astype('int'))

        lowest_models  = list(model_data_sorted.iloc[0:show_howmany].index)
        middle_models  = list(model_data_sorted.iloc[randgraph_idx].index)
        highest_models = list(model_data_sorted.iloc[-show_howmany:].index)
        

        lowest_models_values  = [metrics[columnName][lowest_model] for lowest_model in lowest_models]
        highest_models_values = [metrics[columnName][highest_model] for highest_model in highest_models]
        middle_models_values  = [metrics[columnName][middle_model] for middle_model in middle_models]
    

        box = plt.boxplot(lowest_models_values+middle_models_values+highest_models_values, 
                            vert=False, patch_artist=True)

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
         
        plt.yticks(ticks=list(range(1,show_howmany*3 + 1)), labels=lowest_models+middle_models+highest_models)
        plt.title(f"{columnName} - Evaluation")
        plt.legend(handles = [lightgreen_patch, lightblue_patch, pink_patch])
        plt.tight_layout()
        
        output_boxplot_path = os.path.join(output_directory, f"{column_filename}.png")
        plt.savefig(output_boxplot_path)
        logger.debug(f"Saved Boxplot - {output_boxplot_path}")
        
        plt.cla()

def main():
    
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

    parser.add_argument('--inputMetrics', dest='inputMetrics', type=str, required=True,
                help='Path to Input Metrics')
    parser.add_argument('--inputCSVs', dest='inputCSVs', type=str, required=True,
                help='Path to the Input CSVs Summary generated from Metric Evaluation')
    parser.add_argument('--evaluationMetric', dest='evaluationMetric', type=str, required=True,
                help='Input Metric Pixel Evaluation or Cellular Evaluation?')
    parser.add_argument('--outputBoxplots', dest='outputBoxplots', type=str, required=True,
                help='Path to Output Boxplots')
    
    args = parser.parse_args()
    input_metrics_dirpath = args.inputMetrics
    input_CSVs_dirpath    = args.inputCSVs
    output_boxplots_dirpath = args.outputBoxplots
    evaluation_metric = args.evaluationMetric
    
    if not os.path.exists(output_boxplots_dirpath):
        raise ValueError(f"Output Directory ({output_boxplots_dirpath}) does not exist")
    
    logger.info(f"Input Metrics Directory : {input_metrics_dirpath}")
    logger.info(f"Output Boxplots Directory : {output_boxplots_dirpath}")
    logger.info(f"Evaluation Metric : {evaluation_metric}")
    
    input_metrics_list = os.listdir(input_metrics_dirpath)
    num_models = len(input_metrics_list)
    
    input_CSVs_avg_path = os.path.join(input_CSVs_dirpath, "avg.csv")
    input_CSVs_df = pd.read_csv(input_CSVs_avg_path, index_col=0)
    num_models = len(input_CSVs_df)

    if evaluation_metric == "PixelEvaluation":
        eval_name = "pixeleval"
    else:
        eval_name = "celleval"
        
    metrics = {metric : {} for metric in input_CSVs_df.columns}

    show_howmany = min(num_models//3, 50)
    
    logger.info(f"\nIterating through {num_models} models ... ")
    logger.info(f"Will be concatenating information from {num_models} models " + \
                    f"to show the top {show_howmany}, middle {show_howmany}, " + \
                        f"and bottom {show_howmany} models")

    counter = 0
    for curr_smp_metric in input_metrics_list:
        
        counter += 1
        logger.info(f"\n{counter}/{num_models}. {curr_smp_metric}")
        
        if curr_smp_metric not in input_CSVs_df.index:
            logger.debug(f"Not Including {curr_smp_metric} - Not in the Dataset")
            continue
        
        input_metric_dirpath = os.path.join(input_metrics_dirpath, curr_smp_metric)
        logger.debug(f"Input Metric Path : {input_metric_dirpath}")
        
        eval_dirpath = os.path.join(input_metric_dirpath, eval_name)
        
        total_stats_result_path = os.path.join(eval_dirpath, "total_stats_result.csv")
        if not os.path.exists(total_stats_result_path):
            logger.debug(f"Not Including {curr_smp_metric} - Total Stats Results was NOT made")
            continue
        logger.debug(f"Total Stats Path : {os.path.abspath(total_stats_result_path)}")
        
        result_path = os.path.join(eval_dirpath, "result.csv")
        logger.debug(f"Result Path : getting data from {os.path.abspath(result_path)}")
    
        result_df = pd.read_csv(result_path)

        for column in result_df.columns:
            if column == "Image_Name":
                continue
            metrics[column][curr_smp_metric] = list(result_df[column])
        
    create_boxplots(input_CSVs_df, output_boxplots_dirpath, show_howmany, metrics)
    logger.info("Done Creating Boxplots!")

main()
