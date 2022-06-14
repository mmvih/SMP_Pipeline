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

def append_to_dataframe(eval_dirpath, curr_smp_metric, metrics):
    
    total_stats_result_path = os.path.join(eval_dirpath, "total_stats_result.csv")
    logger.debug(f"Total Stats Path : {os.path.abspath(total_stats_result_path)}")
    if not os.path.exists(total_stats_result_path):
        return 0
        
    result_path = os.path.join(eval_dirpath, "result.csv")
    logger.debug(f"Result Path : {os.path.abspath(result_path)}")
    
    result_df = pd.read_csv(result_path)
    
    result_df_columns = result_df.columns
    for column in result_df_columns:
        if column == "Image_Name":
            continue
        if column == "Class":
            continue
        if column not in metrics:
            metrics[column] = {curr_smp_metric : list(result_df[column])}
        else:
            metrics[column][curr_smp_metric] = list(result_df[column])
    
    result_df.loc[f'{curr_smp_metric}_mean'] = result_df[0:1249].mean()
    result_df.loc[f'{curr_smp_metric}_std']  = result_df[0:1249].std()
            
    result_df['Image_Name'] = result_df.index
    result_df = result_df.loc[[f'{curr_smp_metric}_mean', f'{curr_smp_metric}_std']]
        
    return result_df

def create_boxplots(dataframe, output_directory, show_howmany, metrics):
    
    plt.rcParams["figure.figsize"] = (20,30)
    colors = ['pink']*show_howmany + ['lightblue']*show_howmany + ['lightgreen']*show_howmany
    pink_patch = mpatches.Patch(color='pink', label="Lowest Scoring Models")
    lightblue_patch = mpatches.Patch(color='lightblue', label='(Random) Middle Scoring Models')
    lightgreen_patch = mpatches.Patch(color='lightgreen', label='Highest Scoring Models')

    for columnName in dataframe.keys():
        
        if columnName == "Image_Name":
            continue
        
        if columnName == "Class":
            continue
        
        column_filename = columnName.replace(" ", "_").replace("/","-")
        model_data = dataframe[[columnName, "Image_Name"]]
        
        # split up the _mean and _std into two separate dataframes
        model_data_avg = model_data[model_data["Image_Name"].str.contains("_mean") == True]
        model_data_std = model_data[model_data["Image_Name"].str.contains("_std")  == True]
        
        # rename Models so that it does not end with _mean or _std
        model_data_avg["Image_Name"] = model_data_avg["Image_Name"].str[:-5]
        model_data_std["Image_Name"] = model_data_std["Image_Name"].str[:-4]
        
        model_data_merged = pd.merge(model_data_avg, model_data_std, on="Image_Name")
        model_data_merged = model_data_merged.sort_values(by=[columnName+"_x"], ignore_index=True)

        columnName_avg = columnName + "_x"
        columnName_std = columnName + "_y"

        randgraph_idx = np.sort(np.random.randint(show_howmany+1, 
                                len(model_data_avg)-show_howmany+1, 
                                show_howmany).astype('int'))
    
        lowest_models  = list(model_data_merged["Image_Name"].iloc[0:show_howmany])
        middle_models  = list(model_data_merged["Image_Name"].iloc[randgraph_idx])
        highest_models = list(model_data_merged["Image_Name"].iloc[-show_howmany:])
        
        print(lowest_models)
        
        lowest_models_values  = [metrics[columnName][lowest_model] for lowest_model in lowest_models]
        highest_models_values = [metrics[columnName][lowest_model] for lowest_model in highest_models]
        middle_models_values  = [metrics[columnName][lowest_model] for lowest_model in middle_models]
    

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
        
        plt.close()
        del box

def main():
    
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

    parser.add_argument('--inputMetrics', dest='inputMetrics', type=str, required=True,
                help='Path to Input Metrics')
    parser.add_argument('--outputBoxplots', dest='outputBoxplots', type=str, required=True,
                help='Path to Output Boxplots')
    
    args = parser.parse_args()
    input_metrics_dirpath = args.inputMetrics
    output_boxplots_dirpath = args.outputBoxplots
    
    if not os.path.exists(output_boxplots_dirpath):
        raise ValueError(f"Output Directory ({output_boxplots_dirpath}) does not exist")
    
    logger.info(f"Input Metrics Directory : {input_metrics_dirpath}")
    logger.info(f"Output Boxplots Directory : {output_boxplots_dirpath}")
    
    input_metrics_list = os.listdir(input_metrics_dirpath)
    num_models = len(input_metrics_list)
    
    show_howmany = min(num_models//3, 50)
    
    counter = 0
    
    # intialize data for summarizing
    pixel_metrics_dataframe = pd.DataFrame()
    cell_metrics_dataframe  = pd.DataFrame()
    pixel_metrics = {}
    cell_metrics = {}
    
    logger.info(f"\nIterating through {num_models} models ... ")
    logger.info(f"Will be concatenating information from {num_models} models " + \
                    f"to show the top {show_howmany}, middle {show_howmany}, " + \
                        f"and bottom {show_howmany} models")

    for curr_smp_metric in input_metrics_list:
        
        counter += 1
        logger.info(f"\n{counter}. {curr_smp_metric}")
        
        input_metric_dirpath = os.path.join(input_metrics_dirpath, curr_smp_metric)
        logger.debug(f"Input Metric Path : {input_metric_dirpath}")
        
        pixeleval_dirpath = os.path.join(input_metric_dirpath, "pixeleval")
        celleval_dirpath  = os.path.join(input_metric_dirpath, "celleval")
        
        pixel_metric_dataframe = append_to_dataframe(pixeleval_dirpath, curr_smp_metric, pixel_metrics)
        cell_metric_dataframe  = append_to_dataframe(celleval_dirpath, curr_smp_metric, cell_metrics)
        
        if not isinstance(pixel_metric_dataframe, int):
            pixel_metrics_dataframe = pixel_metrics_dataframe.append(pixel_metric_dataframe, ignore_index=True)
        if not isinstance(pixel_metric_dataframe, int):
            cell_metrics_dataframe  = cell_metrics_dataframe.append(cell_metric_dataframe, ignore_index=True)
    
    
    output_pixel_boxplot_dirpath = os.path.join(output_boxplots_dirpath, "pixel_boxplots")
    output_cell_boxplot_dirpath  = os.path.join(output_boxplots_dirpath, "cell_boxplots")
    
    if not os.path.exists(output_pixel_boxplot_dirpath):
        os.mkdir(output_pixel_boxplot_dirpath)
        
    if not os.path.exists(output_cell_boxplot_dirpath):
        os.mkdir(output_cell_boxplot_dirpath)
        
    logger.debug(f"Output Pixel Boxplot Directory : {output_pixel_boxplot_dirpath}")
    logger.debug(f"Output Cell Boxplot Directory : {output_cell_boxplot_dirpath}")
    
    create_boxplots(pixel_metrics_dataframe, output_pixel_boxplot_dirpath, show_howmany, pixel_metrics)
    create_boxplots(cell_metrics_dataframe, output_cell_boxplot_dirpath, show_howmany, cell_metrics)


        
    
main()
 