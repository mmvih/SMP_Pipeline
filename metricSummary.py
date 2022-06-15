import os, sys, json
import logging, argparse

import numpy as np
import pandas as pd

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("savepredictions")
logger.setLevel("DEBUG")

def append_to_dataframe(eval_dirpath, curr_smp_metric):
    
    total_stats_result_path = os.path.join(eval_dirpath, "total_stats_result.csv")
    logger.debug(f"Total Stats Path : {os.path.abspath(total_stats_result_path)}")
    if not os.path.exists(total_stats_result_path):
        return 0, 0, 0
        
    result_path = os.path.join(eval_dirpath, "result.csv")
    logger.debug(f"Result Path : {os.path.abspath(result_path)}")
    
    result_df = pd.read_csv(result_path)
    
    result_df_avg = result_df[0:1249].mean()
    result_df_std = result_df[0:1249].std()
    
    result_df.loc[f'{curr_smp_metric}_mean'] = result_df_avg
    result_df.loc[f'{curr_smp_metric}_std']  = result_df_std
    
    result_df_avg = result_df_avg.rename(curr_smp_metric)
    result_df_std = result_df_std.rename(curr_smp_metric)
            
    result_df['Image_Name'] = result_df.index
    result_df = result_df.loc[[f'{curr_smp_metric}_mean', f'{curr_smp_metric}_std']]
        
    return result_df, result_df_avg, result_df_std

def main():
    
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

    parser.add_argument('--inputMetrics', dest='inputMetrics', type=str, required=True,
                help='Path to Input Metrics')
    parser.add_argument('--evaluationMetric', dest='evaluationMetric', type=str, required=True,
                help='Input Metric Pixel Evaluation or Cellular Evaluation?')
    parser.add_argument('--outputCSVs', dest='outputCSVs', type=str, required=True,
                help='Path to Output Boxplots')
    
    args = parser.parse_args()
    input_metrics_dirpath = args.inputMetrics
    output_CSVs_dirpath = args.outputCSVs
    evaluation_metric = args.evaluationMetric
    
    if not os.path.exists(output_CSVs_dirpath):
        raise ValueError(f"Output Directory ({output_CSVs_dirpath}) does not exist")
    
    logger.info(f"Input Metrics Directory : {input_metrics_dirpath}")
    logger.info(f"Output Boxplots Directory : {output_CSVs_dirpath}")
    logger.info(f"Evaluation Metric : {evaluation_metric}")
    
    input_metrics_list = os.listdir(input_metrics_dirpath)
    num_models = len(input_metrics_list)
    
    show_howmany = min(num_models//3, 50)
    
    if evaluation_metric == "PixelEvaluation":
        eval_name = "pixeleval"
    else:
        eval_name = "celleval"
        
    
    # intialize data for summarizing
    metrics_dataframe = pd.DataFrame()
    metrics = {}
    
    logger.info(f"\nIterating through {num_models} models ... ")
    logger.info(f"Will be concatenating information from {num_models} models " + \
                    f"to show the top {show_howmany}, middle {show_howmany}, " + \
                        f"and bottom {show_howmany} models")

    metrics_dataframe_avg = pd.DataFrame()
    metrics_dataframe_std = pd.DataFrame()
    
    counter = 0
    for curr_smp_metric in input_metrics_list:
        
        counter += 1
        logger.info(f"\n{counter}/{num_models}. {curr_smp_metric}")
        
        input_metric_dirpath = os.path.join(input_metrics_dirpath, curr_smp_metric)
        logger.debug(f"Input Metric Path : {input_metric_dirpath}")
        
        eval_dirpath = os.path.join(input_metric_dirpath, eval_name)

        metric_dataframe, metric_dataframe_avg, metric_dataframe_std = \
            append_to_dataframe(eval_dirpath, curr_smp_metric)

        if not isinstance(metric_dataframe, int):
            metrics_dataframe = metrics_dataframe.append(metric_dataframe, ignore_index=True)
            metrics_dataframe_avg = metrics_dataframe_avg.append(metric_dataframe_avg)
            metrics_dataframe_std = metrics_dataframe_std.append(metric_dataframe_std)


    # saving the average data
    metrics_dataframe_avg.to_csv(os.path.join(output_CSVs_dirpath, "avg.csv"), index=True)

    # saving the standard deviation data
    metrics_dataframe_std.to_csv(os.path.join(output_CSVs_dirpath, "std.csv"), index=True)

    logger.debug(f"\n Done Generating Output CSVs")
    
main()
 