import os
import logging,argparse

import subprocess
from concurrent.futures import ThreadPoolExecutor

import bfio
from bfio import BioWriter

polus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polus-plugins")

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger("evaluationPlugins")
logger.setLevel("DEBUG")

"""MAKING PREDICTIONS"""
def evaluation(input_prediction_dirpath,
               input_groundtruth_dirpath,
               output_evaluation_dirpath,
               evaluation_metric):
    
    try:

        logfile = os.path.join(output_evaluation_dirpath, f"{evaluation_metric}_logs.log")
        logfile = open(logfile, 'w')
        python_arguments = f" --GTDir {input_groundtruth_dirpath}" + \
                           f" --inputClasses 1" + \
                           f" --totalStats True" + \
                           f" --outDir {output_evaluation_dirpath}"
        
        if evaluation_metric == "PixelEvaluation":
            predictions_path = os.path.join(input_prediction_dirpath, "predictions")
            python_main = os.path.join(polus_dir, "features/polus-cellular-evaluation-plugin/src/main.py")
            python_command = f"python {python_main}" + python_arguments + \
                                f" --totalSummary True" + \
                                f" --PredDir {predictions_path}"
        else:
            ftl_path = os.path.join(input_prediction_dirpath, "ftl")
            python_main = os.path.join(polus_dir, "features/polus-pixelwise-evaluation-plugin/src/main.py")
            python_command = f"python {python_main}" + python_arguments + \
                                f" --individualStats False" + \
                                f" --PredDir {ftl_path}"
        
        subprocess.call(python_command, shell=True, stdout=logfile, stderr=logfile)
        logger.debug(python_command)

    except Exception as e:
        print(e)



def main():
    
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')
    
    
    parser.add_argument('--inputPredictions', dest='inputPredictions', type=str, required=True, \
                        help='Path to Directory containing Input Predictions for all Models')
    parser.add_argument('--inputGroundtruth', dest='inputGroundtruth', type=str, required=True,
                        help='Path to Groundtruth Directory that match the inputPredictions')
    parser.add_argument('--outputMetrics', dest='outputMetrics', type=str, required=True,
                        help='Path to where the Output Metrics will be saved')
    parser.add_argument('--evaluationMetric', dest='evaluationMetric', type=str, required=True,
                        help="Running either Pixel Evaluation or Cellular Evaluation Plugin")

    args = parser.parse_args()
    input_predictions_dirpath = args.inputPredictions
    input_groundtruth_dirpath = args.inputGroundtruth
    output_metrics_dirpath    = args.outputMetrics
    evaluation_metric      = args.evaluationMetric
    
    logger.info(f"Input Predictions Directory : {input_predictions_dirpath}")
    logger.info(f"Input Groundtruth Directory : {input_groundtruth_dirpath}")
    logger.info(f"Output Metrics Path : {output_metrics_dirpath}")
    logger.info(f"Evaluation Metric : {evaluation_metric}")
    
    input_predictions_list = os.listdir(input_predictions_dirpath)
    num_models = len(input_predictions_list)

    input_groundtruth_list = os.listdir(input_groundtruth_dirpath)
    num_examples = len(input_groundtruth_list)
    
    counter = 0
    logger.info(f"\nIterating through {num_models} models ...")
    logger.info(f"Each model will be generating {num_examples} predictions")
    with ThreadPoolExecutor(max_workers=os.cpu_count()-10) as executor:
        for curr_smp_model in input_predictions_list:
            
            counter += 1
            logger.info(f"\n{counter}. {curr_smp_model}")
            
            input_prediction_dirpath = os.path.join(input_predictions_dirpath, curr_smp_model)
            output_metric_dirpath = os.path.join(output_metrics_dirpath, curr_smp_model)
            logger.debug(f"Input Prediction Path : {input_prediction_dirpath}")
            logger.debug(f"Output Label Path : {output_metric_dirpath}") 
        
            output_evaluation_dirpath = os.path.join(output_metric_dirpath, evaluation_metric)
            total_stats_result_path = os.path.join(output_evaluation_dirpath, "total_stats_result.csv")
            if os.path.exists(total_stats_result_path):
                logger.debug(f"Not Running {counter}/{num_models} - output already exists ({total_stats_result_path})")
                continue

            if not os.path.exists(output_evaluation_dirpath):
                os.mkdir(output_evaluation_dirpath)

            executor.submit(evaluation, 
                            input_prediction_dirpath,
                            input_groundtruth_dirpath,
                            output_evaluation_dirpath,
                            evaluation_metric)

            logger.info(f"analyzed {counter}/{num_models} models\n")
    logger.info(f"DONE ANALYZING ALL MODELS!")
    
main()