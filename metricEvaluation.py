import os
import logging,argparse

import subprocess
import concurrent.futures

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
def evaluation(input_prediction_path,
               input_groundtruth_path,
               output_evaluation_path,
               evaluation_metric):
    
    try:

        if not os.path.exists(output_evaluation_path):
            os.mkdir(output_evaluation_path)

        logfile = os.path.join(output_evaluation_path, f"{evaluation_metric}_logs.log")
        logfile = open(logfile, 'w')
        python_arguments = f" --GTDir {input_groundtruth_path}" + \
                           f" --inputClasses 1" + \
                           f" --totalStats True" + \
                           f" --outDir {output_evaluation_path}"
        
        if evaluation_metric == "PixelEvaluation":
            prediction_path = os.path.join(input_prediction_path, "predictions")
            python_main = os.path.join(polus_dir, "features/polus-cellular-evaluation-plugin/src/main.py")
            python_command = f"python {python_main}" + python_arguments + \
                                f" --totalSummary True" + \
                                f" --PredDir {prediction_path}"
        else:
            prediction_path = os.path.join(input_prediction_path, "ftl")
            python_main = os.path.join(polus_dir, "features/polus-pixelwise-evaluation-plugin/src/main.py")
            python_command = f"python {python_main}" + python_arguments + \
                                f" --individualStats False" + \
                                f" --PredDir {prediction_path}"
        
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
    input_predictions_path = args.inputPredictions
    input_groundtruth_path = args.inputGroundtruth
    output_metrics_path    = args.outputMetrics
    evaluation_metric      = args.evaluationMetric
    
    logger.info(f"Input Predictions Directory : {input_predictions_path}")
    logger.info(f"Input Groundtruth Directory : {input_groundtruth_path}")
    logger.info(f"Output Metrics Path : {output_metrics_path}")
    logger.info(f"Evaluation Metric : {evaluation_metric}")
    
    input_predictions_list = os.listdir(input_predictions_path)
    num_models = len(input_predictions_list)
    logger.info(f"There are {num_models} to iterate through")
    
    input_groundtruth_list = os.listdir(input_groundtruth_path)
    num_examples = len(input_groundtruth_list)
    logger.info(f"Each model has generated {num_examples} predictions")

    counter = 0
    logger.info("\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-10) as executor:
        for curr_smp_model in input_predictions_list:
            
            logger.info(f"Looking at {curr_smp_model}")
            
            input_prediction_path = os.path.join(input_predictions_path, curr_smp_model)
            output_metric_path = os.path.join(output_metrics_path, curr_smp_model)
            logger.debug(f"Input Predictions Path : {input_prediction_path}")
            logger.debug(f"Output Labels Path : {output_metric_path}") 
        
            output_evaluation_path = os.path.join(output_metric_path, evaluation_metric)
            total_stats_result_path = os.path.join(output_evaluation_path, "total_stats_result.csv")
            if os.path.exists(total_stats_result_path):
                counter = counter + 1
                logger.debug(f"Not Running {counter}/{num_models} - output already exists ({total_stats_result_path})")
                continue


            output_evaluation_path = os.path.join(output_metric_path, evaluation_metric)
            total_stats_result_path = os.path.join(output_evaluation_path, "total_stats_result.csv")
            if os.path.exists(total_stats_result_path):
                logger.debug(f"Not Running {counter}/{num_models} - output already exists ({total_stats_result_path})")
            else:
                executor.submit(evaluation, 
                                input_prediction_path,
                                input_groundtruth_path,
                                output_evaluation_path,
                                evaluation_metric)

            counter = counter + 1
            logger.info(f"analyzed {counter}/{num_models} models\n")
            
    logger.info(f"DONE ANALYZING ALL MODELS")
    
main()