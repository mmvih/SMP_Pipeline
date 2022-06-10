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
logger = logging.getLogger("ftl")
logger.setLevel("DEBUG")


"""MAKING MULTI INSTANCE LABELS"""
def evaluation(predictions_path : str,
               ftl_path : str):
    
    try:

        assert os.path.exists(predictions_path)
        
        assert os.path.dirname(predictions_path) == os.path.dirname(ftl_path)
        
        python_ftl_main = os.path.join(polus_dir, "transforms/images/polus-ftl-label-plugin/src/main.py")
        python_command = f"python {python_ftl_main} " + \
                         f"--inpDir {predictions_path} " + \
                         f"--outDir {ftl_path} " + \
                         f"--connectivity 1"

        ftl_logpath = os.path.join(os.path.dirname(os.path.abspath(ftl_path)), "ftl_logs.log")
        ftl_logfile = open(ftl_logpath, 'w')
        subprocess.call(python_command, shell=True, stdout=ftl_logfile, stderr=ftl_logfile)

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
    parser.add_argument('--outputLabels', dest='outputLabels', type=str, required=True,
                        help='Path to where the Output Labels will be saved')

    args = parser.parse_args()
    input_predictions_dirpath = args.inputPredictions
    input_groundtruth_dirpath = args.inputGroundtruth
    output_labels_dirpath     = args.outputLabels
    
    logger.info(f"Input Predictions Directory : {input_predictions_dirpath}")
    logger.info(f"Input Groundtruth Directory : {input_groundtruth_dirpath}")
    logger.info(f"Output Labels Directory : {output_labels_dirpath}")
    
    input_groundtruth_list = os.listdir(input_groundtruth_dirpath)
    num_examples = len(input_groundtruth_list)

    input_predictions_list = os.listdir(input_predictions_dirpath)
    num_models = len(input_predictions_list)

    counter = 0
    logger.info(f"\nIterating through {num_models} models ...")
    logger.info(f"Each model will be generating {num_examples} predictions")
    with ThreadPoolExecutor(max_workers=os.cpu_count()-10) as executor:
        for curr_smp_model in input_predictions_list:
            
            counter += 1
            logger.info(f"\n{counter}. {curr_smp_model}")

            # input and output path for the models
            input_prediction_dirpath  = os.path.join(input_predictions_dirpath, curr_smp_model)
            output_label_dirpath = os.path.join(output_labels_dirpath, curr_smp_model)
            logger.debug(f"Input Predictions Path : {input_prediction_dirpath}")
            logger.debug(f"Output Labels Path : {output_label_dirpath}")
            
            # make sure there are expect number of input predictions for the 
            #   current segmentation model that is being analyzed
            predictions_path = os.path.join(input_prediction_dirpath, "predictions")
            if os.path.exists(predictions_path):
                num_predictions = len(os.listdir(predictions_path))
                if num_predictions < num_examples:
                    logger.debug(f"Not Running {counter}/{num_models} - there are {num_predictions} and not {num_examples}\n")
                    continue
            else:
                logger.debug(f"Not running {counter}/{num_models} - {curr_smp_model} does not have a prediction directory at {predictions_path}\n")
                continue
            
            # if theres already an output, then do not waste time/resources rerunning it
            ftl_path = os.path.join(output_label_dirpath, "ftl")
            if os.path.exists(ftl_path):
                num_ftls = len(os.listdir(ftl_path))
                if num_ftls >= num_examples:
                    logger.debug(f"Not Running {counter}/{num_models} - output already exists with {num_examples} for {output_label_dirpath}\n")
                    continue
            
            if not os.path.exists(ftl_path):
                os.mkdir(ftl_path)
            
            executor.submit(evaluation, predictions_path, ftl_path)
            logger.info(f"analyzed {counter}/{num_models} models\n") 
        logger.info(f"DONE ANALYZING ALL MODELS!")
            
main()