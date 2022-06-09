import os
import logging,argparse

import subprocess
import concurrent.futures

import bfio
from bfio import BioWriter

polus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polus-plugins")

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("ftl")
logger.setLevel("DEBUG")


"""MAKING MULTI INSTANCE LABELS"""
def evaluation(prediction_input_path : str,
               ftl_output_path : str):
    
    try:

        assert os.path.exists(prediction_input_path)
        
        if not os.path.exists(ftl_output_path):
            os.mkdir(ftl_output_path)
        
        python_ftl_main = os.path.join(polus_dir, "transforms/images/polus-ftl-label-plugin/src/main.py")
        python_command = f"python {python_ftl_main}" + \
                         f"--inpDir {prediction_input_path} " + \
                         f"--outDir {ftl_output_path} " + \
                         f"--connectivity 1"

        ftl_logpath = os.path.join(os.path.dirname(os.path.abspath(ftl_output_path)), "ftl_logs.log")
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
    parser.add_argument('--inputGroundtruth', dest='inputGroundtruth', type=str, required=False,
                        help='Path to Groundtruth Directory that match the inputPredictions')
    parser.add_argument('--outputLabels', dest='outputLabels', type=str, required=True,
                        help='Path to where the Output Labels will be saved')

    args = parser.parse_args()
    input_predictions_path = args.inputPredictions
    input_groundtruth_path = args.inputGroundtruth
    output_labels_path     = args.outputLabels
    
    logger.info(f"Input Predictions Directory : {input_predictions_path}")
    logger.info(f"Input Groundtruth Directory : {input_groundtruth_path}")
    logger.info(f"Output Labels Directory : {output_labels_path}")
    
    input_predictions_list = os.listdir(input_predictions_path)
    num_models = len(input_predictions_list)
    logger.info(f"There are {num_models} to iterate through")
    
    input_groundtruth_list = os.listdir(input_groundtruth_path)
    num_examples = len(input_groundtruth_list)
    logger.info(f"Each model has generated {num_examples} predictions")

    iter_smp_inputs_list = iter(input_predictions_list)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-10) as executor:
        while counter != len(input_predictions_list)-1:
            
            curr_smp_model = next(iter_smp_inputs_list)
            logger.info(f"Looking at {curr_smp_model}")

            # input and output path for the models
            input_smp_path  = os.path.join(input_predictions_path, curr_smp_model)
            output_smp_path = os.path.join(output_labels_path, curr_smp_model)
            
            # make sure there are input predictions for the current segmentation model that is being analyzed
            prediction_input_path = os.path.join(input_smp_path, "predictions")
            num_predictions = len(os.listdir(prediction_input_path))
            if os.path.exists(prediction_input_path):
                if num_predictions != num_examples:
                    logger.debug(f"Not Running - there are {num_predictions} and not {num_examples}")
                    counter += 1
                    continue
            else:
                logger.debug(f"{curr_smp_model} does not have a prediction directory at {prediction_input_path}")
                counter += 1
                continue
            
            # if theres already an output, then do not waste time/resources rerunning it
            ftl_output_path = os.path.join(output_smp_path, "ftl")
            num_ftls = len(os.listdir(ftl_output_path))
            if os.path.exists(ftl_output_path):
                if num_ftls == num_examples:
                    logger.debug(f"Not Running - output already exists with {num_examples} for {output_smp_path}")
                    counter += 1
                    continue
            
            executor.submit(evaluation, prediction_input_path, ftl_output_path)
            counter = counter + 1
            logger.info(f"analyzed {counter}/{num_models-1} models\n")
            break
            
main()