import os, sys
import csv
import logging, argparse
import itertools

polus_smp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polus-plugins/segmentation/polus-smp-training-plugin/")
sys.path.append(polus_smp_dir)

from src.utils import MODELS
from src.utils import LOSSES
from src.utils import ENCODERS
from src.utils import OPTIMIZERS

# all potential combinations
modelName = list(MODELS.keys())

lossName = list(LOSSES.keys())

encoderBase = list(ENCODERS.keys())

encoderVariant = [variant for base in encoderBase for variant in ENCODERS[base]]

encoderWeights = list(set(itertools.chain.from_iterable([weight for variant in ENCODERS.values() for weight in variant.values()])))
encoderWeights.append("random")

optimizerName = list(OPTIMIZERS.keys())

# only interested with these combinations
lossName = ["MCCLoss"]
encoderWeights = ["random", "imagenet"] #2
optimizerName = ['Adam'] 

# summarize into dictionary
arguments = {
    "modelName":modelName,
    "lossName":lossName,
    "encoderVariant":encoderVariant,
    "encoderWeights":encoderWeights,
    "optimizerName":optimizerName,
}


logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("pipeline")
logger.setLevel("INFO")


track_changes = True
def main():
    try:
        """ Argument parsing """
        logger.info("Parsing arguments...")
        parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

        parser.add_argument('--csvFile', dest='csvFile', type=str, required=True, \
                            help='Path to csv File')

        args = parser.parse_args()
        csv_path = args.csvFile
        logger.info(f"CSV: {os.path.abspath(csv_path)}\n")

        values = list(arguments.values())
        keys   = list(arguments.keys())
        iter_combos = itertools.product(*values)
        
        for argument in arguments:
            logger.info(f"{argument} has {len(arguments[argument])} options")
        

        with open(csv_path, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(keys)
            for it in iter_combos:
                logger.debug(f"Appending : {it}")
                writer.writerow(it)
            
    except Exception as e:
        print(f"ERROR: {e}")

main()
