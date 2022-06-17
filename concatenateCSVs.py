import os,sys
import logging,argparse

import pandas as pd

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("concatenateCSVs")
logger.setLevel("INFO")

def main():
    
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

    parser.add_argument('--pixelAvg_csvdir', dest='pixelAvg_csvdir', type=str, required=True, \
        help="Path to Directory containing the avg.csv for Pixel Evaluations")
    parser.add_argument('--cellAvg_csvdir', dest='cellAvg_csvdir', type=str, required=True,
        help="Path to Directory containing the avg.csv for Cell Evaluations")
    parser.add_argument('--outputCSV', dest='outputCSV', type=str, required=True,
        help="Path to Output CSV Directory")
    
    args = parser.parse_args()
    pixelavg_csv_dirpath = args.pixelAvg_csvdir
    cellavg_csv_dirpath  = args.cellAvg_csvdir
    output_csv_dirpath   = args.outputCSV
    
    pixelavg_csv_path = os.path.join(pixelavg_csv_dirpath, "avg.csv")
    cellavg_csv_path  = os.path.join(cellavg_csv_dirpath, "avg.csv")
    
    if not os.path.exists(pixelavg_csv_path):
        raise ValueError(f"{pixelavg_csv_path} does not exist!")

    if not os.path.exists(cellavg_csv_path):
        raise ValueError(f"{cellavg_csv_path} does not exist!")
    
    if not os.path.exists(output_csv_dirpath):
        raise ValueError(f"{output_csv_dirpath} does not exist!")
    
    logger.info(f"Input Pixel Average CSV : {pixelavg_csv_path}")
    logger.info(f"Input Cell Average CSV  : {cellavg_csv_path}")
    logger.info(f"Output Directory : {output_csv_dirpath}")
    
    pixel_dataframe = pd.read_csv(pixelavg_csv_path, index_col=0)
    pixel_dataframe = pixel_dataframe.rename(columns={pixel_col : pixel_col + "_pixel" 
                                                        for pixel_col in pixel_dataframe.columns})
    
    cell_dataframe  = pd.read_csv(cellavg_csv_path, index_col=0)
    cell_dataframe  = cell_dataframe.rename(columns={cell_col: cell_col + "_cell"
                                                        for cell_col in cell_dataframe})
    
    summarized_dataframe = pd.concat([pixel_dataframe, cell_dataframe], axis=1)
    summarized_dataframe.to_csv(os.path.join(output_csv_dirpath, "pixel_cell_avg.csv"), index=True)
    
main()
