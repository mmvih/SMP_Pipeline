import os, sys
import csv
import logging, argparse
# import pandas
import itertools
from itertools import repeat

from tqdm import tqdm
import numpy as np

from multiprocessing import Pool, current_process, Queue 
import torch
import subprocess

import concurrent.futures


available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

modelName = ['Unet', 'UnetPlusPlus', 'MAnet', 'Linknet', 'FPN', 'PSPNet', 'PAN', 'DeepLabV3', 'DeepLabV3Plus']
lossName = ['MCCLoss', 'JaccardLoss', 'DiceLoss', 'TverskyLoss', 'FocalLoss', 'LovaszLoss', 'SoftBCEWithLogitsLoss', 'SoftCrossEntropyLoss']
encoderVariant = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', "resnext50_32x4d", "resnext101_32x4d", 
"resnext101_32x8d", "resnext101_32x16d", "resnext101_32x32d", "resnext101_32x48d", "timm-resnest14d", "timm-resnest26d", 
"timm-resnest50d", "timm-resnest101e", "timm-resnest200e", "timm-resnest269e", "timm-resnest50d_4s2x40d", "timm-resnest50d_1s4x24d",
"timm-res2net50_26w_4s", "timm-res2net101_26w_4s", "timm-res2net50_26w_6s", "timm-res2net50_26w_8s", "timm-res2net50_48w_2s", 
"timm-res2net50_14w_8s", "timm-res2next50", "timm-regnetx_002", "timm-regnetx_004", "timm-regnetx_006", "timm-regnetx_008", 
"timm-regnetx_016", "timm-regnetx_032", "timm-regnetx_040", "timm-regnetx_064", "timm-regnetx_080", "timm-regnetx_120", 
"timm-regnetx_160", "timm-regnetx_320", "timm-regnety_002", "timm-regnety_004", "timm-regnety_006", "timm-regnety_008", 
"timm-regnety_016", "timm-regnety_032", "timm-regnety_040", "timm-regnety_064", "timm-regnety_080",
"timm-regnety_120", "timm-regnety_160", "timm-regnety_320", "timm-gernet_s", "timm-gernet_m", "timm-gernet_l", "senet154", 
"se_resnet50", "se_resnet101", "se_resnet152", "se_resnext50_32x4d", "se_resnext101_32x4d", "timm-skresnet18",
"timm-skresnet34", "timm-skresnext50_32x4d", "densenet121", "densenet169", "densenet201", "densenet161", "inceptionresnetv2", 
"inceptionv4", "xception", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4",
"efficientnet-b5", "efficientnet-b6", "efficientnet-b7", "timm-efficientnet-b0", "timm-efficientnet-b1", "timm-efficientnet-b2", 
"timm-efficientnet-b3", "timm-efficientnet-b4", "timm-efficientnet-b5", "timm-efficientnet-b6",
"timm-efficientnet-b7", "timm-efficientnet-b8", "timm-efficientnet-l2",
"timm-efficientnet-lite0", "timm-efficientnet-lite1", "timm-efficientnet-lite2",
"timm-efficientnet-lite3", "timm-efficientnet-lite4", "mobilenet_v2",
"timm-mobilenetv3_large_075", "timm-mobilenetv3_large_100", "timm-mobilenetv3_large_minimal_100",
"timm-mobilenetv3_small_075", "timm-mobilenetv3_small_100", "timm-mobilenetv3_small_minimal_100",
"dpn68", "dpn68b", "dpn92", "dpn98", "dpn107", "dpn131", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn",
"vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]
encoderWeights = ["random", "imagenet"]
optimizerName = ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'] #'LBFGS'
segmentationMode = ["binary", "multilabel"]
trainalbumentations = ["HorizontalFlip_ShiftScaleRotate_PadIfNeeded_RandomCrop_" + \
                  "GaussianNoise_Perspective_RandomBrightnessContrast_" + \
                  "RandomGamma_Sharpen_Blur_MotionBlur", "NA"]
validablumentations = ["NA"]

arguments = {
    "modelName":modelName,
    "lossName":lossName, 
    "encoderVariant":encoderVariant,
    "encoderWeights":encoderWeights,
    "optimizerName":optimizerName,
    "segmentationMode":segmentationMode,
    "trainAlbumentations":trainalbumentations,
    "validAlbumentations":validablumentations
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

        values = list(arguments.values())
        keys   = list(arguments.keys())
        iter_combos = itertools.product(*values)
        
        with open(csv_path, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(keys)
            for it in iter_combos:
                writer.writerow(it)
            
    except Exception as e:
        print(e)

main()
