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

ENCODERS = {
    'ResNet': {
        'resnet18': ['imagenet', 'ssl', 'swsl'],
        'resnet34': ['imagenet'],
        'resnet50': ['imagenet', 'ssl', 'swsl'],
        'resnet101': ['imagenet'],
        'resnet152': ['imagenet'],
    },
    'ResNeXt': {
        'resnext50_32x4d': ['imagenet', 'ssl', 'swsl'],
        'resnext101_32x4d': ['ssl', 'swsl'],
        'resnext101_32x8d': ['imagenet', 'instagram', 'ssl', 'swsl'],
        'resnext101_32x16d': ['instagram', 'ssl', 'swsl'],
        'resnext101_32x32d': ['instagram'],
        'resnext101_32x48d': ['instagram'],
    },
    'ResNeSt': {
        'timm-resnest14d': ['imagenet'],
        'timm-resnest26d': ['imagenet'],
        'timm-resnest50d': ['imagenet'],
        'timm-resnest101e': ['imagenet'],
        'timm-resnest200e': ['imagenet'],
        'timm-resnest269e': ['imagenet'],
        'timm-resnest50d_4s2x40d': ['imagenet'],
        'timm-resnest50d_1s4x24d': ['imagenet'],
    },
    'Res2Ne(X)t': {
        'timm-res2net50_26w_4s': ['imagenet'],
        'timm-res2net101_26w_4s': ['imagenet'],
        'timm-res2net50_26w_6s': ['imagenet'],
        'timm-res2net50_26w_8s': ['imagenet'],
        'timm-res2net50_48w_2s': ['imagenet'],
        'timm-res2net50_14w_8s': ['imagenet'],
        'timm-res2next50': ['imagenet'],
    },
    'RegNet(x/y)': {
        'timm-regnetx_002': ['imagenet'],
        'timm-regnetx_004': ['imagenet'],
        'timm-regnetx_006': ['imagenet'],
        'timm-regnetx_008': ['imagenet'],
        'timm-regnetx_016': ['imagenet'],
        'timm-regnetx_032': ['imagenet'],
        'timm-regnetx_040': ['imagenet'],
        'timm-regnetx_064': ['imagenet'],
        'timm-regnetx_080': ['imagenet'],
        'timm-regnetx_120': ['imagenet'],
        'timm-regnetx_160': ['imagenet'],
        'timm-regnetx_320': ['imagenet'],
        'timm-regnety_002': ['imagenet'],
        'timm-regnety_004': ['imagenet'],
        'timm-regnety_006': ['imagenet'],
        'timm-regnety_008': ['imagenet'],
        'timm-regnety_016': ['imagenet'],
        'timm-regnety_032': ['imagenet'],
        'timm-regnety_040': ['imagenet'],
        'timm-regnety_064': ['imagenet'],
        'timm-regnety_080': ['imagenet'],
        'timm-regnety_120': ['imagenet'],
        'timm-regnety_160': ['imagenet'],
        'timm-regnety_320': ['imagenet'],
    },
    'GERNet': {
        'timm-gernet_s': ['imagenet'],
        'timm-gernet_m': ['imagenet'],
        'timm-gernet_l': ['imagenet'],
    },
    'SE-Net': {
        'senet154': ['imagenet'],
        'se_resnet50': ['imagenet'],
        'se_resnet101': ['imagenet'],
        'se_resnet152': ['imagenet'],
        'se_resnext50_32x4d': ['imagenet'],
        'se_resnext101_32x4d': ['imagenet'],
    },
    'SK-ResNe(X)t': {
        'timm-skresnet18': ['imagenet'],
        'timm-skresnet34': ['imagenet'],
        'timm-skresnext50_32x4d': ['imagenet'],
    },
    'DenseNet': {
        'densenet121': ['imagenet'],
        'densenet169': ['imagenet'],
        'densenet201': ['imagenet'],
        'densenet161': ['imagenet'],
    },
    'Inception': {
        'inceptionresnetv2': ['imagenet', 'imagenet+background'],
        'inceptionv4': ['imagenet', 'imagenet+background'],
        'xception': ['imagenet'],
    },
    'EfficientNet': {
        'efficientnet-b0': ['imagenet'],
        'efficientnet-b1': ['imagenet'],
        'efficientnet-b2': ['imagenet'],
        'efficientnet-b3': ['imagenet'],
        'efficientnet-b4': ['imagenet'],
        'efficientnet-b5': ['imagenet'],
        'efficientnet-b6': ['imagenet'],
        'efficientnet-b7': ['imagenet'],
        'timm-efficientnet-b0': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b1': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b2': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b3': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b4': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b5': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b6': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b7': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b8': ['imagenet', 'advprop'],
        'timm-efficientnet-l2': ['noisy-student'],
        'timm-efficientnet-lite0': ['imagenet'],
        'timm-efficientnet-lite1': ['imagenet'],
        'timm-efficientnet-lite2': ['imagenet'],
        'timm-efficientnet-lite3': ['imagenet'],
        'timm-efficientnet-lite4': ['imagenet'],
    },
    'MobileNet': {
        'mobilenet_v2': ['imagenet'],
        'timm-mobilenetv3_large_075': ['imagenet'],
        'timm-mobilenetv3_large_100': ['imagenet'],
        'timm-mobilenetv3_large_minimal_100': ['imagenet'],
        'timm-mobilenetv3_small_075': ['imagenet'],
        'timm-mobilenetv3_small_100': ['imagenet'],
        'timm-mobilenetv3_small_minimal_100': ['imagenet'],
    },
    'DPN': {
        'dpn68': ['imagenet'],
        'dpn68b': ['imagenet+5k'],
        'dpn92': ['imagenet+5k'],
        'dpn98': ['imagenet'],
        'dpn107': ['imagenet+5k'],
        'dpn131': ['imagenet'],
    },
    'VGG': {
        'vgg11': ['imagenet'],
        'vgg11_bn': ['imagenet'],
        'vgg13': ['imagenet'],
        'vgg13_bn': ['imagenet'],
        'vgg16': ['imagenet'],
        'vgg16_bn': ['imagenet'],
        'vgg19': ['imagenet'],
        'vgg19_bn': ['imagenet'],
    },
}

modelName = ['Unet', 'UnetPlusPlus', 'MAnet', 'Linknet', 'FPN', 'PSPNet', 'PAN', 'DeepLabV3', 'DeepLabV3Plus'] #9
lossName = ['MCCLoss'] #, 'JaccardLoss', 'DiceLoss', 'TverskyLoss', 'FocalLoss', 'LovaszLoss', 'SoftBCEWithLogitsLoss', 'SoftCrossEntropyLoss'] #1
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
"vgg16", "vgg16_bn", "vgg19", "vgg19_bn"] #109
encoderWeights = ["random", "imagenet"] #2
optimizerName = ['Adam'] #['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD']  #9 #'LBFGS'
segmentationMode = ["binary"] # + [multilabel, multiclass] #1
trainalbumentations = ["HorizontalFlip,ShiftScaleRotate,PadIfNeeded,RandomCrop," + \
                  "GaussianNoise,Perspective,RandomBrightnessContrast," + \
                  "RandomGamma,Sharpen,Blur,MotionBlur"] # "NA" #1
validablumentations = ["NA"] #1

arguments = {
    "modelName":modelName,
    "lossName":lossName, 
    # "encoderBase":list(ENCODERS.keys()),
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
