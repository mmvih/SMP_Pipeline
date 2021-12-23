import os, json
import logging
import argparse
import numpy as np
import copy

from sklearn.metrics import fbeta_score, jaccard_score

import torch
from torchsummary import summary
from torch.utils.data import Dataset

import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn

import albumentations 

import bfio
from bfio import BioWriter, BioReader

from tqdm import tqdm
from torch import Tensor
import torchvision

from typing import Union

class LocalNorm(object):
    def __init__(
        	self, 
         	window_size: int = 129,
            max_response: Union[int, float] = 6,
    ):
        assert window_size % 2 == 1, 'window_size must be an odd integer'

        self.window_size: int = window_size
        self.max_response: float = float(max_response)
        self.pad = torchvision.transforms.Pad(window_size // 2 + 1, padding_mode='reflect')
        # Mode can be 'test', 'train' or 'eval'.
        self.mode: str = 'eval'

    def __call__(self, x: Tensor):
        return torch.clip(
            self.local_response(self.pad(x)),
            min=-self.max_response,
            max=self.max_response,
        )

    def image_filter(self, image: Tensor) -> Tensor:
        """ Use a box filter on a stack of images
        This method applies a box filter to an image. The input is assumed to be a
        4D array, and should be pre-padded. The output will be smaller by
        window_size - 1 pixels in both width and height since this filter does not pad
        the input to account for filtering.
        """
        integral_image: Tensor = image.cumsum(dim=-1).cumsum(dim=-2)
        return (
                integral_image[..., :-self.window_size - 1, :-self.window_size - 1]
                + integral_image[..., self.window_size:-1, self.window_size:-1]
                - integral_image[..., self.window_size:-1, :-self.window_size - 1]
                - integral_image[..., :-self.window_size - 1, self.window_size:-1]
        )

    def local_response(self, image: Tensor):
        """ Regional normalization.
        This method normalizes each pixel using the mean and standard deviation of
        all pixels within the window_size. The window_size parameter should be
        2 * radius + 1 of the desired region of pixels to normalize by. The image should
        be padded by window_size // 2 on each side.
        """
        local_mean: Tensor = self.image_filter(image) / (self.window_size ** 2)
        local_mean_square: Tensor = self.image_filter(image.pow(2)) / (self.window_size ** 2)

        # Use absolute difference because sometimes error causes negative values
        local_std = torch.clip(
            (local_mean_square - local_mean.pow(2)).abs().sqrt(),
            min=1e-3,
        )

        min_i, max_i = self.window_size // 2, -self.window_size // 2 - 1
        response = image[..., min_i:max_i, min_i:max_i]

        return (response - local_mean) / local_std

def add_to_axis(image, groundtruth, threshold, axis=None):

	
	new_img = copy.deepcopy(image)
	new_img[image > threshold]  = 1
	new_img[image <= threshold] = 0

	groundtruth = groundtruth.ravel()
	unravel_new_img = new_img.ravel()
	f1_score = fbeta_score(y_true=groundtruth, 
							y_pred=unravel_new_img,
							average=None,
							beta=1, zero_division='warn')
	f1_score = np.around(np.average(f1_score), 4)

	j_score  = jaccard_score(y_true=groundtruth,
								y_pred=unravel_new_img,
								average=None, zero_division='warn')
	j_score  = np.around(np.average(j_score), 4)
	# print(f1_score, j_score)

	threshold = round(threshold, 4)
	if axis != None:
		axis.imshow(new_img)
		axis.set_title(f"Threshold: {threshold} - F1: {f1_score}, JACCARD: {j_score}")

	return f1_score, j_score

class DatasetforPytorch(Dataset):
    
    def __init__(self, 
                images_dir,
                masks_dir,
                preprocessing=None,
                augmentations=None):

        self.images_fps = [os.path.join(images_dir, image) for image in os.listdir(images_dir)]
        self.masks_fps =  [os.path.join(masks_dir, mask)   for mask in os.listdir(masks_dir)]
        self.preprocessing = preprocessing # this is a function that is getting intialized
        self.augmentations = augmentations # this is a function that is getting initialized

    def __getitem__(self, i):

        image          = np.array(Image.open(self.images_fps[i]))
        mask           = np.array(Image.open(self.masks_fps[i]))
        image_shape    = image.shape
        mask_shape     = mask.shape
        # mask[mask > 0] = 1 

        if self.augmentations:
            sample = self.augmentations(image=image, mask=mask)
            image, mask = sample['image'], sample['mask'] 

        if self.preprocessing:
            image = self.preprocessing(image).numpy()

        image          = np.reshape(image, (1, image_shape[0], image_shape[1])).astype("float32")
        assert np.isnan(image).any() == False
        assert np.isinf(image).any() == False

        
        mask           = np.reshape(mask, (1, mask_shape[0], mask_shape[1])).astype("float32")
        assert np.isnan(mask).any() == False
        assert np.isinf(mask).any() == False



        return image, mask

    def __len__(self):
        return(len(self.images_fps))

def main():

	parser = argparse.ArgumentParser(prog='main', description='Segmentation Models Testing')
 
	parser.add_argument('--modelDir', dest='modelDir', type=str, required=True, \
     					help="Path to model to load for testing.")
	parser.add_argument('--imagesTestDir', dest='imagesTestDir', type=str, required=True, \
     					help="Path to Images Test Directory")
	parser.add_argument('--labelsTestDir', dest="labelsTestDir", type=str, required=False, \
     					help="Path to Labels Test Directory")

	
	args = parser.parse_args()
	modelDir = args.modelDir
	imagesTestDir = args.imagesTestDir 
	labelsTestDir = args.labelsTestDir
 
	num_images = os.listdir(imagesTestDir)
	num_labels = os.listdir(labelsTestDir)
	assert num_images == num_labels
 

 
	bestmodelPath = os.path.join(modelDir, "model.pth")
	configPath = os.path.join(modelDir, "config.json")
	configObj = open(configPath, 'r')
	configDict = json.load(configObj)
 
	print(configDict)
	
	
	preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            LocalNorm(window_size=257)])
	# if configDict["encoder_weights"] == "imagenet":
	# 	preprocessing = smp.encoders.get_preprocessing_fn(configDict["encoder_variant"], pretrained='imagenet')
    
	test_dataset_vis = DatasetforPytorch(images_dir=imagesTestDir, masks_dir=labelsTestDir, \
     	preprocessing=None)
	test_dataset = DatasetforPytorch(images_dir=imagesTestDir, masks_dir=labelsTestDir, \
     	preprocessing=preprocessing)

	MODELS = {
		'Unet': smp.Unet,
		'UnetPlusPlus': smp.UnetPlusPlus,
		'MAnet': smp.MAnet,
		'Linknet': smp.Linknet,
		'FPN': smp.FPN,
		'PSPNet': smp.PSPNet,
		'PAN': smp.PAN,
		'DeepLabV3': smp.DeepLabV3,
		'DeepLabV3Plus': smp.DeepLabV3Plus,
	}
	 
 
	modelPath = os.path.join(modelDir, "model.pth")
	outputTestDir = os.path.join(modelDir, "testedImages")
	if not os.path.exists(outputTestDir):
		os.mkdir(outputTestDir)
	bestmodel = torch.load(bestmodelPath)
	bestmodel.to(torch.device('cpu'))
	# bestmodel = MODELS[bestmodeldata['model_name']](
	# 	encoder_name=bestmodeldata["encoder_variant"],
	# 	encoder_weights=bestmodeldata["encoder_weights"],
	# 	in_channels=1,
	# 	activation='sigmoid'
	# )
	sig = nn.Sigmoid()
 
 
	max_f1 = {}
	max_j  = {}
	n_vals = []
 
	outputText = os.path.join(outputTestDir, "Averages.txt")
	with open(outputText, 'a+') as writeOutput:
		for i in tqdm(range(5)):
			n = np.random.choice(len(test_dataset_vis))
			n_vals.append(n)
	
			image_vis = test_dataset_vis[n][0]
			image, gt_mask = test_dataset[n]
			
			gt_mask = gt_mask.squeeze()
			gt_mask[gt_mask > 0] = 1 
	
			x_tensor = torch.from_numpy(image).to('cpu').unsqueeze(0)
			pr_mask = bestmodel.predict(x_tensor)

			# pr_mask = sig(pr_mask) # need to make predicitions range from 0 to 1
			pr_mask = pr_mask.squeeze().cpu().numpy()
			print(np.min(pr_mask), np.max(pr_mask))
			pr_mask_shape = pr_mask.shape

			
			thres_min = np.min(pr_mask)
			thres_max = np.max(pr_mask)
			assert thres_max <= 1, f"Max Prediction value is greater than 1: {thres_max}"
			assert thres_min >= 0, f"Min Prediction value is less than 0: {thres_min}"
			thres_range = thres_max-thres_min
			thresholds = np.arange(thres_min, thres_max, thres_range/10)

			fig, ((ax_img, ax_groundtruth, ax_prediction), 
			(ax_pred1, ax_pred2, ax_pred3), 
			(ax_pred4, ax_pred5, ax_pred6),
			(ax_pred7, ax_pred8, ax_pred9))= plt.subplots(4, 3, figsize = (24, 24))

			ax_img.imshow(image_vis.squeeze())
			ax_img.set_title("Image")	
			ax_groundtruth.imshow(gt_mask.squeeze())
			ax_groundtruth.set_title("Groundtruth")
			ax_prediction.imshow(pr_mask)
			ax_prediction.set_title("Prediction Channel 0")

			f1_score_1, j_score_1 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=thresholds[1], axis=ax_pred1)
			f1_score_2, j_score_2 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=thresholds[2], axis=ax_pred2)
			f1_score_3, j_score_3 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=thresholds[3], axis=ax_pred3)
			f1_score_4, j_score_4 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=thresholds[4], axis=ax_pred4)
			f1_score_5, j_score_5 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=thresholds[5], axis=ax_pred5)
			f1_score_6, j_score_6 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=thresholds[6], axis=ax_pred6)
			f1_score_7, j_score_7 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=thresholds[7], axis=ax_pred7)
			f1_score_8, j_score_8 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=thresholds[8], axis=ax_pred8)
			f1_score_9, j_score_9 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=thresholds[9], axis=ax_pred9)

			list_threshold_f1 = [f1_score_1, f1_score_2, f1_score_3, f1_score_4, \
									f1_score_5, f1_score_6, f1_score_7, f1_score_8, f1_score_9]
			list_threshold_j = [j_score_1, j_score_2, j_score_3, j_score_4, \
									j_score_5, j_score_6, j_score_7, j_score_8, j_score_9]
			threshold_max_f1 = max(list_threshold_f1)
			threshold_max_j  = max(list_threshold_j)
			max_f1[i] = threshold_max_f1
			max_j[i]  = threshold_max_j

			fig.suptitle(f"Testing Image {n}")
			plot_name = os.path.join(outputTestDir, f"testingimage_{n}")
			plt.savefig(plot_name)
  
		average_f1 = np.average(list(max_f1.values()))
		average_j  = np.average(list(max_j.values()))

		max_f1 = sorted(max_f1.items(), key=lambda kv: kv[1])
		max_j  = sorted(max_j.items(), key=lambda kv: kv[1])


		writeOutput.write(f"Averages for {n_vals}\n")
		writeOutput.write(f"Average F1: {average_f1}\n")
		writeOutput.write(f"Average Jaccard: {average_j}\n")
		writeOutput.write("MAX F1 Sorted\n")
		writeOutput.write(str(max_f1) + "\n")
		writeOutput.write("MAX J Sorted\n")
		writeOutput.write(str(max_j))

main()
