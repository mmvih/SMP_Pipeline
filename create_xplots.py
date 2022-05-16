# %%
import os, sys, json
import pandas as pd
import numpy as np

import shutil

import matplotlib.pyplot as plt
from PIL import Image

# %%
output_metrics_path = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_metrics"

testingoutputs_dir = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ftl"
ftl_dir = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/ftl_outputs/"
predictions_dir = "predictions"
groundtruth_dir = "groundtruths"
groundtruth_labeldir = "/home/vihanimm/SegmentationModelToolkit/Data/ometif_data/nuclear/test/groundtruth/"

# %%
output_metrics_modellist = os.listdir(output_metrics_path)

# %%
num_tests = 5

# %%
fig, ax = plt.subplots(num_tests, 4, figsize = (4*4, num_tests*4), constrained_layout=True)

for model in output_metrics_modellist:
    
    if ("UnetPlusPlus-MCCLoss-vgg16-imagenet-Adam" not in model):
        continue
    
    print(model)
    model_outputs_path = os.path.join(testingoutputs_dir, model)
    model_predictions_dir = os.path.join(model_outputs_path, predictions_dir)
    model_groundtruth_dir = os.path.join(model_outputs_path, groundtruth_dir)
    model_ftl_dir         = os.path.join(os.path.join(ftl_dir, model), "ftl")

    if not os.path.exists(model_predictions_dir):
        continue
    
    if not os.path.exists(model_groundtruth_dir):
        continue
    
    if not os.path.exists(model_ftl_dir):
        continue
    
    model_outputmetrics_path = os.path.join(output_metrics_path, model)
    
    model_eval_paths = [os.path.join(model_outputmetrics_path, "pixeleval"), os.path.join(model_outputmetrics_path, "celleval")]
    
    model_pixeleval_resultpath = os.path.join(model_eval_paths[0], "result.csv")
    model_celleval_resultpath  = os.path.join(model_eval_paths[1], "result.csv")
    if not os.path.exists(model_pixeleval_resultpath):
        continue
    if not os.path.exists(model_celleval_resultpath):
        continue
    
    for eval in model_eval_paths:
        
        evaltype = os.path.basename(eval)
        evalgraphs = evaltype[:-4] + "graphs"
        model_resultpath = os.path.join(eval, "result.csv")
        pandas_results = pd.read_csv(model_resultpath)
        num_results = len(pandas_results)
        column_names = list(pandas_results.columns)
        
        for column_name in column_names:
            
            if (column_name == "Image_Name"):
                continue
            if (column_name == "Class"):
                continue
            
            if "weighted" not in column_name:
                continue
            
            column_filename_json = column_name.replace("/", "-")
            
            column_filename = column_filename_json.replace(" ", "_")
            column_filename_json = column_filename_json + "_models.json"
            
            json_dir = os.path.join("/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/currentGraphs", evalgraphs)
            models_ranked_dict = json.load(open(os.path.join(json_dir, column_filename_json)))
            all_models = list(models_ranked_dict.keys())
            num_models = len(all_models)
            avg_std = [float(item) for item in models_ranked_dict[model].split("-")]
            
            model_rank = f"{num_models - all_models.index(model)}/{num_models}"
            
            column_name_dir = os.path.join(eval, column_filename)
            
            if os.path.exists(column_name_dir):
                shutil.rmtree(column_name_dir)
            
            if not os.path.exists(column_name_dir):
                os.mkdir(column_name_dir)
            
            sorted_results = pandas_results.sort_values(by = column_name)
            fig.suptitle(f"\n{model} is ranked {model_rank} in {column_name}\n" +
                         f"(AVG-STD) : {round(avg_std[0],3)}-{round(avg_std[1],4)}\n" + 
                         f"BOTTOM {num_tests} ({evaltype}) Results for {column_name}\n", fontsize = 16)
            
            for worst_test in range(num_tests):
                worst_test_idx = worst_test
                worst = sorted_results.iloc[worst_test_idx]
                worst_image = worst["Image_Name"]
                worst_value = worst[column_name]
                
                ax[worst_test, 0].set_ylabel(worst_image)
                
                ax[worst_test, 0].imshow(np.array(Image.open(os.path.join(model_groundtruth_dir, worst_image))))
                ax[worst_test, 1].imshow(np.array(Image.open(os.path.join(groundtruth_labeldir, worst_image))))
                ax[worst_test, 2].imshow(np.array(Image.open(os.path.join(model_predictions_dir, worst_image))))
                ax[worst_test, 3].imshow(np.array(Image.open(os.path.join(model_ftl_dir, worst_image))))
                
                if worst_test == num_tests-1:
                    ax[num_tests-1, 0].set_xlabel("Groundtruth Binary!", fontsize=12)
                    ax[num_tests-1, 1].set_xlabel("Groundtruth Labelled!", fontsize=12)
                    ax[num_tests-1, 2].set_xlabel("Predictions Binary!", fontsize=12)
                    ax[num_tests-1, 3].set_xlabel("Predictions Labelled!", fontsize=12)

                ax[worst_test, 1].set_title(f"{column_name}: {worst_value} - Ranked ({num_results-worst_test}/{num_results})")
            worst_output_name = f"worst_{num_tests}_{evaltype}_{column_filename}_{model}.jpeg"
            plt.savefig(os.path.join(column_name_dir, worst_output_name))
            plt.cla()

            fig.suptitle(f"\n{model} is ranked {model_rank} in {column_name}\n" +
                         f"(AVG-STD) : {round(avg_std[0],3)}-{round(avg_std[1],4)}\n" + 
                         f"TOP {num_tests} ({evaltype}) Results for {column_name}\n", fontsize = 16)
            
            for best_test in range(num_tests):
                best_test_idx = num_results - best_test - 1
                best = sorted_results.iloc[best_test_idx]
                best_image = best["Image_Name"]
                best_value = best[column_name]
                ax[best_test, 0].set_ylabel(best_image)
                
                ax[best_test, 0].imshow(np.array(Image.open(os.path.join(model_groundtruth_dir, best_image))))
                ax[best_test, 1].imshow(np.array(Image.open(os.path.join(groundtruth_labeldir, best_image))))
                ax[best_test, 2].imshow(np.array(Image.open(os.path.join(model_predictions_dir, best_image))))
                ax[best_test, 3].imshow(np.array(Image.open(os.path.join(model_ftl_dir, best_image))))
                
                if best_test == num_tests-1:
                    ax[num_tests-1, 0].set_xlabel("Groundtruth Binary!", fontsize=12)
                    ax[num_tests-1, 1].set_xlabel("Groundtruth Labelled!", fontsize=12)
                    ax[num_tests-1, 2].set_xlabel("Predictions Binary!", fontsize=12)
                    ax[num_tests-1, 3].set_xlabel("Predictions Labelled!", fontsize=12)
                
                
                ax[best_test, 1].set_title(f"{column_name}: {best_value} - Ranked ({best_test+1}/{num_results})")
            best_output_name = f"best_{num_tests}_{evaltype}_{column_filename}_{model}.jpeg"
            plt.savefig(os.path.join(column_name_dir, best_output_name))
            plt.cla()
            
            middle_idx = np.linspace(num_tests*2, num_results-num_tests*2, num_tests).astype('int')
            fig.suptitle(f"\n{model} is ranked {model_rank} in {column_name}\n" +
                         f"(AVG-STD) : {round(avg_std[0],3)}-{round(avg_std[1],4)}\n" + 
                         f"MIDDLE {num_tests} ({evaltype}) Results for {column_name}\n", fontsize = 16)
            for middle_test in range(num_tests):
                middle = sorted_results.iloc[middle_idx[middle_test]]
                middle_image = middle["Image_Name"]
                middle_value = middle[column_name]
                
                ax[middle_test, 0].set_ylabel(middle_image)
                
                ax[middle_test, 0].imshow(np.array(Image.open(os.path.join(model_groundtruth_dir, middle_image))))
                ax[middle_test, 1].imshow(np.array(Image.open(os.path.join(groundtruth_labeldir, middle_image))))
                ax[middle_test, 2].imshow(np.array(Image.open(os.path.join(model_predictions_dir, middle_image))))
                ax[middle_test, 3].imshow(np.array(Image.open(os.path.join(model_ftl_dir, middle_image))))
                
                
                if middle_test == num_tests-1:
                    ax[num_tests-1, 0].set_xlabel("Groundtruth Binary!", fontsize=12)
                    ax[num_tests-1, 1].set_xlabel("Groundtruth Labelled!", fontsize=12)
                    ax[num_tests-1, 2].set_xlabel("Predictions Binary!", fontsize=12)
                    ax[num_tests-1, 3].set_xlabel("Predictions Labelled!", fontsize=12)
                    
                ax[middle_test, 1].set_title(f"{column_name}: {middle_value} - Ranked ({middle_test+1}/{num_results})")
            middle_output_name = f"middle_{num_tests}_{evaltype}_{column_filename}_{model}.jpeg"
            plt.savefig(os.path.join(column_name_dir, middle_output_name))
            plt.cla()
            
            print(column_name_dir)

