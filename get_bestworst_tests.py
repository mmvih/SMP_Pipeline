import os, sys
import pandas as pd

import matplotlib.pyplot as plt

output_metrics_path = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/dummy_output_metrics"

testingoutputs_dir = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ftl"
predictions_dir = "predictions"
groundtruth_dir = "groundtruths"

output_metrics_modellist = os.listdir(output_metrics_path)

num_tests = 5
fig, ax = plt.subplots(3, num_tests, figsize = (num_tests*4, 3*4))

for model in output_metrics_modellist:
    
    model_outputs_path = os.path.join(testingoutputs_dir, model)
    model_predictions_dir = os.path.join(model_outputs_path, predictions_dir)
    model_groundtruth_dir = os.path.join(model_outputs_path, groundtruth_dir)

    if not os.path.exists(model_predictions_dir):
        continue
    
    if not os.path.exists(model_groundtruth_dir):
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
        
        model_resultpath = os.path.join(eval, "result.csv")
        pandas_results = pd.read_csv(model_resultpath)
        column_names = list(pandas_results.columns)
        
        for column_name in column_names:
            if (column_name == "Image_Name"):
                continue
            
            sorted_results = pandas_results.sort_values(by = column_names)
            print(sorted_results)
        
        exit()
    