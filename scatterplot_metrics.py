# %%
import os, sys, json
from matplotlib import pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi'] = 300

import numpy as np
import itertools

# %%
model_dirpath = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS"
models = os.listdir(model_dirpath)

dictionary = {
"accuracy"  : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/accuracy_models.json",
"fscore"    : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/fscore_models.json",
"iou_score" : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/iou_score_models.json",
"MCCLoss"   : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/MCCLoss_models.json",
"precision" : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/precision_models.json",
"recall"    : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/recall_models.json",
"time"      : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/time_models.json"
}

# %%
# import operator
# time_dictionary = {}

# analysis_notdone = []
# for model in models:
#     model_path = os.path.join(model_dirpath, model)
#     # print(model_path)
#     json_path =  os.path.join(model_path, "metrics_accuracy.json")
#     if not os.path.exists(json_path):
#         analysis_notdone.append(model)
#         continue
#     json_metric = json.load(open(json_path, "r"))
#     time_dictionary[model] = json_metric["time_allmetrics"]
# time_dictionary = {k: str(v)+"-0" for k, v in sorted(time_dictionary.items(), 
#                                             key=lambda item: item[1])}
# time_jsonpath = os.path.join(model_dirpath, "time_models.json")
# with open(time_jsonpath, 'w') as metric_json:
#     json.dump(time_dictionary, metric_json, indent=4)


# %%
show_howmany = 50

# for key in dictionary.keys():
#     if key == "accuracy":
#         print(key)
#         accuracy = json.load(open(dictionary[key], "r"))
#         accuracy_names = list(accuracy.keys())
#         accuracy_values = list(accuracy.values())
#         split_avg = []
#         split_std = []
#         for val in accuracy_values:
#             val_tuple = tuple(val.split("-"))
#             split_avg.append(float(val_tuple[0]))
#             split_std.append(float(val_tuple[1]))
#         # print(split_avg)

#         rand_names = []
#         rand_avgs  = []
#         randgraph_idx = sorted(np.random.randint(51, len(accuracy_names)-51, 50).astype('int'))
#         for idx in randgraph_idx:
#             # print(accuracy_names[idx], accuracy_values[idx])
#             rand_names.append(accuracy_names[idx])
#             rand_avgs.append(split_avg[idx])
        
#         fig, ax = plt.subplots(figsize=(20,30))
#         fig.tight_layout()
#         if (key != "MCCLoss") and (key != "time"):
#             print("mcc loss and time should not be here")
#             bars1 = ax.barh(accuracy_names[0:show_howmany], split_avg[0:show_howmany],label="Worst Performing Models", color='r')
#             bars3 = ax.barh(rand_names, rand_avgs, label="(Random) Middle Performing Models", color='orange')
#             bars2 = ax.barh(accuracy_names[-show_howmany:], split_avg[-show_howmany:],label="Best Performing Models", color='g')
#         else:
#             print("mcc loss and time should be here")
#             bars1 = ax.barh(accuracy_names[0:show_howmany], split_avg[0:show_howmany],label="Best Performing Models", color='g')
#             # ax.barh(["..."], [0])
#             bars3 = ax.barh(rand_names, rand_avgs, label="(Random) Middle Performing Models", color='orange')
#             bars2 = ax.barh(accuracy_names[-show_howmany:], split_avg[-show_howmany:],label="Worst Performing Models", color='r')
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles=handles[::-1], labels=labels[::-1], loc='upper right', bbox_to_anchor=(.99, .99))
#         fig.subplots_adjust(left=.35)

#         if key != "time":
#             plt.xlim([min(split_avg) - 0.05, min(max(split_avg) + .1, 1.0)])
#         else:
#             plt.xlim([min(split_avg) - 5, max(split_avg) + 10])
#         plt.title(f"{key}")
#         output_path = os.path.join("graphs", f"{key}_modelgraphs.png")
#         plt.savefig(output_path)

# %%
dict_keys = list(dictionary.keys())
scatterplot_parameters = itertools.combinations(dict_keys,2)

# %%
MCCLoss_dictionaryvalue = dictionary["MCCLoss"]
MCCLoss = json.load(open(MCCLoss_dictionaryvalue, 'r'))
# names = list(MCCLoss.keys())
# # print(names)
# bottomnames = names[-show_howmany:]

# topnames = names[0:show_howmany]
# print(topnames)
# # MCCLoss_avgs = [float(MCCLoss[name].split("-")[0]) for name in names]

# for scatterplot_parameter in scatterplot_parameters:
#     print(scatterplot_parameter)
#     # if 'accuracy' not in scatterplot_parameter:
#     #     continue
#     # if 'MCCLoss' not in scatterplot_parameter:
#     #     continue
#     print(scatterplot_parameter)
    
#     x_dictionaryvalue = dictionary[scatterplot_parameter[0]]
#     y_dictionaryvalue = dictionary[scatterplot_parameter[1]]
#     x = json.load(open(x_dictionaryvalue, 'r'))
#     y = json.load(open(y_dictionaryvalue, 'r'))
    
#     # names = list(x.keys())
#     # x_avgs = []
#     # y_avgs = []
#     # for name in names:
#     #     x_avgs.append()
#     #     y_avgs.append()
#     x_avgs = [float(x[name].split("-")[0]) for name in names if name not in topnames]
#     y_avgs = [float(y[name].split("-")[0]) for name in names if name not in topnames]
    
#     top_x_avgs = [float(x[name].split("-")[0]) for name in topnames]
#     top_y_avgs = [float(y[name].split("-")[0]) for name in topnames]
   
#     # plt.scatter(x_avgs, y_avgs)
#     plt.cla()
#     plt.clf()
#     plt.figure(figsize=(6,6))
#     plt.scatter(top_x_avgs, top_y_avgs, color='g')
#     plt.xlabel(scatterplot_parameter[0])
#     plt.ylabel(scatterplot_parameter[1])
#     top50_output_path = os.path.join("graphs", f"SCATTER_top50_{scatterplot_parameter[0]}_{scatterplot_parameter[1]}")
#     plt.savefig(top50_output_path)

#     plt.scatter(x_avgs, y_avgs, color='b')
#     plt.scatter(top_x_avgs, top_y_avgs, color='g')
#     output_path = os.path.join("graphs", f"SCATTER_{scatterplot_parameter[0]}_{scatterplot_parameter[1]}")
#     plt.savefig(output_path)
    
#     plt.savefig(output_path)
#     print(f"SAVED {output_path}")
#     plt.clf()
#     plt.cla()


# %%
# print("BUILD HISTOGRAMS")
# for key in dictionary.keys():
    
#     print(key)
#     dict = json.load(open(dictionary[key], 'r'))
#     # print(dict)
#     plt.cla()
#     plt.clf()
    
#     metric_names = list(dict.keys())
#     metric_values = list(dict.values())
#     metric_values = [1-float(val.split("-")[0])for val in metric_values]
#     min_metric_value = min(metric_values)
#     max_metric_value = max(metric_values)
#     print(metric_values)
#     width_metric_value = abs(max_metric_value-min_metric_value)/100
    
#     bins_range = list(np.arange(min_metric_value, max_metric_value, width_metric_value))
#     print(bins_range)
#     plt.hist(metric_values, bins=bins_range)
#     plt.xlim(min_metric_value-.1, max_metric_value+.1)
#     plt.title(f"{key} Histogram - 100 Bins Spread across {min_metric_value:.4}-{max_metric_value:.4}")
#     output_hist = os.path.join("graphs", f"{key}_histogram.png")
#     plt.savefig(output_hist)

# %%
