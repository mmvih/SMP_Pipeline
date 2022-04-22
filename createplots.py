import os, sys, json
from matplotlib import pyplot as plt

import numpy as np

dictionary = {
"accuracy"  : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/accuracy_models.json",
"fscore"    : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/fscore_models.json",
"iou_score" : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/iou_score_models.json",
"MCCLoss"   : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/MCCLoss_models.json",
"precision" : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/precision_models.json",
"recall"    : "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_ANALYSIS/recall_models.json"
}


show_howmany = 50

for key in dictionary.keys():
    accuracy = json.load(open(dictionary[key], "r"))
    accuracy_names = list(accuracy.keys())
    accuracy_values = list(accuracy.values())
    split_avg = []
    split_std = []
    for val in accuracy_values:
        val_tuple = tuple(val.split("-"))
        split_avg.append(float(val_tuple[0]))
        split_std.append(float(val_tuple[1]))
    print(split_avg)

    rand_names = []
    rand_avgs  = []
    randgraph_idx = sorted(np.random.randint(51, len(accuracy_names)-51, 50).astype('int'))
    for idx in randgraph_idx:
        print(accuracy_names[idx], accuracy_values[idx])
        rand_names.append(accuracy_names[idx])
        rand_avgs.append(split_avg[idx])
    
    fig, ax = plt.subplots(figsize=(20,30))
    fig.tight_layout()
    bars1 = ax.barh(accuracy_names[0:show_howmany], split_avg[0:show_howmany],label="Worst Performing Models")
    # ax.barh(["..."], [0])
    bars3 = ax.barh(rand_names, rand_avgs, label="(Random) Middle Performing Models")
    bars2 = ax.barh(accuracy_names[-show_howmany:], split_avg[-show_howmany:],label="Best Performing Models")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[::-1], labels=labels[::-1], loc='upper right', bbox_to_anchor=(.99, .99))
    fig.subplots_adjust(left=.35)

    plt.xlim([min(split_avg) - 0.05, min(max(split_avg) + .1, 1.0)])
    plt.title(f"{key}")
    plt.savefig(f"{key}_modelgraphs.png")