import os, sys, json
from matplotlib import pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi'] = 300

import numpy as np
import itertools

import pandas as pd

import seaborn as sns

input_directory  = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_eval"
output_directory = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/pixelgraphs/"

input_directory_list = os.listdir(input_directory)

df = pd.DataFrame()
for smp_model in input_directory_list:
    
    smp_model_path = os.path.join(os.path.join(input_directory, smp_model), "pixeleval")
    
    total_stats_csvpath = os.path.join(smp_model_path, "total_stats_result.csv")
    
    if not os.path.exists(total_stats_csvpath):
        continue

    model_df = pd.read_csv(total_stats_csvpath)
    model_df.rename(columns={"Name":smp_model})
    model_df["Model"] = smp_model
    df = df.append(model_df, ignore_index=True)

# df.to_csv("pixeleval.csv")
# del df['Class']
# del df['prevalence']
# del df["TP"]
# del df["TN"]
# del df["FP"]
# del df["FN"]

print(df['accuracy', 'accuracy'])
sns_plot = sns.pairplot(df)
fig = sns_plot.fig
fig.savefig("image.png")

