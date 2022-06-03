# %%
import os, sys, json
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import matplotlib
matplotlib.rcParams['figure.dpi'] = 300

import numpy as np
import itertools

import pandas as pd

# %%
eval = "pixel"
input_directory  = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/TOP10_output_metrics"
output_directory = f"/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/TOP10_boxplots/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

input_directory_list = os.listdir(input_directory)

# %%
df = pd.DataFrame()
for smp_model in input_directory_list:
    
    # print(smp_model)
    smp_model_path = os.path.join(os.path.join(input_directory, smp_model), f"{eval}eval")
    
    # total_stats_csvpath = os.path.join(smp_model_path, "total_stats_result.csv")
    results_csvpath = os.path.join(smp_model_path, "result.csv")
    
    if not os.path.exists(results_csvpath):
        continue
    
    df_results = pd.read_csv(results_csvpath)
    df_results.loc[f'{smp_model}_mean'] = df_results[0:1249].mean()
    df_results.loc[f'{smp_model}_std']  = df_results[0:1249].std()
    df_results['Image_Name'] = df_results.index

    df_results = df_results.loc[[f'{smp_model}_mean', f'{smp_model}_std']]
    # print( df_results.loc[[f'{smp_model}_mean', f'{smp_model}_std']])
    df = df.append(df_results, ignore_index=True)


# %%
metrics = {}
for smp_model in input_directory_list:
    
    # print(smp_model)
    smp_model_path = os.path.join(os.path.join(input_directory, smp_model), f"{eval}eval")
    
    # total_stats_csvpath = os.path.join(smp_model_path, "total_stats_result.csv")
    results_csvpath = os.path.join(smp_model_path, "result.csv")
    
    if not os.path.exists(results_csvpath):
        continue
    
    df_results = pd.read_csv(results_csvpath)
    df_results_columns = df_results.columns
    for column in df_results_columns:
        if column == "Image_Name":
            continue
        if column == "Class":
            continue
        if column not in metrics:
            metrics[column] = {smp_model : list(df_results[column])}
        else:
            metrics[column][smp_model] = list(df_results[column])


# %%
# plt.rcParams["figure.figsize"] = (20,30)
# show_howmany = 5
# colors = ['pink']*show_howmany + ['lightblue']*show_howmany + ['lightgreen']*show_howmany
# pink_patch = mpatches.Patch(color='pink', label="Lowest Scoring Models")
# lightblue_patch = mpatches.Patch(color='lightblue', label='(Random) Middle Scoring Models')
# lightgreen_patch = mpatches.Patch(color='lightgreen', label='Highest Scoring Models')

# for columnName in metrics.keys():
    
#     column_filename = columnName.replace(" ", "_").replace("/","-")
#     model_data = df[[columnName, "Image_Name"]]
#     model_data_avg = model_data[model_data["Image_Name"].str.contains("_mean") == True]
#     model_data_std = model_data[model_data["Image_Name"].str.contains("_std") == True]
    
#     model_data_avg["Image_Name"] = model_data_avg["Image_Name"].str[:-5]
#     model_data_std["Image_Name"] = model_data_std["Image_Name"].str[:-4]


#     model_data_merged = pd.merge(model_data_avg, model_data_std, on="Image_Name")
#     model_data_merged = model_data_merged.sort_values(by=[columnName+"_x"], ignore_index=True)
    
#     randgraph_idx = np.sort(np.random.randint(show_howmany+1, len(model_data_avg)-show_howmany+1, show_howmany).astype('int'))
    
#     lowest_models  = list(model_data_merged["Image_Name"].iloc[0:show_howmany])
#     highest_models = list(model_data_merged["Image_Name"].iloc[-show_howmany:])
#     middle_models  = list(model_data_merged["Image_Name"].iloc[randgraph_idx])
    
#     lowest_models_values  = [metrics[columnName][lowest_model] for lowest_model in lowest_models]
#     highest_models_values = [metrics[columnName][lowest_model] for lowest_model in highest_models]
#     middle_models_values  = [metrics[columnName][lowest_model] for lowest_model in middle_models]
    
#     box = plt.boxplot(lowest_models_values+middle_models_values+highest_models_values, vert=False, patch_artist=True)
#     for patch, color in zip(box['boxes'], colors):
#         patch.set_facecolor(color)
#     plt.yticks(ticks=list(range(1,show_howmany*3 + 1)), labels=lowest_models+middle_models+highest_models)
#     plt.title(f"{columnName} - {eval} Evaluation")
#     plt.legend(handles = [lightgreen_patch, lightblue_patch, pink_patch])
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_directory, column_filename+f"_{eval}"+".jpeg"))
#     plt.cla()
#     plt.clf()

#%%
for columnName in metrics.keys():
    
    column_filename = columnName.replace(" ", "_").replace("/","-")
    model_data = df[[columnName, "Image_Name"]]
    model_data_avg = model_data[model_data["Image_Name"].str.contains("_mean") == True]
    model_data_std = model_data[model_data["Image_Name"].str.contains("_std") == True]
    
    model_data_merged = pd.merge(model_data_avg, model_data_std, on="Image_Name")
    model_data_merged = model_data_merged.sort_values(by=[columnName+"_x"], ignore_index=True)
    print(model_data_merged)