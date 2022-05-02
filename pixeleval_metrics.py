import os, sys, json
from matplotlib import pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi'] = 300

import numpy as np
import itertools

import pandas as pd

input_directory  = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_metrics"
output_directory = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/pixelgraphs/"
evaltype = "pixeleval"

input_directory_list = os.listdir(input_directory)

df = pd.DataFrame()
for smp_model in input_directory_list:
    
    # print(smp_model)
    smp_model_path = os.path.join(os.path.join(input_directory, smp_model), evaltype)
    
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

    
    # print(df)
    
    # print(model_metrics_results)
    
    # if not os.path.exists(total_stats_csvpath):
    #     continue

    # print(total_stats_csvpath)
    # model_df = pd.read_csv(total_stats_csvpath)
    # model_df.rename(columns={"Name":smp_model})
    # model_df["Model"] = smp_model
    # df = df.append(model_df, ignore_index=True)
    

# print(df[['prevalence','TP', 'FP', 'FN', 'TN']], df['TP']+df['FN'], df['TP']+df['TN']+df['FP']+df['FN'])

""" BAR PLOTS """
show_howmany = 50
for (columnName, columnData) in df.iteritems():
    
    columnName_file = columnName.replace(" ", "_")
    columnName_file = columnName.replace("/", "-")

    if columnName == "Image_Name":
        continue
    
    if not df[columnName].dtype == np.float64:
        continue
    
    if len(np.unique(columnData)) <= 1:
        # print(columnName, f" has only 1 value for all models: {np.unique(columnData)}, {list(columnData)}")
        continue

    
    model_data = df[[columnName, "Image_Name"]]
    model_data_avg = model_data[model_data["Image_Name"].str.contains("_mean") == True]
    model_data_avg["Image_Name"] = model_data_avg["Image_Name"].str[:-5]
    model_data_std = model_data[model_data["Image_Name"].str.contains("_std") == True]
    model_data_std["Image_Name"] = model_data_std["Image_Name"].str[:-4]

    print(model_data_avg)

    model_data_avg = model_data_avg.sort_values(by=[columnName], ignore_index=True)
    # print(model_data_avg)
    model_dict = {}
    for index,row in model_data_avg.iterrows():
        model_std = model_data_std[model_data_std['Image_Name'] == row[1]][columnData.name].item()
        model_dict[row[1]] = f"{row[0]}-{model_std}"

    json_model_filename = f"{columnName_file}_models.json"
    with open(os.path.join(output_directory, json_model_filename), "w") as json_model:
        json.dump(model_dict, json_model, indent=4)



    randgraph_idx = np.sort(np.random.randint(show_howmany+1, len(model_data_avg)-show_howmany+1, show_howmany).astype('int'))
    
    fig, ax = plt.subplots(figsize=(20,30))
    fig.tight_layout()
    bars1 = ax.barh(model_data_avg["Image_Name"][0:show_howmany], model_data_avg[columnName][0:show_howmany], label=f"Worst {show_howmany} Performing Models", color='r')
    bars2 = ax.barh(model_data_avg["Image_Name"][randgraph_idx], model_data_avg[columnName][randgraph_idx], label=f"(Random) Middle {show_howmany} Performing Models", color='orange')
    bars3 = ax.barh(model_data_avg["Image_Name"][-show_howmany:],  model_data_avg[columnName][-show_howmany:], label=f"Best {show_howmany} Performing Models", color='g')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[::-1], labels=labels[::-1], loc='upper right', bbox_to_anchor=(.99, .99))
    plt.title(f"{columnName_file}")
    fig.subplots_adjust(left=.35)
    
    savepath = os.path.join(output_directory, f"{columnName_file}_bar.png")
    print(savepath)
    plt.savefig(savepath)

# """ BUILD SCATTER PLOTS """
# scatterplot_parameters = itertools.combinations(list(df.columns), 2)
# for scatterplot_parameter in scatterplot_parameters:
#     if "Model" in scatterplot_parameter:
#         continue
#     if "Class" in scatterplot_parameter:
#         continue
#     print(scatterplot_parameter)
    
#     plt.scatter(df[scatterplot_parameter[0]], df[scatterplot_parameter[1]])
#     plt.xlabel(scatterplot_parameter[0])
#     plt.ylabel(scatterplot_parameter[1])
    
#     if "accuracy/rand index" in scatterplot_parameter[0]:
#         savepath = os.path.join(output_directory, f"accuracy_randidx_scatter_{scatterplot_parameter[1]}.png")
#     elif "F1-Score/dice index" in scatterplot_parameter[0]:
#         savepath = os.path.join(output_directory, f"f1score_diceindex_scatter_{scatterplot_parameter[1]}.png")
#     elif "accuracy/rand index" in scatterplot_parameter[1]:
#         savepath = os.path.join(output_directory, f"accuracy_randidx_scatter_{scatterplot_parameter[0]}.png")
#     elif "F1-Score/dice index" in scatterplot_parameter[1]:
#         savepath = os.path.join(output_directory, f"f1score_diceindex_scatter_{scatterplot_parameter[0]}.png")
#     else:
#         savepath = os.path.join(output_directory, f"{scatterplot_parameter[0]}_{scatterplot_parameter[1]}_scatter.png")
#     plt.savefig(savepath)
#     plt.clf()
#     plt.cla()
    
""" FP, TP, FN, TN Histogram PLOT """

# plt.cla()
# plt.clf()
# interesting_column_count = 0
# for (columnName, columnData) in df.iteritems():
    
#     # print(columnName)
#     if "Model" == columnName:
#         continue
#     if "Class" == columnName:
#         continue
    
#     unique_values = np.unique(columnData)
#     if len(unique_values) <= 1:
#         print("NOT UNIQUE: ", columnName)
#         continue
    
#     interesting_column_count += 1
    
# print(interesting_column_count)
    # n_bins = 50
    # max_metric = max(df[columnName])
    # min_metric = min(df[columnName])
    # datarange = max_metric - min_metric
    # width = datarange/n_bins
    
    # title_name = columnName.rstrip().replace("/", "_")
    # fig, ax = plt.subplots(1, 1, figsize=(10,10))
    # ax.hist(df[columnName], bins=n_bins)
    # plt.ylabel("Number of Models with Value")
    # plt.xlabel(f"{title_name} values")
    
    # savepath = os.path.join(output_directory, title_name + "_histogram.png")
    # plt.savefig(savepath)
    
