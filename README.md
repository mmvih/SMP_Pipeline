# SMP_Pipeline
Uses the Segmentation Model Toolkit to train Multiple Models

## Get the Data in Working Directory
Tissuenet's NPZ directory can be downloaded from https://datasets.deepcell.org/data

```#!/bin/sh
unzip tissuenet_v1.0.zip.download
mkdir Data
python3 tissuenet_parsing.py --npzDir ./tissuenet_1.0 --outputDir ./Data
```

## Set up the Environment and Clone polus-plugins

```#!/bin/sh
git clone https://github.com/PolusAI/polus-plugins.git
```

## Create CSV File for all the Models

```#!/bin/sh
python3 write_csv.py --csvFile models.csv
```

## Train all the Nuclear Models

```#!/bin/sh
mkdir Models
python3 input_script.py \
--mainFile "./polus-plugins/segmentation/polus-smp-training-plugin/src/main.py \
--outputModels ./Models \
--csvFile models.csv \
--imagesTrainDir ./Data/nuclear/train/image \
--labelsTrainDir ./Data/nuclear/train/groundtruth_centerbinary_2pixelsmaller \
--imagesValidDir ./Data/nuclear/validation/image \
--labelsValidDir ./Data/nuclear/validation/groundtruth_centerbinary_2pixelsmaller \
```

## Create Images of the Loss Plots

```#!/bin/sh
python3 input_script.py \
--inputModels ./Models
--outputLosses ./Models
```

## Use Trained Models to save Predictions
```#!/bin/sh
mkdir ModelsOutput
python3 input_script.py \
--inputModels ./Models \
--outputPredictions ./ModelsOutput \
--imagesTestDir ./Data/nuclear/test/image \
--labelsTestDir ./Data/nuclear/test/groundtruth_centerbinary_2pixelsmaller \
```

## Generate Labels from Binary Predictions
```#!/bin/sh
python3 ftl.py \
--inputPredictions ./ModelsOutput \
--outputLabels ./ModelsOutput \
--inputGroundtruth ./Data/nuclear/test/groundtruth
```

## Run Cellular Evaluation 
```#!/bin/sh
python3 metricEvaluation.py \
--inputPredictions ./ModelsOutputs \
--outputMetrics ./ModelsOutput \
--inputGroundtruth ./Data/nuclear/test/groundtruth \
--evaluationMetric "CellularEvaluation" \
```

## Run Pixel Evaluation 
```#!/bin/sh
python3 metricEvaluation.py \
--inputPredictions ./ModelsOutputs \
--outputMetrics ./ModelsOutput \
--inputGroundtruth ./Data/nuclear/test/groundtruth_centerbinary_2pixelsmaller \
--evaluationMetric "PixelEvaluation" \
```

## Run SMP Evaluation (metrics in the SMP training module and time)
```#!/bin/sh
python3 smpEvaluation.py \
--inputModels ./ModelsOutputs \
--outputMetrics ./ModelsOutput \
--imagesTestDir ./Data/nuclear/test/image \
--labelsTestDir ./Data/nuclear/test/groundtruth_centerbinary_2pixelsmaller \
```

## Create BoxPlots and CSV files for Graph Pyramid Plugin
```#!/bin/sh
python boxPlots.py \
--inputMetrics ./ModelsOutputs \
--outputBoxplots ./ModelsOutputs

```


