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
wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
bash Anaconda3-2018.12-Linux-x86_64.sh -b -p
source ~/anaconda/bin/activate
```

```#!/bin/sh
python -m pip install -U pip
pip install -r requirements.txt
pip install git+https://github.com/qubvel/segmentation_models.pytorch@c48c906bc2ee238f45aedf413e9248c37f088894
```

```#!/bin/sh
git clone https://github.com/PolusAI/polus-plugins.git
```
The following plugins are used; make sure these plugins are available to use:
1. polus-plugins/segmentation/**polus-smp-training-plugin**
2. polus-plugins/features/**polus-pixelwise-evaluation-plugin**
3. polus-plugins/features/**polus-cellular-evaluation-plugin**

Can clone branch with all the plugins merged
```#!/bin/sh
git clone --branch smp_Pipeline https://github.com/mmvih/polus-plugins.git
```


## Create CSV File for all the Models

```#!/bin/sh
python3 writeCsv.py --csvFile models.csv
```

## Train all the Nuclear Models

```#!/bin/sh
mkdir Models
python3 inputScript.py \
--outputModels ./Models \
--csvFile models.csv \
--imagesTrainDir ./Data/nuclear/train/image \
--labelsTrainDir ./Data/nuclear/train/groundtruth_centerbinary_2pixelsmaller \
--imagesValidDir ./Data/nuclear/validation/image \
--labelsValidDir ./Data/nuclear/validation/groundtruth_centerbinary_2pixelsmaller \
```

## Create Images of the Loss Plots

```#!/bin/sh
python3 lossPlots.py \
--inputModels ./Models
--outputLosses ./Models
```

## Use Trained Models to Save Predictions
```#!/bin/sh
mkdir ModelsOutput
python3 savePredictions.py \
--inputModels ./Models \
--outputPredictions ./ModelsOutput \
--imagesTestDir ./Data/nuclear/test/image \
--labelsTestDir ./Data/nuclear/test/groundtruth_centerbinary_2pixelsmaller \
```

## Generate Labels from Binary Predictions

We need to use python3.9 to use the polus-ftl-label-plugin
```#!/bin/sh
conda create --name py39 -y python=3.9
conda activate py39

pip install --upgrade cython==3.0.0a9
curl https://sh.rustup.rs -sSf | bash -s -- -y
PATH="~/.cargo/bin:${PATH}"

pip install -r polus-plugins/transforms/images/polus-ftl-label-plugin/rust_requirements.txt
pip install -r polus-plugins/transforms/images/polus-ftl-label-plugin/src/requirements.txt
mkdir polus-plugins/transforms/images/polus-ftl-label-plugin/ftl_rust/src
cd polus-plugins/transforms/images/polus-ftl-label-plugin/
python rust_setup.py install
cd src
mkdir src/
python setup.py build_ext --inplace
mv src/ftl.cpython-39-x86_64-linux-gnu.so .
rm -rf src/
cd ../../../../../
```

```#!/bin/sh
python3 ftl.py \
--inputPredictions ./ModelsOutput \
--outputLabels ./ModelsOutput \
--inputGroundtruth ./Data/nuclear/test/groundtruth
```

## Run Pixel Evaluation 
```#!/bin/sh
mkdir ModelsPixelOutput
python3 metricEvaluation.py \
--inputPredictions ./ModelsOutputs \
--outputMetrics ./ModelsPixelOutput \
--inputGroundtruth ./Data/nuclear/test/groundtruth_centerbinary_2pixelsmaller \
--evaluationMetric "PixelEvaluation" \
```

## Run Cellular Evaluation 
```#!/bin/sh
mkdir ModelsCellOutput
python3 metricEvaluation.py \
--inputPredictions ./ModelsOutputs \
--outputMetrics ./ModelsCellOutput \
--inputGroundtruth ./Data/nuclear/test/groundtruth \
--evaluationMetric "CellularEvaluation" \
```

## Run SMP Evaluation (metrics in the SMP training module and time)
```#!/bin/sh
python3 smpEvaluation.py \
--inputModels ./ModelsOutputs \
--outputMetrics ./ModelsOutput \
--imagesTestDir ./Data/nuclear/test/image \
--labelsTestDir ./Data/nuclear/test/groundtruth_centerbinary_2pixelsmaller \
```

## Create Summary for the Pixel Evaluation Metrics
```#!/bin/sh
python metricSummary.py \
--inputMetrics ./ModelsPixelOutputs \
--evaluationMetric PixelEvaluation
--outputCSVs ./ModelsPixelOutput
```

## Create Summary for the Cell Evaluation Metrics
```#!/bin/sh
python metricSummary.py \
--inputMetrics ./ModelsCellOutputs \
--evaluationMetric CellEvaluation
--outputCSVs ./ModelsCellOutput
```

## Create BoxPlots for Cell Evaluations
```#!/bin/sh
python boxPlots.py \
--inputMetrics ./ModelsPixelOutputs \
--inputCSVs ./ModelsPixelOutput
--evaluationMetric PixelEvaluation
--outputBoxplots ./ModelsPixelOutputs
```

## Create BoxPlots for Cell Evaluations
```#!/bin/sh
python boxPlots.py \
--inputMetrics ./ModelsCellOutputs \
--inputCSVs ./ModelsCellOutput
--evaluationMetric CellEvaluation
--outputBoxplots ./ModelsCellOutputs
```

