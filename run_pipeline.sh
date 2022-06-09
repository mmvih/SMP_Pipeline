csvInput="testing.csv"
outputModelsDir="cytoplasm_models"
outputModelsDataDir="cytoplasm_models_outputs"

tissuenet_basedir="/home/vihanimm/SegmentationModelToolkit/Data/ometif_data"
# how the directories are organized if using preprocess_tissuenet.py
tissuenet_channeldir="${tissuenet_basedir}/cell" # nuclear or cell

tissuenet_traindir="${tissuenet_channeldir}/train"
tissuenet_validdir="${tissuenet_channeldir}/validation"
tissuenet_testdir="${tissuenet_channeldir}/test"

# images and labels (binary&multi-instance) for training binary models
# training data
tissuenet_imagesTrainDir="{tissuenet_traindir}/image"
tissuenet_binarylabelsTrainDir="{tissuenet_traindir}/groundtruth_centerbinary_2pixelsmaller"
tissuenet_multilabelsTrainDir="{tissuenet_traindir}/groundtruth" 
# validation data
tissuenet_imagesValidDir="{tissuenet_validdir}/image"
tissuenet_binarylabelsValidDir="{tissuenet_validdir}/groundtruth_centerbinary_2pixelsmaller"
tissuenet_multilabelValidDir="{tissuenet_validdir}/groundtruth"
# testing data
tissuenet_imagesTestDir="{tissuenet_testdir}/image"
tissuenet_binarylabelsTestDir="{tissuenet_testdir}/groundtruth_centerbinary_2pixelsmaller"
tissuenet_multilabelTestDir="{tissuenet_testdir}/groundtruth"

""" STARTING THE PYTHON SCRIPTS """

# Create CSV File to get the list of different models that need to be trained
python write_csv.py --csvFile $csvInput

# start training all the models
python input_script.py \
--MainFile "./polus-plugins/segmentation/polus-smp-training-plugin/src/main.py" \
--outputModel $outputModelsDir \
--csvFile $csvInput \
--imagesTrainDir $tissuenet_imagesTrainDir \
--labelsTrainDir $tissuenet_binarylabelsTrainDir \
--imagesValidDir $tissuenet_imagesValidDir \
--labelsValidDir $tissuenet_binarylabelsValidDir

# save predictions for every model
python savepredictions.py \
--inputModels $outputModelsDir
--outputPredictions $outputModelsDataDir \
--imagesTestDir $tissuenet_imagesTestDir \
--labelsTestDir $tissuenet_labelsTestDir

# generate labels from binary predictions
python ftl.py \
--inputPredictions $outputModelsDataDir \
--outputLabels $outputModelsDataDir \
--inputGroundtruth $tissuenet_multilabelTestDir

# use cellular evaluation
python metricEvaluation.py \
--inputPrediction $outputModelsDataDir
--inputGroundtruth $tissuenet_multilabelTestDir
--outputMetrics $outputModelsDataDir
--evaluationMetric "CellEvaluation"

# use pixel evaluation
python metricEvaluation.py \
--inputPrediction $outputModelsDataDir
--inputGroundtruth $tissuenet_binarylabelTestDir
--outputMetrics $outputModelsDataDir
--evaluationMetric "PixelEvaluation"
