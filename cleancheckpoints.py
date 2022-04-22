import os, sys
import shutil

def main():

	outputmodel_dirpath = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP/"
	outputmodels = os.listdir(outputmodel_dirpath)
	errorcount = 1
	for outputmodel in outputmodels:
		outputmodelpath = os.path.join(outputmodel_dirpath,outputmodel)
		if os.path.exists(os.path.join(outputmodelpath, "ERROR")):
			print(f"Error Count {errorcount}")
			errorcount += 1
			continue
		outputmodel_checkpointpath = os.path.join(outputmodelpath, "checkpoints")
		if os.path.exists(os.path.join(outputmodelpath, "model.pth")):
			if os.path.exists(outputmodel_checkpointpath):
				modelfinal_path = os.path.join(outputmodel_checkpointpath, "model_final.pth")
				checkpointfinal_path = os.path.join(outputmodel_checkpointpath, "checkpoint_final.pth")
				if os.path.exists(modelfinal_path) and os.path.exists(outputmodel_checkpointpath):
					print(modelfinal_path, checkpointfinal_path)
					print(outputmodelpath)
					shutil.move(modelfinal_path, outputmodelpath)
					shutil.move(checkpointfinal_path, outputmodelpath)
				if (os.path.exists(os.path.join(outputmodelpath, "model_final.pth")) and os.path.exists(os.path.join(outputmodelpath, "checkpoint_final.pth"))):
					print("removing: ", outputmodel_checkpointpath)
					shutil.rmtree(outputmodel_checkpointpath)
				else:
					print("no model or checkpoint final?: ", outputmodel_checkpointpath)

main()
