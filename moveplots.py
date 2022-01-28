import os, sys
import shutil

smp_inputs_path =  "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP"
smp_outputs_path = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/output_SMP_plots"

smp_inputs_list = os.listdir(smp_inputs_path)

for smp_input_file in smp_inputs_list:
    smp_input_path = os.path.join(smp_inputs_path, smp_input_file)
    plot_path = os.path.join(smp_input_path, "Logs.jpg")
    print(plot_path)
    
    if os.path.exists(plot_path):
        shutil.copy(plot_path, os.path.join(smp_outputs_path, smp_input_file + ".jpg"))