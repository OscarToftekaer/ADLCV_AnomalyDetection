# !/bin/sh
### General options
### �- specify queue --
# BSUB -q gpua100 #SELECT YOUR GPU QUEUE
### -- set the job Name --
# BSUB -J DDPM_train
### -- ask for number of cores (default: 1) --
# BSUB -n 4 #Number of cores (4/gpu)
# BSUB -R "span[hosts=1]" #Always set this to 1
## -- Select the resources: 1 gpu in exclusive process mode --
# BSUB -gpu "num=1:mode=exclusive_process" #Number of GPUs requested
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
# BSUB -W 23:59 #Walltime needed for computation (Be sure to set a lower time, if you want to have your job run faster)
# request 5GB of system-memory
# BSUB -R "rusage[mem=8GB]" #RAM Required for computation (per CPU, careful about modifying this)
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
# BSUB -o gpu-%J.out
# BSUB -e gpu_%J.err
# -- end of LSF options --

# Load the relevant CUDA module (Only necessary on a100s)
module load cuda/12.1.1


# Make sure to change directory to the directory of your project
module load python3/3.11.3
cd /zhome/31/c/147318/Advaned_DLCV/exam_project/ADLCV_AnomalyDetection/

# Make sure to load your environment
source /zhome/31/c/147318/irishcream/bin/activate

JID=${LSB_JOBID}
python train.py 