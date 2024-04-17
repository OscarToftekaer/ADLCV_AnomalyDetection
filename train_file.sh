# !/bin/sh
#BSUB -q gpua100
#BSUB -J DDIM_train

#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process" #Number of GPUs requested
#BSUB -R "select[model==XeonGold6226]" 
#BSUB -R "rusage[mem=20GB]"

#BSUB -W 23:59 
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# Load the relevant CUDA module (Only necessary on a100s)
# module load cuda/12.1.1


# Make sure to change directory to the directory of your project
module load python3/3.11.3
cd /zhome/31/c/147318/Advaned_DLCV/exam_project/ADLCV_AnomalyDetection/

# Make sure to load your environment
source /zhome/31/c/147318/irishcream/bin/activate

JID=${LSB_JOBID}
python train.py 