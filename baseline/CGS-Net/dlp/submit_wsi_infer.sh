#!/usr/bin/env bash
#SBATCH --job-name=wsi_inf_more07 # Job name
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jeremy.juybari@maine.edu     # Where to send mail
#SBATCH --ntasks=1                   # Run on a single node
#SBATCH --cpus-per-task=28        # Number of cores
#SBATCH --mem=600gb                     # Job memory request
#SBATCH --time=20:05:00               # Time limit hrs:min:sec
#SBATCH --output=wsi_inf_more07.log # Standard output and error log
#SBATCH --partition=epyc-hm        # dgx or gpu
pwd; hostname; date
#module load apptainer

source /home/jjuybari/DLP/dlp/.datav/bin/activate
which python
python  /home/jjuybari/DLP/dlp/wsi_inference.py


# note when doing a shell, $ apptainer shell ex.simg you need to be on a node, if you are on login1 then the command will error out