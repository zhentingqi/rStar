#!/bin/bash
#SBATCH --job-name=paper_dis
#SBATCH -c 2
#SBATCH -N 1
#SBATCH -t 0-12:00              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu                  # Partition to submit to
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1     
#SBATCH --mem=128GB

#SBATCH -o /n/home01/minma/slurm_out/paper_dis_%j.out      # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home01/minma/slurm_out/paper_dis_%j.err      # File to which STDERR will be written, %j inserts jobid

#SBATCH --mail-type=FAIL,END      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mingyuan_ma@g.harvard.edu

# --- load env here ---
module load python/3.10.12-fasrc01
source activate
source activate rstar
# ---------------------

python -c 'print("Hi. Your job is running!")'

# --- run your code here ---
# bash /n/home01/minma/rStar/jobs/ours/mingyuan/llama_31_8b_amc.sh
bash /n/home01/minma/rStar/jobs/ours/mingyuan/llama_31_8b_instruct_amc.sh
# --------------------------

python -c 'print("Hi. Everything is done!")'
