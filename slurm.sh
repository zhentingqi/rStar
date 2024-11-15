#!/bin/bash
#SBATCH --job-name=paper_dis
#SBATCH -c 2
#SBATCH -N 1
#SBATCH -t 1-00:00              # Runtime in D-HH:MM, minimum of 10 minutes
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
# bash jobs/ours/zhenting/exp_MATH/run_ours_9.sh
# bash jobs/ours/zhenting/exp_FOLIO/run_ours_9.sh
# bash jobs/ours/zhenting/exp_BGQA/run_ours_4.sh
# bash jobs/baselines/zhenting/fewshot_cot/run_fewshot_5.sh
# bash jobs/ours/zhenting/exp_model_limit/run_ours_2.sh
bash /n/home01/minma/rStar/jobs/ours/mingyuan/llama_31_8b_instruct_amc.sh
# bash jobs/ours/zhenting/exp_BGQA/run_ours_4.sh
# --------------------------

python -c 'print("Hi. Everything is done!")'
