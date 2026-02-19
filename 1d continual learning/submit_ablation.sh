#!/bin/bash
# ============================================================
# Submit one SLURM job per (max_outputs, num_batches) combination
# Usage:  bash submit_ablation.sh
# ============================================================

# ---------- ablation grid (edit these) ----------
MAX_OUTPUTS_LIST=(10 50 100 500 1000)
NUM_BATCHES_LIST=(2 5 10 20 50 100)
# ------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for max_outputs in "${MAX_OUTPUTS_LIST[@]}"; do
  for num_batches in "${NUM_BATCHES_LIST[@]}"; do

    job_name="sens_mo${max_outputs}_nb${num_batches}"

    echo "Submitting job: max_outputs=${max_outputs}  num_batches=${num_batches}  (${job_name})"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --time=0-72:00:00
#SBATCH --partition=nvl
#SBATCH --signal=USR2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=12
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mzaki4@jhu.edu
#SBATCH --output=${job_name}_%j.out
#SBATCH --error=${job_name}_%j.err
#SBATCH -A mshiel10

# --- Environment setup ---
cd "${SCRIPT_DIR}"

module load cuda/12.3
module load anaconda3/2024.02-1
conda activate /home/mzaki4/scratchmshiel10/mzaki4/bayesian_mpp

export CUDA_VISIBLE_DEVICES=0
if [ -z "\$CUDA_HOME" ]; then
    CUDA_HOME=\$(dirname \$(dirname \$(which nvcc)))
    export CUDA_HOME
fi
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# --- Run single combination ---
echo ">>> max_outputs=${max_outputs}  num_batches=${num_batches}"
python -u sens_ncwno_data_fast.py \
  --max_outputs ${max_outputs} \
  --num_batches ${num_batches} \
  > sens_ncwno_data_fast_out_nbatches_${num_batches}_maxoutputs_${max_outputs}_\${SLURM_JOB_ID}.txt \
  2> sens_ncwno_data_fast_err_nbatches_${num_batches}_maxoutputs_${max_outputs}_\${SLURM_JOB_ID}.txt
EOF

  done
done

echo ""
echo "All ${#MAX_OUTPUTS_LIST[@]} x ${#NUM_BATCHES_LIST[@]} = $(( ${#MAX_OUTPUTS_LIST[@]} * ${#NUM_BATCHES_LIST[@]} )) jobs submitted."
