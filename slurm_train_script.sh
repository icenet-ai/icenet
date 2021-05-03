#!/bin/bash
#
#SBATCH -o /data/hpcdata/users/tomand/gpu%j.%N.out
#SBATCH -D /data/hpcdata/users/tomand
#SBATCH -J gpu
#SBATCH --time=48:00:00
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --mail-user=tomand@bas.ac.uk
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name=EGU_icenet2
#SBATCH --account=gpu
#SBATCH --nodelist=node022
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#

# Get command line inputs
SEED=42
LR=0.0001
FILTER_SIZE=3
N_FILTERS_FACTOR=2
LR_10E_DECAY_FAC=0.5
LR_DECAY_START=10
LR_DECAY_END=30

for arg in "$@"
do
        case $arg in
                --seed=*)
                SEED="${arg#*=}"
                shift
                ;;
                --lr=*)
                LR="${arg#*=}"
                shift
                ;;
                --filter_size=*)
                FILTER_SIZE="${arg#*=}"
                shift
                ;;
                --n_filters_factor=*)
                N_FILTERS_FACTOR="${arg#*=}"
                shift
                ;;
                --lr_10e_decay_fac=*)
                LR_10E_DECAY_FAC="${arg#*=}"
                shift
                ;;
                --lr_decay_start=*)
                LR_DECAY_START="${arg#*=}"
                shift
                ;;
                --lr_decay_end=*)
                LR_DECAY_END="${arg#*=}"
                shift
                ;;
        esac
done

# Print variable
echo -e "\n\n\n"
echo "SEED=$SEED"
echo "LR=$LR"
echo "FILTER_SIZE=$FILTER_SIZE"
echo "N_FILTERS_FACTOR=$N_FILTERS_FACTOR"
echo "LR_10E_DECAY_FAC=$LR_10E_DECAY_FAC"
echo "LR_DECAY_START=$LR_DECAY_START"
echo "LR_DECAY_END=$LR_DECAY_END"
echo -e "\n\n\n"

# Load the module script (if modules are required)
. /hpcpackages/python/miniconda3/etc/profile.d/conda.sh                                                                             
conda activate /data/hpcdata/users/tomand/anaconda/envs/icenet2
cd /data/hpcdata/users/tomand/code/icenet2/                                                                       
# Train the network

python3 -u icenet2/train_icenet2.py --seed $SEED --lr $LR --filter_size $FILTER_SIZE --n_filters_factor $N_FILTERS_FACTOR --lr_10e_decay_fac $LR_10E_DECAY_FAC --lr_decay_start $LR_DECAY_START --lr_decay_end $LR_DECAY_END > slurm_logs/$SEED.txt 2>&1
