#!/bin/bash
#
#SBATCH --output /data/hpcdata/users/jambyr/icenet2/pipeline/g%j.%N.out
#SBATCH --error /data/hpcdata/users/jambyr/icenet2/pipeline/g%j.%N.err
#SBATCH --chdir /data/hpcdata/users/jambyr/icenet2/pipeline
#SBATCH --job-name gItest
#SBATCH --time=1:00:00
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --mail-user=jambyr@bas.ac.uk
#SBATCH --partition=gpu
#SBATCH --account=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=node022

. "/data/hpcdata/users/jambyr/miniconda3/etc/profile.d/conda.sh"
conda activate icenet2
cd /data/hpcdata/users/jambyr/icenet2/pipeline

LOGFILE="gpunode.`date +%d%m%y`.log"
echo "START `date +%F-%T`" >$LOGFILE
python train_model.py tffiles 2>&1 >>$LOGFILE
echo "END `date +%F-%T`" >>$LOGFILE


