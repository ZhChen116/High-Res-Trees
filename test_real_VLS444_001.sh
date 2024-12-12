#!/bin/bash 
#SBATCH --job-name=test_simple_var
#SBATCH --account=commons 
#SBATCH --partition=high_priority
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1024m 
#SBATCH --time=3:00:00 
#SBATCH --mail-user=zc56@rice.edu 
#SBATCH --mail-type=ALL


module load GCCcore/13.2.0
module load Python/3.11.5

echo "My job ran on:" 
echo $SLURM_NODELIST 


if [[ -d $SHARED_SCRATCH/$USER && -w $SHARED_SCRATCH/$USER ]]; then
    cd $SHARED_SCRATCH/$USER
    srun python /home/zc56/Bayes_Tensor_Tree/serverBTR/test_real_VLS444_001.py
fi