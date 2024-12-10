#!/usr/bin/env bash
#SBATCH --ntasks=11
#SBATCH --cpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH -J fig3_mpi_20
#SBATCH -o log.%j
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

echo '-------------------------------'
cd ${SLURM_SUBMIT_DIR}
echo ${SLURM_SUBMIT_DIR}
echo Running on host $(hostname)
echo Time is $(date)
echo SLURM_NODES are $(echo ${SLURM_NODELIST})
echo '-------------------------------'
echo -e '\n\n'

mpiexec -np 1 R --no-save < fig3_mpi_20.R

