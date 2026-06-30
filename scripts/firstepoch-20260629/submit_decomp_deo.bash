#!/bin/bash
#SBATCH -N 1
#SBATCH -p xeon-g6-volta
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2:00:00
#SBATCH -J fe-decomp-deo
#SBATCH -o /home/gridsan/ihammond/firstepoch-compare-20260629/decomp/deo-slurm-%j.out

set -u
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
module load julia/1.11.3
module load intel
module load mpi   # after intel so system OpenMPI wins (fixes libmpi_mpifh ompi_instance_count)
export ONEAPI_ROOT=/state/partition1/llgrid/pkg/intel-oneapi/2025.0
export MKLROOT=$ONEAPI_ROOT/mkl/latest
IOMP_DIR=$(dirname "$(find "$ONEAPI_ROOT" -name libiomp5.so | head -n1)")
MPI_LIB_DIR=$(dirname "$(dirname "$(which mpirun)")")/lib
export LD_LIBRARY_PATH="$MPI_LIB_DIR:$IOMP_DIR:$MKLROOT/lib/intel64:$MKLROOT/lib:$LD_LIBRARY_PATH"
export MKL_THREADING_LAYER=INTEL
export JULIA_NUM_PRECOMPILE_TASKS=1   # serial precompile avoids Pidfile lock races

export FIRSTEPOCH_OLD_ROOT=/home/gridsan/ihammond/GitHub/Emitter3DTopOpt
export FIRSTEPOCH_NEW_ROOT=/home/gridsan/ihammond/GitHub/worktrees/DEO-firstepoch-20260629
export FIRSTEPOCH_RUNROOT=/home/gridsan/ihammond/firstepoch-compare-20260629
SCRIPTS="$FIRSTEPOCH_NEW_ROOT/scripts/firstepoch-20260629"
export JULIA_NUM_THREADS="${SLURM_CPUS_ON_NODE:-40}"

julia --project="$FIRSTEPOCH_NEW_ROOT" "$SCRIPTS/decomp_deo.jl"
rc=$?
echo "exit $rc at $(date)"
exit $rc
