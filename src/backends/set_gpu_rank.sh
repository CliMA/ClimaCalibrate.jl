#!/bin/bash
#
# Maps MPI ranks to GPUs for ClimaCalibrate's PBS backend.
#
# This is adapted from NCAR Derecho's /opt/utils/bin/set_gpu_rank.

case $LMOD_FAMILY_MPI in
    cray-mpich)
        export MPICH_GPU_SUPPORT_ENABLED=1
        export MPICH_OFI_NIC_POLICY=GPU
        local_rank=$PMI_LOCAL_RANK
        local_size=$PMI_LOCAL_SIZE
        ;;
    openmpi)
        local_rank=$OMPI_COMM_WORLD_LOCAL_RANK
        local_size=$OMPI_COMM_WORLD_LOCAL_SIZE
        ;;
esac

if [[ -n $local_rank && ${NGPUS:-0} -gt 0 ]]; then
    if [[ $((local_size % NGPUS)) -eq 0 ]]; then
        rank_gpu_id=$((local_rank / (local_size / NGPUS)))
    else
        rank_gpu_id=$((local_rank % NGPUS))
    fi
    export CUDA_VISIBLE_DEVICES=$rank_gpu_id
elif [[ -z $local_rank && ${NGPUS:-0} -gt 0 ]]; then
    # No recognized MPI library: expose all GPUs to the single process.
    export CUDA_VISIBLE_DEVICES=$(seq 0 $((NGPUS - 1)) | paste -sd ',' -)
fi

exec "$@"
