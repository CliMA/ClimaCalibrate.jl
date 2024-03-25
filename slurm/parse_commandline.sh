# Default arguments
slurm_time="2:00:00"
slurm_ntasks="1"
slurm_cpus_per_task="1"
slurm_gpus_per_task="0"
generate_data=false
help_message="Usage:
    ./pipeline.sh [options] experiment_id

Options:
    -t, --time=HH:MM:SS: Set max wallclock time (default: 2:00:00).
    -n, --ntasks:        Set number of tasks to launch (default: 1).
    -c, --cpus_per_task: Set CPU cores per task (mutually exclusive with -g, default: 8).
    -g, --gpus_per_task: Set GPUs per task (mutually exclusive with -c, default: 0).
    --generate_data:     If set, generates observational data for use in the calibration.
    -h, --help:          Display this help message.

Arguments:
    experiment_id:   A unique identifier for your experiment (required)."

# Parse arguments using getopt
VALID_ARGS=$(getopt -o h,t:,n:,c:,g: --long help,time:,ntasks:,cpus_per_task:,gpus_per_task:,generate_data -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"

# Process arguments
while [ : ]; do
  case "$1" in
    -t | --time)
        slurm_time="$2"
        shift 2
        ;;
    -n | --ntasks)
        slurm_ntasks="$2"
        shift 2
        ;;
    -c | --cpus_per_task)
        slurm_cpus_per_task="$2"
        shift 2
        ;;
    -g | --gpus_per_task)
        slurm_gpus_per_task="$2"
        shift 2
        ;;
    --generate_data)
        generate_data=true
        shift 1
        ;;
    -h | --help)
        printf "%s\n" "$help_message"
        exit 0
        ;;
    --) shift; break ;;  # End of options
  esac
done

experiment_id="$1"
if [ -z $experiment_id ] ; then
    echo "Error: No experiment ID provided."
    exit 1
fi

# Get values from EKP config file
ensemble_size=$(grep "ensemble_size:" experiments/$experiment_id/ekp_config.yml | awk '{print $2}')
n_iterations=$(grep "n_iterations:" experiments/$experiment_id/ekp_config.yml | awk '{print $2}')
output=$(grep "output_dir:" experiments/$experiment_id/ekp_config.yml | awk '{print $2}')
logfile=$output/experiment_log.txt

# Set partition
if [[ $slurm_gpus_per_task -gt 0 ]]; then
    partition=gpu
else
    partition=expansion
fi

# Output slurm configuration
echo "Running experiment: $experiment_id"
indent=" └ "
printf "Slurm configuration (per ensemble member):\n"
printf "%sTime limit: %s\n" "$indent" "$slurm_time"
printf "%sTasks: %s\n" "$indent" "$slurm_ntasks"
printf "%sCPUs per task: %s\n" "$indent" "$slurm_cpus_per_task"
printf "%sGPUs per task: %s\n" "$indent" "$slurm_gpus_per_task"
echo ""
