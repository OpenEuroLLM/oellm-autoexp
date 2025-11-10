#!/bin/bash
#SBATCH --job-name=34B-16N-reproducible-no_tuner
#SBATCH --nodes=16
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=00:40:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000353
#SBATCH -o logs/%x.log

set -eox pipefail
echo "Starting bash script"
module purge
#Save the current working directory for shorter path variables
wd=(`pwd`)

#COMPILER
#This is for the c++ dataset helper
export CC=gcc-12
export CXX=g++-12

# parsing input arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export CUDA_DEVICE_MAX_CONNECTIONS=1                    #This is needed for sequence paralellism

#OMP THREADING
export OMP_NUM_THREADS=1                                #Virtual threads to accompany the HW threads set by slurm

#HSA=Heterogeneous System Architecture
#AMD provides HSA through ROCR https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.2.4/index.html#hsa-runtime-api-and-runtime-for-rocm
export HSA_ENABLE_SDMA=0                                #https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-gpu-aware-mpi-readme/
                                                        #https://rocm.docs.amd.com/en/docs-6.2.0/conceptual/gpu-memory.html#system-direct-memory-access
                                                        #HSA_ENABLE_SDMA=0 -- Uses blit kernels for comms through the gpu -- more bandwith between gcd's
                                                        #With HSA_ENABLE_SDMA=1 capped at 50GB/S uni-directional bandwidth
                                                        #Enables overlapping communication with computation
#export HSA_ENABLE_PEER_SDMA=0                          #Enable or disable SDMA engines for device-to-device copies, HSA_ENABLE_SDMA overrides this value
                                                        #https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.2.4/api-reference/environment_variables.html
#export HSA_XNACK=1                                     #A fallback for pagefault https://rocm.docs.amd.com/en/docs-6.2.4/conceptual/gpu-memory.html#xnack

export PYTHONWARNINGS=ignore
#####
#DEBUGGING, INCREASE VERBOSITY IN LOGS

#TORCH
#export TORCH_SHOW_CPP_STACKTRACES=1                    #C++ traceback for pytorch crashes
#export TORCH_DIST_INIT_BARRIER=1                       #Initialize torch distributed new group instance with a barrier to synchronize ranks

##TORCH COMPILE/TORCHINDUCTOR##
#export TORCH_COMPILE_DISABLE=1                         #Disable torch.compile
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1             #Disable all caching for Torchinductor
export TORCHINDUCTOR_MAX_AUTOTUNE=1                     #More rigorous autotuning done by Triton for GPU kernels, this might lead to more performant kernels at the cost of increased compilation time
#export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1             #Enabling this decreases the chance of cache hit slightly
#export TORCHINDUCTOR_COMPILE_THREADS=$
export TORCH_LOGS="+inductor,+dynamo"
export TORCHDYNAMO_VERBOSE=1
export ROCM_PYTORCH_ARCH=gfx90a

##TORCHNCCL##
#TORCH FLIGHTRECORDER, see https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html
#export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
#export TORCH_NCCL_DUMP_ON_TIMEOUT=true
#export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=logs/fr/nccl_trace_rank

##LIBFABRIC##
#export FI_LOG_LEVEL=debug                              #Very verbose Libfabric logging. May cause performance degredation when set to 'debug'
export FI_LOG_PROV=cxi                                  #Determine which libfabric provider to get logs from
#export FI_CXI_ODP=1                                    #On demand paging. This should be an alternative to the CXI_FORK_SAFE for unsafe fork. See more with man intro_mpi
export FI_HMEM=rocr                                     #Ensure that libfabric uses rocr's hmem implementation. Not sure if this supported for rocm < 6.4
export FI_HMEM_ROCR_USE_DMABUF=1
export FI_MR_ROCR_CACHE_MONITOR_ENABLED=1               #Detecting ROCR device memory (FI_HMEM_ROCR) changes made between the device virtual addresses used by an application and the underlying device physical pages.
#export SLINGSHOT_DEVICES=cxi0,cxi1                     #Specify which cxi NIC's get enabled
#export FI_CXI_REQ_BUF_SIZE=
#export FI_CXI_RDZV_GET_MIN=16364                       #Min byte size of message for cxi to use rendezvouz instead of eager messaging
#export FI_CXI_RDZV_THRESHOLD=32728                     #Max byte size
#export FI_CXI_DEFAULT_TX_SIZE=4096                     #Default is 1024
#export FI_CXI_RX_SIZE=4096
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software                    #Address matching can be either hardware(cxi) or software
export FI_CXI_RDZV_PROTO=alt_read
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

#NCCL/RCCL
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1                 #Print c++ code traceback for nccl collectives
export NCCL_DEBUG=INFO                                  #Verbose logging from RCCL
export NCCL_DEBUG_SUBSYS=ALL                            #Specify which stuff is logged from NCCL_DEBUG=INFO. Choices are P2P(Peer2Peer), COLL(Collecetives) etc.
#export NCCL_NCHANNELS_PER_PEER=16                      #Increases the amount of channels available for p2p communication. Should be a slight performance boost. Might lead to NaN's
#export NCCL_MAX_NCHANNELS=16
#export HIP_LAUNCH_BLOCKING=1                           #MAKES NCCL COLLECTIVES SYNCHRONOUS INSTEAD OF ASYNC
export NCCL_DMABUF_ENABLE=1                             #Enable DMA buffers from the RCCL side
#export GPU_MAX_HW_QUEUES=2
#export NCCL_SET_THREAD_NAME=1                          #Give meaningful names to NCCL cpu threads
export NCCL_TUNER_PLUGIN=/projappl/project_462000353/villekom/rccl-tuner/librccl-tuner.so
#export NCCL_TOPO_DUMP_FILE=/scratch/project_462000963/users/villekom/frameworks/Megatron-LM/logs/topo

##HIP/AMD##
#export AMD_SERIALIZE_KERNEL=3                          #Verbose logs from HIP
#export AMD_LOG_LEVEL=1                                 #Very verbose logging. In our context seems to concern hipified cuda kernels
#export HIPBLASLT_LOG_LEVEL=4                            #https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/logging-heuristics.html#logging-heuristics
#export HIPBLASLT_LOG_FILE=logs/hipblas_%i.log           #Redirect hipblas logging output

##TRITON##
#KEEP THIS AT 1
export TRITON_ALWAYS_COMPILE=1                         #Always force Triton to compile even in the case of cache hit.

##TransformerEngine##
#export NVTE_DEBUG=1                                    #Shows things like what dot product attention implementation is being used
#export NVTE_DEBUG_LEVEL=2
#export NVTE_USE_RMSNORM_TRITON=1                        #Experimental RMRSNORM triton kernel optimized for AMD/ROCM
#export GEMM_TUNING=1
#export TE_HIPBLASLT_TUNING_RUN_COUNT=10
#export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
#export TE_HIPBLASLT_ALGO_SAVE=logs/algo_tune.csv

# sets TRAIN_DATA and VALIDATION_DATA
#TRAIN_DATA=reproducible/redpajama-text_document
DATA_CACHE=/flash/project_462000353/megatron_cache/
MERGES=reproducible/merges.txt
VOCAB=reproducible/vocab.json

#Optional cli arguments
LR="${LR:-1.5e-4}"
MIN_LR="${MIN_LR:-0}"
MODEL_SIZE="${MODEL_SIZE:-8}"
TP_SIZE="${TP:-2}"
PP_SIZE="${PP:-1}"
VPP_SIZE="${VPP_SIZE:-1}"
NN="${NN:-$SLURM_NNODES}"
N_tasks="${N_tasks:-$SLURM_NTASKS}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
SEQ_LEN="${SEQ_LEN:-4096}"
ENABLE_PROFILING="${ENABLE_PROFILING:-0}" #enable pytorch profiling
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-"debug"}"

WARMUP_FRACTION=1/10
COOLDOWN_FRACTION=1/5

##BATCH SIZE
GLOBAL_BATCH_SIZE="${GBS:-1024}"
#MBS max 2 with the default model and model parallel parameters
MICRO_BATCH_SIZE="${MBS:-2}"

divide_rounding_up() {
    echo $((($1+$2-1)/$2))
}

# Set LR_WARMUP_ITERS and LR_WSD_DECAY_ITERS based on WARMUP_FRACTION
# and COOLDOWN_FRACTION
LR_WARMUP_ITERS=$((TRAIN_ITERS*${WARMUP_FRACTION}))
LR_WSD_DECAY_ITERS=$((TRAIN_ITERS*${COOLDOWN_FRACTION}))

# LR_DECAY_ITERS is simply set to TRAIN_ITERS
LR_DECAY_ITERS=$TRAIN_ITERS

#LLama 34B
if [[ $MODEL_SIZE -eq 34 ]]; then
    MODEL=PORO_34B
    NLAYERS=56
    NHIDDEN=7168
    NHEADS=56
    FFN_HIDDEN_SIZE=20480
    SEQ_LEN=4096
    NUM_KV_HEADS=8
    NUM_QUERY_GROUPS=8
elif [[ $MODEL_SIZE -eq 8 ]]; then
    MODEL=LLAMA3_8B
    NHIDDEN=4096
    FFN_HIDDEN_SIZE=14336
    NLAYERS=32
    NHEADS=32
    SEQ_LEN=2048
    NUM_KV_HEADS=8
    GROUP_SIZE=4
    NUM_QUERY_GROUPS=8
fi

TENSORBOARD_PATH=/scratch/project_462000353/$USER/logs/megLM/$SLURM_JOB_NAME
WANDB_SAVE_DIR=/scratch/project_462000353/$USER/logs/megLM/$SLURM_JOB_NAME
CHECKPOINT_PATH=/scratch/project_462000353/$USER/ckpts/megLM/$SLURM_JOB_NAME

if [[ -d "$TENSORBOARD_PATH" && $(ls -A "$TENSORBOARD_PATH" | wc -l) -gt 0 ]]; then
    rm -rf "$TENSORBOARD_PATH"
fi
if [[ -d "$CHECKPOINT_PATH" && $(ls -A "$CHECKPOINT_PATH" | wc -l) -gt 0 ]]; then
    rm -rf "$CHECKPOINT_PATH"
fi

LOG_INTERVAL=1
SAVE_INTERVAL=500
EVAL_INTERVAL=2000
EVAL_ITERS=1
INIT_METHOD_STD=0.00747017

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --use-distributed-optimizer \
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.05 \
    "

LAST_RANK=$((SLURM_NTASKS - 1))

#PYTORCH PROFILER ARGS
PROFILE_ARGS=" \
    --profile \
    --use-pytorch-profiler \
    --profile-step-start 6 \
    --profile-step-end 10 \
    --profile-ranks 0 1 2 3 4 5 6 7 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    "

DATA_ARGS=(
    --mock-data
    --data-cache-path
    "$DATA_CACHE"
    --tokenizer-type
    HuggingFaceTokenizer
    --tokenizer-model
    google/gemma-3-27b-pt
    --dataloader-type
    single
    --num-workers
    2
)

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --max-position-embeddings $SEQ_LEN \
    --seq-length $SEQ_LEN \
    --train-iters $TRAIN_ITERS \
    --eval-iters $EVAL_ITERS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --tokenizer-type GPT2BPETokenizer \
    --normalization RMSNorm \
    --bf16 \
    --init-method-std $INIT_METHOD_STD \
    --seed 42 \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --use-rotary-position-embeddings \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    --distributed-timeout-minutes 10 \
    --recompute-activations \
    "

FUSED_KERNEL_ARGS=" \
    --no-gradient-accumulation-fusion \
    "

OUTPUT_ARGS=" \
    --log-throughput \
    --log-timers-to-tensorboard \
    --log-interval 1 \
    --logging-level 40 \
    "

PARALLEL_ARGS="\
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel \
"

if (( VPP_SIZE > 1)); then
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --num-layers-per-virtual-pipeline-stage $VPP_SIZE"
fi

CMD=" \
    pretrain_gpt.py \
    $GPT_ARGS \
    $PARALLEL_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZER_ARGS \
    ${DATA_ARGS[@]} \
    $FUSED_KERNEL_ARGS \
    "

if [ "$ENABLE_PROFILING" -eq 1 ]; then
    CMD+=" $PROFILE_ARGS"
fi

c="fe"

# Bind mask for one hardware thread per core
# Set hardware threads on slurm with --threads-per-core. Default is 1.
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

echo "START $SLURM_JOBID: $(date)"
echo "NNODES" $SLURM_NNODES
echo "CPUS PER TASK" $SLURM_CPUS_PER_TASK

#SINGULARITY ENVIRONMENT
export CONTAINER=/scratch/project_462000394/containers/for-turkunlp-team/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1-te-dockerhash-560b76ceab56.sif
export SINGULARITY_BIND=/boot/config-5.14.21-150500.55.49_13.0.56-cray_shasta_c,/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd,/scratch/project_462000394/containers/for-turkunlp-team/tuner-2025-07-09/
echo "CONTAINER" $CONTAINER

export PWD=(`pwd -P`)

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
# --network=disable_rdzv_get: An alternative rendezvous protocol designed by HPE spesifically for AI workloads
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    --cpu-bind=mask_cpu:$BIND_MASK \
    --network=disable_rdzv_get \
    "

srun $SRUN_ARGS \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    ./slurm_scripts/launch_debug.sh \
    $CMD
