GPUS=$1
NODES=$2
PORT=$3
SLEEPTIME=$4

date
#sleep 28500
#date
CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$NODES --master_port=$PORT main.py --config config/ntu/train_joint_amstgcn_ntu.yaml
