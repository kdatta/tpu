#!/usr/bin/env bash
cur_date=`date +%F-%H-%M-%S`
result_dir="/tmp/mcnn_${cur_date}"
KMP_AFFINITY="granularity=fine,compact,1,0"
mkdir -p ${result_dir}

#HOROVOD_TIMELINE=${result_dir}/horovod_timeline.json HOROVOD_FUSION_THRESHOLD=201326592 OMP_NUM_THREADS=10 \
OMP_NUM_THREADS=10 \
 /nfs/pdx/home/kdatta1/openmpi/bin/mpirun -np 4 \
 --map-by socket --report-bindings --oversubscribe \
 -x OMP_NUM_THREADS -x LD_LIBRARY_PATH -x PATH\
 numactl -l python multiscalecnn_main.py \
 --train_batch_size 32 \
 --train_steps=10 \
 --num_intra_threads 10 \
 --num_inter_threads 8 \
 --kmp_affinity ${KMP_AFFINITY} \
 --kmp_blocktime 1 \
 --data_dir /data01/kushal/novartis/mcnn \
 --model_dir ${result_dir} \
 --data_format NCHW \
 --use_tpu=False \
 --precision float32 \
 --mode train_and_eval \
 --image_height 1024 \
 --image_width 1280 \
 --trace
