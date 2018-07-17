cur_date=`date +%F-%H-%M-%S`
result_dir="/tmp/mcnn_${cur_date}"
KMP_AFFINITY="granularity=fine,compact,1,0"
mkdir -p ${result_dir}

HOROVOD_TIMELINE=${result_dir}/horovod_timeline.json HOROVOD_FUSION_THRESHOLD=201326592 OMP_NUM_THREADS=10 \
 /nfs/pdx/home/sunchoi/openmpi-3.0/openmpi-3.0.0/bin/mpirun -np 4 -cpus-per-proc 10 \
 --map-by socket --report-bindings --oversubscribe \
 -x HOROVOD_FUSION_THRESHOLD -x OMP_NUM_THREADS -x LD_LIBRARY_PATH -x PATH\
 numactl -l python multiscalecnn_main.py \
 --train_batch_size 32 \
 --train_steps=10 \
 --num_intra_threads 10 \
 --num_inter_threads 2 \
 --data_dir /data/02/sunchoi/large_aug_slices/ \
 --model_dir ${result_dir} \
 --data_format NCHW \
 --use_tpu=False \
 --precision float32 \
 --mode train_and_eval \
 --kmp_blocktime 1 \
 --trace \
 --kmp_affinity ${KMP_AFFINITY}
