cur_date=`date +%F-%H-%M-%S`
result_dir="/tmp/mcnn_${cur_date}"
mkdir -p ${result_dir}

OMP_NUM_THREADS=10 mpirun -np 32 -cpus-per-proc 10 --map-by socket \
 -H gold01-opa,gold09-opa,gold03-opa,gold04-opa,gold05-opa,gold06-opa,gold07-opa,gold08-opa \
 --oversubscribe -x OMP_NUM_THREADS \
 numactl -l python multiscalecnn_main.py \
 --train_batch_size 256 \
 --eval_batch_size 16 \
 --train_steps 2000 \
 --num_intra_threads 10 \
 --num_inter_threads 4 \
 --data_dir /data/02/kdatta1 \
 --model_dir ${result_dir} \
 --use_tpu=False \
 --precision float32 \
 --mode train_and_eval 2>&1 | tee ${result_dir}/outfile
