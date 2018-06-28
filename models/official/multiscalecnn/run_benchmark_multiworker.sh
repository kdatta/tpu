OMP_NUM_THREADS=10 mpirun -np 4 --map-by socket -x OMP_NUM_THREADS numactl -l python multiscalecnn_main.py \
 --train_batch_size 32 \
 --train_steps 1000 \
 --num_intra_threads 10 \
 --num_inter_threads 4 \
 --data_dir /data01/kushal/novartis/mcnn \
 --model_dir /tmp/test \
 --use_tpu=False \
 --precision float32 \
 --mode train_and_eval 
# --trace
