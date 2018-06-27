export OMP_NUM_THREADS=40

python multiscalecnn_main.py \
 --train_batch_size 8 \
 --train_steps 1000 \
 --num_intra_threads 40 \
 --num_inter_threads 8 \
 --data_dir /data01/kushal/novartis/mcnn \
 --model_dir /tmp/mcnn \
 --use_tpu=False \
 --precision float32 \
 --mode train_and_eval
