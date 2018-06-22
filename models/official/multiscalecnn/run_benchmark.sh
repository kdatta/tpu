python multiscalecnn_main.py \
 --train_batch_size 8 \
 --train_steps 1000 \
 --num_intra_threads 30 \
 --num_inter_threads 4 \
 --data_dir /data/02/kdatta1 \
 --model_dir /tmp/mcnn \
 --use_tpu=False \
 --precision float32 \
 --mode train_and_eval
