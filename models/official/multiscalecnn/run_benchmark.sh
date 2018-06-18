python multiscalecnn_main.py \
 --train_batch_size 8 \
 --train_steps 5000 \
 --num_intra_threads 10 \
 --num_inter_threads 2 \
 --data_dir /data/02/kdatta1 \
 --model_dir /tmp/mcnn/ \
 --use_tpu=False \
 --kmp_blocktime 1 \
 --kmp_settings 1 \
 --precision float32 \
 --mode train
