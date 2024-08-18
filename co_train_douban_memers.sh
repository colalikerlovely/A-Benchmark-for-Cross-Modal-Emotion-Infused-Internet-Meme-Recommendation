DATA_PATH=/data/Datasets/Douban_MEMERS
python -m torch.distributed.launch --nproc_per_node=1  --master_port 12963 \
train_image.py \
--do_train  --num_thread_reader=0 --epochs=4 --batch_size=128 --n_display=20 \
--train_csv ${DATA_PATH}/input_file/train_id_9k.csv \
--val_csv ${DATA_PATH}/input_file/test_4k.json  \
--data_path ${DATA_PATH}/input_file/train_9k.json \
--features_path ${DATA_PATH}/image \
--train_emotion ${DATA_PATH}/input_file/train_emotion.json \
--test_emotion ${DATA_PATH}/input_file/test_emotion.json \
--output_dir ckpts/IEF_model \
--lr 1e-4 --max_words 32 --image_num 1 --batch_size_val 128 \
--datatype Douban_MEMERS \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32 \
--interaction wti  --text_pool_type transf_avg \
--world_size 1;\
DATA_PATH=data/Datasets/Douban_MEMERS
python -m torch.distributed.launch --nproc_per_node=1  --master_port 12962 \
train_caption.py \
--do_train  --num_thread_reader=0 --epochs=4 --batch_size=128 --n_display=20 \
--train_csv ${DATA_PATH}/input_file/train_id_9k.csv \
--val_csv ${DATA_PATH}/input_file/test_4k.json  \
--data_path ${DATA_PATH}/input_file/train_9k.json \
--features_path ${DATA_PATH}/image \
--train_emotion ${DATA_PATH}/input_file/train_emotion.json \
--test_emotion ${DATA_PATH}/input_file/test_emotion.json \
--output_dir ckpts/IEF_model \
--lr 1e-4 --max_words 32 --image_num 1 --batch_size_val 128 \
--datatype Douban_MEMERS  \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32 \
--interaction dp  --text_pool_type transf_avg \
--world_size 1 \
