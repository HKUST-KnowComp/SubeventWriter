lr=$1
epoch=$2
train_bs=$3
output_dir=$4
data_dir=$5
CUDA_VISIBLE_DEVICES=0 python train_coherence_model.py \
--per_device_train_batch_size ${train_bs} --per_device_eval_batch_size 32 \
--gradient_accumulation_steps 1 --num_train_epochs ${epoch} \
--model_name_or_path bert-base-uncased --dataloader_num_workers 4 \
--output_dir ${output_dir} --negative_size 4 \
--sentence1_column title --sentence2_column subevents \
--train_file ${data_dir}/coherence_train_42.json \
--validation_file ${data_dir}/coherence_valid_42.json \
--test_file ${data_dir}/coherence_test_42.json \
--preprocessing_num_workers 4 --optimizer AdamW --learning_rate ${lr} \
--evaluation_strategy epoch --save_strategy no \
--report_to none --dataloader_pin_memory False --overwrite_cache False \
--eval_accumulation_steps 8 --group_by_length --do_eval \
--save_checkpoint --do_train