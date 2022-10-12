lr=$1
epoch=$2
train_bs=$3
accu_step=$4
eval_bs=$5
model_name=$6
prompt_type=$7
optimizer=$8
output_dir=$9
data_dir=${10}
CUDA_VISIBLE_DEVICES=0 python ds_t5_finetune.py \
--per_device_train_batch_size ${train_bs} --per_device_eval_batch_size ${eval_bs} \
--gradient_accumulation_steps ${accu_step} --num_train_epochs ${epoch} \
--model_name_or_path ${model_name} --num_beams 4 --dataloader_num_workers 4 \
--output_dir ${output_dir} \
--max_source_length 128 --max_target_length 80 --val_max_target_length 80 \
--train_file ${data_dir}/sentence_train_42.json \
--validation_file ${data_dir}/sentence_valid_42.json \
--test_file ${data_dir}/sentence_test_42.json \
--text_column title --summary_column subevents --prompt_type ${prompt_type} \
--preprocessing_num_workers 4 --optimizer ${optimizer} --learning_rate ${lr} \
--evaluation_strategy epoch --prediction_loss_only --save_strategy no \
--report_to none --dataloader_pin_memory False --overwrite_cache False \
--store_generation --eval_accumulation_steps 8 --group_by_length --save_checkpoint --do_predict --do_train
