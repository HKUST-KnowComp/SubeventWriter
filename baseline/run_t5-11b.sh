eval_bs=$1
output_dir=$2
data_dir=$3
deepspeed zero_shot_prompt.py --deepspeed deepspeed/ds_config_zero3.json \
--per_device_train_batch_size 0 --per_device_eval_batch_size ${eval_bs} \
--gradient_accumulation_steps 0 --num_train_epochs 0 \
--model_name_or_path t5-11b --num_beams 4 --dataloader_num_workers 4 \
--output_dir ${output_dir} \
--max_source_length 128 --max_target_length 80 --val_max_target_length 80 \
--train_file ${data_dir}/sentence_train_42.json \
--validation_file ${data_dir}/sentence_valid_42.json \
--test_file ${data_dir}/sentence_test_42.json \
--text_column title --summary_column subevents \
--preprocessing_num_workers 4 --learning_rate 0 \
--save_strategy no \
--report_to none --dataloader_pin_memory False --overwrite_cache False \
--eval_accumulation_steps 8 --do_predict --predict_with_generate

