weight=$1
eval_bs=$2
prompt_type=$3
model_path=$4
coh_model=$5
output_dir=$6
data_dir=$7
CUDA_VISIBLE_DEVICES=0 python ds_t5_finetune.py \
--per_device_train_batch_size 0 --per_device_eval_batch_size ${eval_bs} \
--gradient_accumulation_steps 0 --num_train_epochs 0 \
--model_name_or_path ${model_path} --num_beams 4 --dataloader_num_workers 4 \
--output_dir ${output_dir} \
--max_source_length 128 --max_target_length 80 --val_max_target_length 80 \
--train_file ${data_dir}/sentence_train_42.json \
--validation_file ${data_dir}/sentence_valid_42.json \
--test_file ${data_dir}/sentence_test_42.json \
--text_column title --summary_column subevents --prompt_type ${prompt_type} \
--preprocessing_num_workers 4 --learning_rate 0 \
--evaluation_strategy epoch --prediction_loss_only --save_strategy no \
--report_to none --dataloader_pin_memory False --overwrite_cache False \
--store_generation --eval_accumulation_steps 1 --iterative --group_by_length \
--do_iterative_predict --global_controller \
--coherence_model_path ${coh_model} \
--coherence_weight ${weight}
