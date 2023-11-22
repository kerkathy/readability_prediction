# write a train script for main.py with arguments
# Usage: bash train.sh
# --do_eval \
# --resume_from_checkpoint output/checkpoint-500/ \

python ./main.py \
--model_name_or_path bert-base-cased \
--task_name ose \
--do_train \
--do_eval \
--max_train_samples 40 \
--max_eval_samples 40 \
--train_file "data/ose/short_train_pair.csv" \
--validation_file "data/ose/short_val_pair.csv" \
--test_file "data/ose/short_test_pair.csv" \
--pad_to_max_length True \
--max_seq_length 512 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--overwrite_output_dir \
--output_dir output/ \
--logging_strategy steps \
--logging_steps 4 \
--report_to wandb \