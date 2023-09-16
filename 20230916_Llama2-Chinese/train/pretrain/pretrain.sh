output_model=/mnt/data1/atomgpt                                  # output_model：输出模型路径
if [ ! -d ${output_model} ];then                                 # -d：判断是否为目录，如果不是目录则创建
    mkdir ${output_model}                                        # mkdir：创建目录
fi
cp ./pretrain.sh ${output_model}                                 # cp：复制文件pretrain.sh到output_model目录下
cp ./ds_config_zero*.json ${output_model}                        # cp：复制文件ds_config_zero*.json到output_model目录下

deepspeed --num_gpus 1 pretrain_clm.py \                         # deepspeed：分布式训练，num_gpus：使用的gpu数量，pretrain_clm.py：训练脚本
    --model_name_or_path L:/20230903_Llama2/Llama-2-7b-hf \      # model_name_or_path：模型名称或路径
    --train_files ../../data/train_sft.csv \                     # train_files：训练数据集路径
                ../../data/train_sft_sharegpt.csv \
    --validation_files  ../../data/dev_sft.csv \                 # validation_files：验证数据集路径
                         ../../data/dev_sft_sharegpt.csv \
    --per_device_train_batch_size 10 \                           # per_device_train_batch_size：每个设备的训练批次大小
    --per_device_eval_batch_size 10 \                            # per_device_eval_batch_size：每个设备的验证批次大小
    --do_train \                                                 # do_train：是否进行训练
    --output_dir ${output_model} \                               # output_dir：输出路径
    --evaluation_strategy  steps \                               # evaluation_strategy：评估策略，steps：每隔多少步评估一次
    --use_fast_tokenizer false \                                 # use_fast_tokenizer：是否使用快速分词器
    --max_eval_samples 500 \                                     # max_eval_samples：最大评估样本数，500：每次评估500个样本
    --learning_rate 3e-5 \                                       # learning_rate：学习率
    --gradient_accumulation_steps 4 \                            # gradient_accumulation_steps：梯度累积步数
    --num_train_epochs 3 \                                       # num_train_epochs：训练轮数
    --warmup_steps 10000 \                                       # warmup_steps：预热步数
    --logging_dir ${output_model}/logs \                         # logging_dir：日志路径
    --logging_strategy steps \                                   # logging_strategy：日志策略，steps：每隔多少步记录一次日志
    --logging_steps 2 \                                          # logging_steps：日志步数，2：每隔2步记录一次日志
    --save_strategy steps \                                      # save_strategy：保存策略，steps：每隔多少步保存一次
    --preprocessing_num_workers 10 \                             # preprocessing_num_workers：预处理工作数
    --save_steps 500 \                                           # save_steps：保存步数，500：每隔500步保存一次
    --eval_steps 500 \                                           # eval_steps：评估步数，500：每隔500步评估一次
    --save_total_limit 2000 \                                    # save_total_limit：保存总数，2000：最多保存2000个
    --seed 42 \                                                  # seed：随机种子
    --disable_tqdm false \                                       # disable_tqdm：是否禁用tqdm
    --ddp_find_unused_parameters false \                         # ddp_find_unused_parameters：是否找到未使用的参数
    --block_size 4096 \                                          # block_size：块大小
    --overwrite_output_dir \                                     # overwrite_output_dir：是否覆盖输出目录
    --report_to tensorboard \                                    # report_to：报告给tensorboard
    --run_name ${output_model} \                                 # run_name：运行名称
    --bf16 \                                                     # bf16：是否使用bf16
    --bf16_full_eval \                                           # bf16_full_eval：是否使用bf16进行完整评估
    --gradient_checkpointing \                                   # gradient_checkpointing：是否使用梯度检查点
    --deepspeed ./ds_config_zero3.json \                         # deepspeed：分布式训练配置文件
    --ignore_data_skip true \                                    # ignore_data_skip：是否忽略数据跳过
    --ddp_timeout 18000000 \                                     # ddp_timeout：ddp超时时间，18000000：18000000毫秒
    | tee -a ${output_model}/train.log                           # tee：将标准输出重定向到文件，-a：追加到文件末尾
    
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \# resume_from_checkpoint：从检查点恢复训练