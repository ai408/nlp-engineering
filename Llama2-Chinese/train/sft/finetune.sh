output_model=save_folder
# 需要修改到自己的输入目录
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ./finetune.sh ${output_model} # 复制脚本到输出目录
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2  finetune_clm.py \  # deepspeed：分布式训练，num_gpus：使用的gpu数量，finetune_clm.py：训练脚本
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \            # model_name_or_path：模型名称或路径
    --train_files ../../data/train_sft.csv \                        # train_files：训练数据集路径
                ../../data/train_sft_sharegpt.csv \                 # train_files：训练数据集路径
    --validation_files  ../../data/dev_sft.csv \                    # validation_files：验证数据集路径
                         ../../data/dev_sft_sharegpt.csv \          # validation_files：验证数据集路径
    --per_device_train_batch_size 1 \                               # per_device_train_batch_size：每个设备的训练批次大小
    --per_device_eval_batch_size 1 \                                # per_device_eval_batch_size：每个设备的验证批次大小
    --do_train \                                                    # do_train：是否训练
    --do_eval \                                                     # do_eval：是否验证
    --use_fast_tokenizer false \                                    # use_fast_tokenizer：是否使用快速分词器
    --output_dir ${output_model} \                                  # output_dir：输出目录
    --evaluation_strategy  steps \                                  # evaluation_strategy：评估策略
    --max_eval_samples 800 \                                        # max_eval_samples：最大评估样本数
    --learning_rate 1e-4 \                                          # learning_rate：学习率
    --gradient_accumulation_steps 8 \                               # gradient_accumulation_steps：梯度累积步数
    --num_train_epochs 10 \                                         # num_train_epochs：训练轮数
    --warmup_steps 400 \                                            # warmup_steps：预热步数
    --logging_dir ${output_model}/logs \                            # logging_dir：日志目录
    --logging_strategy steps \                                      # logging_strategy：日志策略
    --logging_steps 10 \                                            # logging_steps：日志步数
    --save_strategy steps \                                         # save_strategy：保存策略
    --preprocessing_num_workers 10 \                                # preprocessing_num_workers：预处理工作数
    --save_steps 20 \                                               # save_steps：保存步数
    --eval_steps 20 \                                               # eval_steps：评估步数
    --save_total_limit 2000 \                                       # save_total_limit：保存总数限制
    --seed 42 \                                                     # seed：随机种子
    --disable_tqdm false \                                          # disable_tqdm：禁用tqdm
    --ddp_find_unused_parameters false \                            # 注释：ddp查找未使用的参数
    --block_size 2048 \                                             # block_size：块大小
    --report_to tensorboard \                                       # report_to：报告给tensorboard
    --overwrite_output_dir \                                        # overwrite_output_dir：覆盖输出目录
    --deepspeed ds_config_zero2.json \                              # deepspeed：分布式训练配置文件
    --ignore_data_skip true \                                       # ignore_data_skip：忽略数据跳过
    --bf16 \                                                        # bf16：使用bf16
    --gradient_checkpointing \                                      # gradient_checkpointing：梯度检查点
    --bf16_full_eval \                                              # bf16_full_eval：bf16全评估
    --ddp_timeout 18000000 \                                        # ddp_timeout：ddp超时
    | tee -a ${output_model}/train.log                              # tee：将标准输出重定向到文件，同时显示在屏幕上

    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \    # resume_from_checkpoint：从检查点恢复