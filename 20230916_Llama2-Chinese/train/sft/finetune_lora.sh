output_model=save_folder
# 需要修改到自己的输入目录
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ./finetune.sh ${output_model}
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2  finetune_clm_lora.py \              # 用于训练的脚本
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \                             # 预训练模型路径
    --train_files ../../data/train_sft.csv \                                         # 训练数据
                ../../data/train_sft_sharegpt.csv \                                  # 训练数据
    --validation_files  ../../data/dev_sft.csv \                                     # 验证数据
                         ../../data/dev_sft_sharegpt.csv \                           # 验证数据
    --per_device_train_batch_size 1 \                                                # 每个设备的训练批次大小
    --per_device_eval_batch_size 1 \                                                 # 每个设备的验证批次大小
    --do_train \                                                                     # 是否训练
    --do_eval \                                                                      # 是否验证
    --use_fast_tokenizer false \                                                     # 是否使用快速分词器
    --output_dir ${output_model} \                                                   # 输出目录
    --evaluation_strategy  steps \                                                   # 评估策略
    --max_eval_samples 800 \                                                         # 最大验证样本数
    --learning_rate 1e-4 \                                                           # 学习率
    --gradient_accumulation_steps 8 \                                                # 梯度累积步数
    --num_train_epochs 10 \                                                          # 训练轮数
    --warmup_steps 400 \                                                             # 预热步数
    --load_in_bits 4 \                                                               # 加载位数
    --lora_r 8 \                                                                     # lora_r表示秩的大小
    --lora_alpha 32 \                                                                # lora_alpha表示控制模型对原始预训练参数的更新程度
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \       # 目标模块
    --logging_dir ${output_model}/logs \                                             # 日志目录
    --logging_strategy steps \                                                       # 日志策略
    --logging_steps 10 \                                                             # 日志步数
    --save_strategy steps \                                                          # 保存策略
    --preprocessing_num_workers 10 \                                                 # 预处理工作数
    --save_steps 20 \                                                                # 保存步数
    --eval_steps 20 \                                                                # 评估步数
    --save_total_limit 2000 \                                                        # 保存总数限制
    --seed 42 \                                                                      # 种子
    --disable_tqdm false \                                                           # 禁用tqdm
    --ddp_find_unused_parameters false \                                             # ddp_find_unused_parameters
    --block_size 2048 \                                                              # 块大小
    --report_to tensorboard \                                                        # 报告到tensorboard
    --overwrite_output_dir \                                                         # 覆盖输出目录
    --deepspeed ds_config_zero2.json \                                               # deepspeed配置文件
    --ignore_data_skip true \                                                        # 忽略数据跳过
    --bf16 \                                                                         # bf16
    --gradient_checkpointing \                                                       # 梯度检查点
    --bf16_full_eval \                                                               # bf16_full_eval
    --ddp_timeout 18000000 \                                                         # ddp_timeout
    | tee -a ${output_model}/train.log                                               # 日志输出

    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \                    # 恢复检查点