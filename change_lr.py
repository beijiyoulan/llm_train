#!/usr/bin/env python
# coding: utf-8

import os
from datetime import datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline
from trl import setup_chat_format, SFTTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import gc

def evaluate_position(sample):
    try:
        prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
        # 提取验证集的位置信息
        generated_positions = json.loads(sample["messages"][2]["content"])["positions"]
        #print(generated_positions)
        # 提取生成的位置信息
        #print(json.loads(sample["messages"][2]["content"])["explanation"])
        #print(f"Generated Answer:{outputs[0]['generated_text'][len(prompt):].strip()}")
        true_positions = json.loads(outputs[0]["generated_text"][len(prompt):].strip())["positions"]
        #print(true_positions)
        # 比较生成的位置信息和验证集中的位置信息是否相等
        if generated_positions == true_positions:
            #print("right")
            return 1
        else:
            #print("wrong")
            return 0
    except json.JSONDecodeError:
        # 如果无法解析为 JSON，直接输出“error format”
        #print("error format")
        return 0

# 设置Hugging Face镜像和CUDA配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 加载数据集
dataset = load_dataset("json", data_files="hospital_dataset.json", split="train")
validation_dataset = load_dataset("json", data_files="hospital_test.json", split="train")
eval_dataset = load_dataset("json", data_files="hospital_test.json", split="train")


# Hugging Face模型ID
model_id = "openlm-research/open_llama_3b" 

# BitsAndBytesConfig配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="/home/kyds/finetune/hf_hub/hf_hub/models--openlm-research--open_llama_3b",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained("/home/kyds/finetune/hf_hub/hf_hub/models--openlm-research--open_llama_3b")
tokenizer.padding_side = 'right'

# 设置聊天模板
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA配置
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
)

# 训练参数
initial_learning_rate = 0.001
learning_rate_step = 0.001



# 初始化学习率
learning_rate = initial_learning_rate

while learning_rate <= 0.01:  # 直到达到最大学习率为止
    # 更新学习率参数
    args = TrainingArguments(
        output_dir="model",  # 保存模型的目录
        num_train_epochs=8,  # 训练轮数
        per_device_train_batch_size=2,  # 每个设备的批次大小
        gradient_accumulation_steps=2,  # 进行反向传播/更新步骤之前的步骤数
        gradient_checkpointing=True,  # 使用梯度检查点来节省内存
        optim="adamw_torch_fused",  # 使用融合的AdamW优化器
        logging_steps=10,  # 每10步记录一次日志
        save_strategy="epoch",  # 每个epoch保存一次检查点
        learning_rate=learning_rate,  # 学习率
        bf16=True,  # 使用bfloat16精度
        tf32=True,  # 使用tf32精度
        max_grad_norm=0.3,  # 最大梯度范数
        warmup_ratio=0.03,  # 热身比率
        lr_scheduler_type="constant",  # 使用恒定的学习率调度器
        push_to_hub=False,  # 不推送模型到hub
        report_to="tensorboard",  # 将指标报告给tensorboard
        eval_steps=10,  # 每100步评估一次
        evaluation_strategy="steps",  # 每个epoch评估一次
    )
    max_seq_length = 3072

    # 创建一个空的txt文档，用于记录微调过程中的日志
    log_file = open(f"fine_tuning_logs_{learning_rate}loss.txt", "w")

    # 创建新的trainer实例
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # 使用特殊标记
            "append_concat_token": False,  # 无需添加额外的分隔标记
        },
    )

    # 训练模型
    trainer.train()

    # 保存模型
    trainer.save_model()
    
    # 记录微调过程中的日志内容到txt文档
    log_file.write(f"Fine-tuning logs for learning rate: {learning_rate}\n")
    for log in trainer.state.log_history:
      log_file.write(str(log) + "\n")



    # 释放内存
    del trainer
    torch.cuda.empty_cache()
    

    peft_model_id = "./model"
    test_model = AutoPeftModelForCausalLM.from_pretrained(
	  peft_model_id,
	  device_map="cuda:0",
	  torch_dtype=torch.float16
    )
    pipe = pipeline("text-generation", model=test_model, tokenizer=tokenizer)


    success_rate = []
    number_of_eval_samples = 127
    # iterate over eval dataset and predict
    for s in tqdm(eval_dataset.shuffle().select(range(number_of_eval_samples))):
        success_rate.append(evaluate_position(s))

    # compute accuracy
    accuracy = sum(success_rate) / len(success_rate)

    log_file.write(f"Accuracy: {accuracy * 100:.2f}%")
    log_file.write("\n\n")


    del test_model
    torch.cuda.empty_cache()

    # 关闭日志文件
    log_file.close()

    file_path = f'fine_tuning_logs_{learning_rate}loss.txt'

    train_losses = []
    eval_losses = []
    epochs_train = []
    epochs_eval = []

    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Remove the whitespace and curly braces at the beginning and end of the line
                line = line.strip().strip('{}')
                # Splits rows into key-value pairs and converts the key-value pairs into dictionaries
                parts = line.split(', ')
                loss_info = {part.split(': ')[0].strip("'"): float(part.split(': ')[1]) for part in parts}
                
                # Assign data according to the key of the dictionary
                if 'loss' in loss_info and 'epoch' in loss_info:
                    train_losses.append(loss_info['loss'])
                    epochs_train.append(loss_info['epoch'])
                if 'eval_loss' in loss_info and 'epoch' in loss_info:
                    eval_losses.append(loss_info['eval_loss'])
                    epochs_eval.append(loss_info['epoch'])
            except Exception as e:
                print(f"An error occurred while parsing the line: {line}")
                print(f"Error: {e}")

    if train_losses and eval_losses:
        plt.figure(figsize=(10, 5), dpi=300)
        plt.plot(epochs_train, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=4)
        plt.plot(epochs_eval, eval_losses, 's-', label='Validation Loss', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(f"Training and Validation Loss for Learning Rate:{learning_rate}", fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"Fine-tuning_loss_for_learning_rate_{learning_rate}.png", format='png')
        # plt.show()  # Just for test.
    else:
        print("No data to plot.")

    # 增加学习率
    learning_rate += learning_rate_step

    gc.collect()


