#!/usr/bin/env python
# coding: utf-8

import os
from datetime import datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline, TrainerCallback, TrainerState, TrainerControl
from trl import setup_chat_format, SFTTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM
from tqdm import tqdm
import json

# 设置Hugging Face镜像和CUDA配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 加载数据集
dataset = load_dataset("json", data_files="hospital_dataset.json", split="train")
validation_dataset = load_dataset("json", data_files="hospital_test.json", split="train")
eval_dataset = validation_dataset

# Hugging Face模型ID
model_id = "/home/HuggingFace-Download-Accelerator/hf_hub/models--openlm-research--open_llama_3b" 

class AccuracyCallback(TrainerCallback):
    @staticmethod
    def evaluate_position(sample, pipe):
        for i in range(1, 6):
            try:
                prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
                outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
                generated_positions = json.loads(sample["messages"][2]["content"])["positions"]
                true_positions = json.loads(outputs[0]["generated_text"][len(prompt):].strip())["positions"]
                if generated_positions == true_positions:
                    return 1
                else:
                    return 0
            except json.JSONDecodeError:
                return 0

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        trainer.save_model()
        peft_model_id = "./model"
        test_model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
        pipe = pipeline("text-generation", model=test_model, tokenizer=tokenizer)
        
        success_rate = []
        eval_dataset = load_dataset("json", data_files="hospital_test.json", split="train")
        for sample in eval_dataset:
            success_rate.append(self.evaluate_position(sample, pipe))
        
        accuracy = sum(success_rate) / len(success_rate)
        log_file.write(f"在第 {state.epoch} 个周期后的准确率: {accuracy * 100:.2f}%")
        log_file.write("\n")

        del test_model
        torch.cuda.empty_cache()


# BitsAndBytesConfig配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
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
initial_learning_rate = 0.0001
learning_rate_step = 0.0001

learning_rate = initial_learning_rate

while learning_rate <= 0.001:  # 直到达到最大学习率为止
    num_epochs=25
    log_file = open(f"fine_tuning_logs_loss{learning_rate}.txt", "w")
    args = TrainingArguments(
        output_dir="model",  # 保存模型的目录
        num_train_epochs=num_epochs,  # 训练轮数
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
        evaluation_strategy="no",  # 每个epoch评估一次
    )
    max_seq_length = 3072

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
        callbacks=[AccuracyCallback()]
    )

    # 训练模型
    trainer.train()

    # 记录微调过程中的日志内容到txt文档
    log_file.write(f"Fine-tuning logs for learning rate: {learning_rate}")
    for log in trainer.state.log_history:
        log_file.write(str(log) + "\n")

    log_file.close()

    # 增加学习率
    learning_rate += learning_rate_step
print("complete")
