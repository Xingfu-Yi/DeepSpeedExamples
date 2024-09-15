import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import torch
import os

model_path = "../training/step1_supervised_finetuning/output_step1_llama3_8b_lora"
save_path = "../training/step1_supervised_finetuning/output_step1_llama3_8b_lora_int4"
device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.uint8,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
    #llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],
    llm_int8_threshold=6.0,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    quantization_config=quantization_config,
    trust_remote_code=True,
)

os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)