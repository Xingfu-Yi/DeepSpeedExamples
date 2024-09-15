from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
import torch

model_name = "../training/step1_supervised_finetuning/output_step1_llama3_8b_lora"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

output_path = "../training/step1_supervised_finetuning/output_step1_llama3_8b_lora/model.safetensors"

state_dict = model.state_dict()

save_file(state_dict, output_path)
