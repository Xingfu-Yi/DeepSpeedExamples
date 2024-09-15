from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import gradio as gr
import random
import time

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

Global_Messages = []


def add_message(history, message, system_prompt):
    if system_prompt.strip() and len(history) == 0:
        Global_Messages.append({"role": "system", "content": system_prompt})

    if message["text"] is not None:
        history.append((message["text"], None))
        Global_Messages.append({"role": "user", "content": message["text"]})

    return (
        history,
        gr.MultimodalTextbox(value=None, interactive=False),
        gr.update(interactive=False),
    )


def clear_all():
    global Global_Messages
    Global_Messages = []
    return gr.update(interactive=True)


def bot(history):
    input_ids = tokenizer.apply_chat_template(
        Global_Messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1] :]
    response_txt = tokenizer.decode(response, skip_special_tokens=True)
    history[-1][1] = response_txt

    Global_Messages.append({"role": "assistant", "content": response_txt})

    all = outputs[0]
    all_txt = tokenizer.decode(all, skip_special_tokens=False)

    return history, all_txt
    # bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    # history[-1][1] = ""
    # for character in bot_message*10:
    #     history[-1][1] += character
    #     time.sleep(0.05)
    #     yield history


with gr.Blocks(fill_height=True) as demo:
    system_prompt = gr.Textbox(
        label="System Prompt",
        interactive=True,
    )
    with gr.Row():
        text_raw = gr.TextArea(
            label="Raw History",
            interactive=False,
            visible=False,
            lines=27,
            show_copy_button=True,
        )
        chatbot = gr.Chatbot(
            label="LittleChat",
            elem_id="chatbot",
            bubble_full_width=False,
            scale=1,
            height=600,
        )
    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
    )
    with gr.Row():
        show_raw = gr.Checkbox(label="Show Raw History", value=False)
        clear_btn = gr.ClearButton([text_raw, chatbot, chat_input])

    chat_msg = chat_input.submit(
        add_message,
        [chatbot, chat_input, system_prompt],
        [chatbot, chat_input, system_prompt],
    )
    bot_msg = chat_msg.then(bot, chatbot, [chatbot, text_raw], api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    show_raw.change(lambda show_raw: gr.update(visible=show_raw), show_raw, text_raw)
    clear_btn.click(clear_all, None, [system_prompt])

if __name__ == "__main__":
    demo.launch()
