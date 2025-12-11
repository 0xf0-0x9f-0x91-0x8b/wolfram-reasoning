import time

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator, dispatch_model

from utils import time_it
from prompts import build_prompt, format_reward

MODEL_NAME = "Qwen/Qwen3-VL-2B-Thinking"


@time_it
def load_model(grad_accum_steps: int = 4):
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_steps
    )
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoProcessor.from_pretrained(MODEL_NAME, padding_side="left")
    if tokenizer.tokenizer.pad_token_id is None:
        tokenizer.tokenizer.pad_token_id = tokenizer.tokenizer.eos_token_id

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.set_attn_implementation("eager")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    model = accelerator.prepare(model)

    return accelerator, model, tokenizer


def get_model_kwargs(tokenizer, do_sample, num_generations, temperature, output_scores):
    gen_kwargs = {
        "max_new_tokens": 4096,
        "do_sample": do_sample,
        "num_return_sequences": num_generations,
        "pad_token_id": tokenizer.tokenizer.pad_token_id,
        "eos_token_id": tokenizer.tokenizer.eos_token_id,
        "temperature": temperature,
        "return_dict_in_generate": True,
        "output_scores": output_scores,
        "output_attentions": True
    }
    return gen_kwargs


@time_it
@torch.no_grad()
def predict(accelerator, model, tokenizer, inputs, return_text=True):
    num_generations = 1
    gen_kwargs = get_model_kwargs(tokenizer, True, num_generations, 1.0, True)
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, **gen_kwargs)
    if not return_text:
        return outputs

    generated_ids = outputs.sequences
    B = len(inputs["input_ids"])
    generated_ids_trimmed = [
        generated_ids[i][len(inputs["input_ids"][i // num_generations]):]
        for i in range(B * num_generations)
    ]

    output_text = tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text

@time_it
@torch.no_grad()
def sample_group(accelerator, model, tokenizer, inputs, num_generations=4, do_sample=True, temperature=1, ret_sequences=False):
    gen_kwargs = get_model_kwargs(tokenizer, do_sample, num_generations, temperature, False)
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, **gen_kwargs)
    sequences = outputs.sequences

    batch_size = inputs["input_ids"].shape[0]
    max_input_len = inputs["input_ids"].shape[1]
    prompt_lengths = inputs["attention_mask"].sum(dim=1)
    prompt_len_tensor = prompt_lengths.repeat_interleave(num_generations)

    sequences = sequences.view(batch_size, num_generations, -1)
    sequence_tokens = []
    output_lengths = []
    for i in range(batch_size):
        for g in range(num_generations):
            gen_tokens = sequences[i, g][max_input_len:]
            gen_tokens = gen_tokens[gen_tokens != tokenizer.tokenizer.pad_token_id]
            sequence_tokens.append(gen_tokens)
            output_lengths.append(len(gen_tokens))
    output_lengths = torch.tensor(output_lengths, device=accelerator.device)

    texts = tokenizer.batch_decode(
        sequence_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    rewards = format_reward(texts, accelerator)
    if ret_sequences:
        return sequences, output_lengths, prompt_len_tensor, rewards, texts
    return output_lengths, prompt_len_tensor, rewards, texts


if __name__ == "__main__":
    accelerator, model, tokenizer = load_model()
    question = "<image>\nWhat fraction of the shapes are squares?\nChoices:\n(A) 5/10\n(B) 3/7\n(C) 3/9\n(D) 5/9"
    images = ['images/Processed-5cb29a4f-240d-4f1f-a063-b7fd922ee9e9-0.jpg']
    inputs = build_prompt(tokenizer, [(question, images)], accelerator)
    output_lengths, prompt_lengths, _, texts = sample_group(accelerator, model, tokenizer, inputs, num_generations=1)
