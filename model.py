import time

start = time.perf_counter()
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from prompts import build_prompt, format_reward
from utils import time_it
end = time.perf_counter()
print(f"Imports: {end - start:.4f} seconds")

MODEL_NAME = "Qwen/Qwen3-VL-2B-Thinking"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """
    Load fine-tunable model with LoRA config
    """
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
        device_map="auto",
        quantization_config=bnb_config,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()
    return model, tokenizer


def predict(model, tokenizer, inputs):
    N = 10
    gen_kwargs = {
        "max_new_tokens": 4096,
        "do_sample": True,
        "num_return_sequences": N,
        "pad_token_id": tokenizer.tokenizer.pad_token_id,
        "eos_token_id": tokenizer.tokenizer.eos_token_id,
        "temperature": 0.9,
        "return_dict_in_generate": True,
        "output_scores": True
    }
    with torch.cuda.amp.autocast():
        outputs = model.generate(**inputs, **gen_kwargs)
    generated_ids = outputs.sequences

    B = len(inputs.input_ids)
    generated_ids_trimmed = [
        generated_ids[i][len(inputs.input_ids[i//N]):]
        for i in range(B * N)
    ]

    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


@torch.no_grad()
def sample_group(model, tokenizer, inputs, num_generations=4, do_sample=True, temperature=0.9):
    gen_kwargs = {
        "max_new_tokens": 1024,
        "do_sample": do_sample,
        "num_return_sequences": num_generations,
        "pad_token_id": tokenizer.tokenizer.pad_token_id,
        "eos_token_id": tokenizer.tokenizer.eos_token_id,
        "temperature": temperature,
        "return_dict_in_generate": True,
        "output_scores": False
    }
    with torch.cuda.amp.autocast():
        outputs = model.generate(**inputs, **gen_kwargs)

    sequences = outputs.sequences    
    batch_size = inputs.input_ids.shape[0]
    prompt_lengths = inputs.attention_mask.sum(dim=1)
    prompt_len_tensor = prompt_lengths.repeat_interleave(num_generations).to(device)

    sequences = sequences.view(batch_size, num_generations, -1)
    sequence_tokens = []
    for i in range(batch_size):
        prompt_len = prompt_lengths[i].item()
        for g in range(num_generations):
            ids = sequences[i, g]
            nonpad = (ids != tokenizer.tokenizer.pad_token_id).long()
            seq_len = nonpad.sum().item()
            gen_tokens = ids[prompt_len:seq_len]
            sequence_tokens.append(gen_tokens)
    texts = tokenizer.batch_decode(sequence_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    rewards = format_reward(texts)
    return outputs.sequences, prompt_len_tensor, rewards, texts


if __name__ == "__main__":
    question = "<image>\nWhat fraction of the shapes are squares?\nChoices:\n(A) 5/10\n(B) 3/7\n(C) 3/9\n(D) 5/9"
    images = ['images/Processed-5cb29a4f-240d-4f1f-a063-b7fd922ee9e9-0.jpg']
    model, tokenizer = md.time_it("load_model", md.load_model)
    inputs = md.time_it("build_prompt", md.build_prompt, tokenizer, [(question, images), (question, images)])
    ret = md.time_it("sample_group", md.sample_group, model, tokenizer, inputs, num_generations=10)
    # outputs = md.time_it("predict", md.predict, model, tokenizer, inputs)
        