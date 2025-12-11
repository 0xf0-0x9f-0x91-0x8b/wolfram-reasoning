import argparse
import random
import time
from collections import deque
from tqdm.auto import tqdm

import numpy as np
import torch
from datasets import Dataset

from utils import time_it, gpu_memory, write_csv_header, write_csv_rows
from data import build_dataset
from prompts import build_prompt
from model import load_model, sample_group

seed = 44
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

@time_it
def completion_log_probs(model, tokenizer, sequences, prompt_lengths, accelerator):
    """
    Compute log probabilities of the continuation part of sequences.

    sequences: (B, T)
    prompt_lengths: (B,) number of tokens in each prompt
    """
    # Ensure tensors are on the correct device
    sequences = sequences.to(accelerator.device)
    prompt_lengths = prompt_lengths.to(accelerator.device)

    attention_mask = (sequences != tokenizer.tokenizer.pad_token_id).long()
    target_ids = sequences[:, 1:]

    outputs = model(input_ids=sequences, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logps = torch.gather(
        log_probs, dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)

    cont_logps = []
    for i in range(sequences.size(0)):
        prompt_len = prompt_lengths[i].item()
        total_len = attention_mask[i].sum().item()
        start = max(prompt_len - 1, 0)
        end = max(total_len - 1, start)
        cont_logps.append(token_logps[i, start:end].sum())

    return torch.stack(cont_logps)


@time_it
def compute_group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    reshaped = rewards.view(-1, group_size)
    mean = reshaped.mean(dim=1, keepdim=True)
    std = reshaped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = (reshaped - mean) / std
    return normalized.view(-1)


@time_it
def grpo_sequence_loss(logp_new: torch.Tensor, logp_ref: torch.Tensor, advantages: torch.Tensor, beta: float, epsilon: float) -> torch.Tensor:
    ratio = torch.exp(logp_new - logp_ref)
    ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surrogate = torch.min(ratio * advantages, ratio_clipped * advantages)
    kl_div = torch.exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1
    loss = -(surrogate - beta * kl_div).mean()
    return loss


def grpo_update(
    accelerator,
    batch_prompts,
    model,
    tokenizer,
    optimizer,
    group_size,
    beta,
    epsilon,
):
    all_rewards: list[torch.Tensor] = []
    all_losses: list[torch.Tensor] = []
    collected_texts: list[str] = []

    # This block participates in gradient accumulation
    with accelerator.accumulate(model):
        # 1) Sample group (RL rollouts)
        sequences, output_lens, prompt_lens, rewards, texts = sample_group(
            accelerator,
            model,
            tokenizer,
            batch_prompts,
            num_generations=group_size,
            do_sample=True,
        )

        # 2) New policy log-probs
        logp_new = completion_log_probs(
            model, tokenizer, sequences, prompt_lens, accelerator
        )

        # 3) Reference policy log-probs (adapters off)
        adapters_supported = hasattr(model, "disable_adapter") and hasattr(
            model, "enable_adapter"
        )
        if adapters_supported:
            model.disable_adapter()

        with torch.no_grad():
            logp_ref = completion_log_probs(
                model, tokenizer, sequences, prompt_lens, accelerator
            )

        if adapters_supported:
            model.enable_adapter()

        # 4) Advantages and GRPO loss
        advantages = compute_group_advantages(rewards, group_size)
        base_loss = grpo_sequence_loss(
            logp_new, logp_ref, advantages, beta=beta, epsilon=epsilon
        )

        # 5) Backward with Accelerate (no manual /accum_steps)
        accelerator.backward(base_loss)

        # 6) Clip and step — Accelerate will only *actually* step
        #    when gradient_accumulation_steps boundary is reached.
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # 7) Collect stats (Python-only, no effect on grads)
        all_losses.append(base_loss.detach())
        all_rewards.append(rewards.detach())
        collected_texts.extend(texts)

    reward_tensor = torch.cat(all_rewards)
    reward_mean = reward_tensor.mean().item()
    reward_sum = reward_tensor.sum().item()
    loss_value = torch.stack(all_losses).mean().item()

    return {
        "loss": loss_value,
        "reward_mean": reward_mean,
        "reward_sum": reward_sum,
        "texts": collected_texts,
        "optimizer_step": accelerator.sync_gradients,  # True when a real step happened
        "grad_accum_steps": accelerator.gradient_accumulation_steps,
    }


def evaluate_format_rate(
    accelerator, model, tokenizer, dataset,
    batch_size=1,
    coding=False,
    csv_path="results.csv"
):
    write_csv_header(csv_path)

    total_matches = 0
    total_samples = 0

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    # tqdm iteration
    for start in tqdm(
        range(43, len(dataset), batch_size),
        desc="Evaluating",
        total=num_batches
    ):
        end = start + batch_size
        batch = dataset.select(range(start, min(end, len(dataset))))

        # build prompts
        prompts = build_prompt(
            tokenizer,
            [(ex["question"], ex["image"]) for ex in batch],
            accelerator,
            coding=coding
        )

        # model inference
        output_lengths, prompt_lengths, rewards, texts = sample_group(
            accelerator, model, tokenizer, prompts,
            num_generations=1,
            do_sample=True,
            temperature=1
        )

        batch_matches = rewards.sum().item()
        batch_total = rewards.numel()

        total_matches += batch_matches
        total_samples += batch_total

        write_csv_rows(batch, texts, output_lengths, prompt_lengths, csv_path)

    # final summary
    rate = (total_matches / total_samples) * 100
    print(f"\nFormat adherence: {rate:.1f}% with {total_samples} samples")

    return rate, total_matches


def main(coding):
    train_ds, test_ds = build_dataset(seed=seed)

    grad_accum_steps = 1
    accelerator, model, tokenizer = load_model(grad_accum_steps=grad_accum_steps)
    if accelerator.is_main_process:
        gpu_memory()

    # Create optimizer and prepare it with accelerator if needed
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer = accelerator.prepare(optimizer)

    beta = 0.05
    epsilon = 0.2
    group_size = 10

    history = deque(maxlen=10)

    reeval = False
    if reeval:
        if accelerator.is_main_process:
            print("Baseline format adherence (pre-RL):")
        csv_path = "results_answers_iter1.csv" if not coding else "results_iter2.csv"
        baseline_rate, baseline_matches = evaluate_format_rate(accelerator, model, tokenizer, test_ds, csv_path=csv_path, coding=coding)

        if accelerator.is_main_process:
            print(f"Baseline: {baseline_matches} matches")
            gpu_memory()

    start_time = time.time()
    batches = []
    collected_texts = []

    for step in tqdm(range(10)):
        batch = train_ds.shuffle(seed=step + seed).select(range(1))
        batches.append(batch)

        prompts_batch = build_prompt(
            tokenizer,
            [(ex["question"], ex["image"]) for ex in batch],
            accelerator,
        )
        answers = [ex["answer"] for ex in batch]

        stats = grpo_update(
            accelerator,
            prompts_batch,
            model,
            tokenizer,
            optimizer,
            group_size,
            beta,
            epsilon,
        )
        
        if accelerator.is_main_process:
            gpu_memory()
            history.append(stats["reward_mean"])
            collected_texts.extend(stats["texts"])
            print(
                f"Step {step + 1:02d} | "
                f"reward_sum={stats['reward_sum']:.3f} | "
                f"reward_mean={stats['reward_mean']:.3f} | "
                f"time={(time.time() - start_time):.1f}s"
            )

    if accelerator.is_main_process:
        batch_ds = Dataset.from_list(
            [
                dict(
                    question=ex["question"],
                    image=ex["image"],
                    answer=ex["answer"],
                    batch_id=i,
                )
                for i, batch in enumerate(batches)
                for ex in batch
            ]
        )
        batch_ds.save_to_disk("./batch_ds")

        with open("./batch_ds_output.txt", "w") as f:
            f.write("\n\n".join(collected_texts))

        print("Format adherence after RL:")
        post_rate, post_matches = evaluate_format_rate(accelerator, model, tokenizer, test_ds)
        print(f"Post-RL: {post_matches} matches")
        gpu_memory()

        # Save only from main process and unwrap the model
        save_path = "model.pt"
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), save_path)
        print(f"Saved model weights to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--coding", action="store_true", default=False, help="Enable coding mode")

    args = parser.parse_args()
    print('Coding:', args.coding)
    main(args.coding)
