import math
import re
import random
import time
from collections import deque
from typing import List

import numpy as np
import torch

from utils import time_it
from data import build_dataset
from prompts import build_prompt
from model import load_model, sample_group

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def completion_log_probs(model, tokenizer, sequences, prompt_lengths):
    attention_mask = (sequences != tokenizer.tokenizer.pad_token_id).long()
    with torch.cuda.amp.autocast():
        outputs = model(input_ids=sequences, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = sequences[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    cont_logps = []
    for i in range(sequences.size(0)):
        prompt_len = prompt_lengths[i].item()
        total_len = attention_mask[i].sum().item()
        start = max(prompt_len - 1, 0)
        end = max(total_len - 1, start)
        cont_logps.append(token_logps[i, start:end].sum())
    return torch.stack(cont_logps)


def compute_group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    num_rewards = rewards.numel()
    reshaped = rewards.view(-1, group_size)
    mean = reshaped.mean(dim=1, keepdim=True)
    std = reshaped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = (reshaped - mean) / std
    return normalized.view(-1)


def grpo_sequence_loss(logp_new: torch.Tensor, logp_ref: torch.Tensor, advantages: torch.Tensor, beta: float, epsilon: float) -> torch.Tensor:
    ratio = torch.exp(logp_new - logp_ref)
    ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surrogate = torch.min(ratio * advantages, ratio_clipped * advantages)
    kl_div = torch.exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1
    loss = -(surrogate - beta * kl_div).mean()
    return loss


def grpo_update(batch_prompts, model, tokenizer, optimizer, group_size, beta, epsilon, grad_accum=None):
    accum_steps = grad_accum if grad_accum is not None else 1
    if accum_steps < 1:
        raise ValueError("grad_accum must be >= 1")

    optimizer.zero_grad()
    all_rewards: list[torch.Tensor] = []
    all_losses: list[torch.Tensor] = []
    collected_texts: list[str] = []
    
    for _ in range(accum_steps):
        sequences, prompt_lens, rewards, texts = sample_group(model, tokenizer, batch_prompts, num_generations=group_size, do_sample=True)

        logp_new = completion_log_probs(model, tokenizer, sequences, prompt_lens)

        adapters_supported = hasattr(model, "disable_adapter") and hasattr(model, "enable_adapter")
        if adapters_supported:
            model.disable_adapter()
        with torch.no_grad():
            logp_ref = completion_log_probs(model, tokenizer, sequences, prompt_lens)
        if adapters_supported:
            model.enable_adapter()

        advantages = compute_group_advantages(rewards, group_size)
        base_loss = grpo_sequence_loss(logp_new, logp_ref, advantages, beta=beta, epsilon=epsilon)
        (base_loss / accum_steps).backward()

        all_losses.append(base_loss.detach())
        all_rewards.append(rewards.detach())
        collected_texts.extend(texts)
        breakpoint()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    reward_tensor = torch.cat(all_rewards)
    reward_mean = reward_tensor.mean().item()
    reward_sum = reward_tensor.sum().item()
    loss_value = torch.stack(all_losses).mean().item()

    return {
        "loss": loss_value,
        "reward_mean": reward_mean,
        "reward_sum": reward_sum,
        "format_match_pct": reward_mean * 100.0,
        "texts": collected_texts,
        "optimizer_step": True,
        "accumulated_microbatches": accum_steps,
        "grad_accum_steps": accum_steps,
    }


def evaluate_format_rate(model, tokenizer, dataset):
    prompts = build_prompt(tokenizer, [(ex["question"], ex["image"]) for ex in dataset])
    _, _, rewards, texts = sample_group(model, tokenizer, prompts, num_generations=1, do_sample=True)
    total = rewards.numel()
    matches = rewards.sum().item()
    rate = (matches / total) * 100
    print(f"Format adherence: {rate:.1f}% with {total} samples")
    return rate, matches, texts


def main():
    train_ds, test_ds = time_it("build_dataset", build_dataset)
    test_ds = test_ds.select(range(10))
    model, tokenizer = time_it("load_model", load_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    beta = 0.05
    epsilon = 0.2
    group_size = 5

    history = deque(maxlen=10)
    print("Baseline format adherence (pre-RL):")
    baseline_rate, baseline_matches, _ = time_it("evaluate_format_rate", evaluate_format_rate, model, tokenizer, test_ds)
    print(f"Baseline: {baseline_matches} matches")

    start_time = time.time()
    for step in range(25):
        batch = train_ds.shuffle(seed=step).select(range(1))  # batch size n
        prompts_batch = build_prompt(tokenizer, [(ex["question"], ex["image"]) for ex in batch])
        answers = [ex["answer"] for ex in batch]
        stats = time_it("grpo_update", grpo_update, prompts_batch, model, tokenizer, optimizer, group_size, beta, epsilon, grad_accum=1)  # simulate batch size nx8
        history.append(stats["reward_mean"])
        print(
            f"Step {step + 1:02d} | reward_sum={stats['reward_sum']:.3f} | reward_mean={stats['reward_mean']:.3f} | format adherence={stats['format_match_pct']:.1f}% | time={(time.time() - start_time):.1f}s"
        )

    print("Format adherence after RL:")
    post_rate, post_matches, _ = time_it("evaluate_format_rate", evaluate_format_rate, model, tokenizer, test_ds)
    print(f"Post-RL: {post_matches} matches")

    save_path = "model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model weights to {save_path}")


if __name__ == "__main__":
    main()