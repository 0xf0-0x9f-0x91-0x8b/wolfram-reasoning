import torch
from train import format_reward, compute_group_advantages, device, grpo_sequence_loss

def test_format_reward():
    valid_resp = "<think>\nreasoning goes here\n</think>\n<answer>\nthe meaning of life is\n</answer>"
    invalid_resp = "<think>missing closing tags"
    scores = format_reward([valid_resp, invalid_resp])
    print('format_reward scores:', scores.cpu().tolist())
    assert scores[0].item() == 1.0
    assert scores[1].item() == 0.0

def test_tensor_close(name, actual, expected, atol=1e-6):
    diff = torch.max(torch.abs(actual - expected)).item()
    if diff > atol:
        raise AssertionError(f"{name} mismatch (max |Δ|={diff:.3e}). Expected {expected}, got {actual}")
    print(f"[OK] {name}: max diff={diff:.2e}")

def test_group_advantages():
    raw_rewards = torch.tensor([1.0, 0.0, 2.0, 4.0], device=device)
    adv = compute_group_advantages(raw_rewards, group_size=2)
    reshaped_adv = adv.view(2, 2)
    test_tensor_close("Group advantage mean", reshaped_adv.mean(dim=1), torch.zeros(2, device=device))
    test_tensor_close("Group advantage std", reshaped_adv.std(dim=1, unbiased=False), torch.ones(2, device=device))

def test_grpo_loss():
    logp_new = torch.log(torch.tensor([0.6, 0.4, 0.7, 0.3], device=device))
    logp_ref = torch.log(torch.tensor([0.5, 0.5, 0.6, 0.4], device=device))
    manual_adv = torch.tensor([1.2, -0.4, 0.5, -1.0], device=device)
    loss = grpo_sequence_loss(logp_new, logp_ref, manual_adv, beta=0.1, epsilon=0.2)
    test_tensor_close("GRPO loss test", loss, -0.2233469)

    logp_new_grad = torch.log(torch.tensor([0.55, 0.45, 0.65, 0.35], device=device))
    logp_new_grad.requires_grad_(True)
    loss_grad = grpo_sequence_loss(logp_new_grad, logp_ref, manual_adv, beta=0.05, epsilon=0.2)
    loss_grad.backward()
    if not torch.all(torch.isfinite(logp_new_grad.grad)):
        raise AssertionError("Non-finite gradients detected in GRPO loss")
    print(f"[OK] GRPO loss backward pass: grad norm={logp_new_grad.grad.norm().item():.4f}")

