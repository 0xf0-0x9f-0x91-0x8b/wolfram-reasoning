# Incentivizing Wolfram-based reasoning in VLMs via reinforcement learning
I'm experimenting with a GRPO-based RL algorithm to incentivize representation and thinking of visual math concepts, by the Qwen3-VL-2B-Instruct model, in Wolfram language. 

## Thinking in Wolfram
<img width="1170" height="297" alt="image" src="https://github.com/user-attachments/assets/1ef96c5a-0be7-4d95-8601-b358e38c60dd" />

**Figure**: Thinking math concepts with Wolfram Language and Engine.

## Overview of training methodology
<img width="615" height="551" alt="image" src="https://github.com/user-attachments/assets/10d153b7-4015-4422-ad0a-86a70dbb29ce" />

**Figure**: For each prompt, G = 10 output sequences from the model are explored, with advantages calculated amongst these outputs based on the reward model. LoRA (injected into each attention layer in the base model) is used for GRPO updates.

## Training script
https://github.com/karthikpalaniappan7/wolfram-reasoning/blob/main/src/train.py

## Dataset
A subset of [🤗 ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) introduced in [VL-Rethinker](https://neurips.cc/virtual/2025/loc/san-diego/poster/119918) paper (see results).

## Latest results (WIP)
<img width="804" height="233" alt="image" src="https://github.com/user-attachments/assets/91d8ab2a-1916-4b7f-a458-bfc8226a3bb9" />

**Table**: Evaluation results with in-context learning and GRPO-based exploration and fine-tuning. Code column denotes fraction of outputs with Wolfram code. No Error column denotes fraction of
outputs with error-free Wolfram code. Accuracy column denotes fraction of correct answers after deduction from Wolfram Engine. The prompt and
output token lengths are averaged category-wise, with mean and std. dev. mentioned. 

Although fraction of error-free code is high, further optimization is essential (increasing G, batch size, number of epochs, etc.) to increase accuracy. 
