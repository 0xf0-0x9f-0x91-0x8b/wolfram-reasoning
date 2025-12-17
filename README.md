# Incentivizing Wolfram-based reasoning in VLMs via reinforcement learning
I'm experimenting with a GRPO-based RL algorithm to incentivize representation and thinking of visual math concepts, by the Qwen3-VL-2B-Thinking model, in Wolfram language. 

## Thinking in Wolfram
<img width="1170" height="297" alt="image" src="https://github.com/user-attachments/assets/1ef96c5a-0be7-4d95-8601-b358e38c60dd" />

**Figure**: Thinking math concepts with Wolfram Language and Engine.

## Overview of training methodology
<img width="615" height="551" alt="image" src="https://github.com/user-attachments/assets/10d153b7-4015-4422-ad0a-86a70dbb29ce" />

**Figure**: For each prompt, G output sequences from the model are explored, with advantages calculated amongst these outputs based on the reward model. LoRA (injected into each attention layer in the base model) is used for GRPO updates.

## Training script
https://github.com/karthikpalaniappan7/wolfram-reasoning/blob/main/src/train.py

## Dataset
A subset of [🤗 ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) introduced in [VL-Rethinker](https://neurips.cc/virtual/2025/loc/san-diego/poster/119918) paper (see results).

## Latest results (WIP)
| Category | Count | Has Code (%) | No Error (%) | Accuracy (%) | Prompt Token Length | Output Token Length |
| :--- | ---:  | ---:  | ---:  | ---:  | :--- | :--- |
| (GradeSchool) Geometric | 16 | 25.0 | 12.5 | 6.25 | 718.38 ± 166.99 | 3628.56 ± 972.42 |
| (GradeSchool) Non-Geo Math | 12 | 33.33 | 25.0 | 8.33 | 812.00 ± 371.30 | 3380.92 ± 1083.33 |
| (GradeSchool) Science | 1 | 0.0 | 0.0 | 0.0 | N/A | N/A |
| Broader STEM Topics | 2 | 50.0 | 0.0 | 0.0 | 993.00 ± 158.39 | 3339.50 ± 1069.85 |
| Social Science | 2 | 50.0 | 0.0 | 0.0 | 923.00 ± 209.30 | 2543.00 ± 2196.27 |
| Spatial Reasoning | 5 | 40.0 | 40.0 | 20.0 | 976.60 ± 319.49 | 3953.00 ± 212.99 |
| Tables/Diagrams/Charts | 5 | 20.0 | 20.0 | 0.0 | 1277.40 ± 1266.48 | 3561.20 ± 881.92 |
| Overall | 43 | 30.23 | 18.6 | 6.98 | 863.23 ± 491.96 | 3536.28 ± 976.09 |

**Table**: Evaluation results with in-context learning and GRPO-based exploration and fine-tuning. Code column denotes fraction of outputs with Wolfram code. No Error column denotes fraction of
outputs with error-free Wolfram code. Accuracy column denotes fraction of correct answers after deduction from Wolfram Engine. The prompt and
output token lengths are averaged category-wise, with mean and std. dev. mentioned. 

