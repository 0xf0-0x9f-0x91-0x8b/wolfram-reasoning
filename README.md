---------------------------------------
Begin Slurm Prolog: Dec-05-2025 13:15:54
Job ID:    3939512
User ID:   kpalaniappan8
Account:   ece
Job name:  my_h200_job
Partition: ice-gpu
QOS:       coe-ice
---------------------------------------
CUDA_VISIBLE_DEVICES=0
Running on host: atl1-1-03-017-2-0.pace.gatech.edu
GPUs allocated:
Fri Dec  5 13:15:54 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H200                    On  |   00000000:66:00.0 Off |                    0 |
| N/A   32C    P0             77W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
Imports: 51.9062 seconds
build_dataset: 1.6302 seconds
trainable params: 6,422,528 || all params: 2,133,954,560 || trainable%: 0.3010
load_model: 25.9001 seconds
Baseline format adherence (pre-RL):
Format adherence: 80.0% with 10 samples
evaluate_format_rate: 111.7911 seconds
Baseline: 8.0 matches
grpo_update: 75.5599 seconds
Step 01 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=75.6s
grpo_update: 72.7053 seconds
Step 02 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=148.3s
grpo_update: 70.8800 seconds
Step 03 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=219.2s
grpo_update: 70.7983 seconds
Step 04 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=290.0s
grpo_update: 131.6365 seconds
Step 05 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=421.7s
grpo_update: 71.1322 seconds
Step 06 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=492.8s
grpo_update: 70.8606 seconds
Step 07 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=563.7s
grpo_update: 69.9206 seconds
Step 08 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=633.6s
grpo_update: 72.0216 seconds
Step 09 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=705.7s
grpo_update: 71.1905 seconds
Step 10 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=776.9s
grpo_update: 71.0845 seconds
Step 11 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=848.0s
grpo_update: 71.0947 seconds
Step 12 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=919.1s
grpo_update: 71.3055 seconds
Step 13 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=990.4s
grpo_update: 70.9416 seconds
Step 14 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=1061.4s
grpo_update: 71.1633 seconds
Step 15 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=1132.5s
grpo_update: 71.3106 seconds
Step 16 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=1203.9s
grpo_update: 71.1474 seconds
Step 17 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=1275.0s
grpo_update: 70.8538 seconds
Step 18 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=1345.9s
grpo_update: 125.7720 seconds
Step 19 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=1471.7s
grpo_update: 72.6454 seconds
Step 20 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=1544.5s
grpo_update: 70.9133 seconds
Step 21 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=1615.6s
grpo_update: 71.9535 seconds
Step 22 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=1689.0s
grpo_update: 197.5506 seconds
Step 23 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=1887.2s
grpo_update: 71.1853 seconds
Step 24 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=1959.7s
grpo_update: 71.3415 seconds
Step 25 | reward_sum=0.000 | reward_mean=0.000 | format adherence=0.0% | time=2031.1s
Format adherence after RL:
Format adherence: 80.0% with 10 samples
evaluate_format_rate: 93.7511 seconds
Post-RL: 8.0 matches
Saved model weights to model.pt
---------------------------------------
Begin Slurm Epilog: Dec-05-2025 13:56:30
Job ID:        3939512
User ID:       kpalaniappan8
Account:       ece
Job name:      my_h200_job
Resources:     cpu=8,gres/gpu:h200=1,mem=64G,node=1
Rsrc Used:     cput=05:24:48,vmem=0,walltime=00:40:36,mem=5201904K,energy_used=0
Partition:     ice-gpu
QOS:           coe-ice
Nodes:         atl1-1-03-017-2-0
---------------------------------------
