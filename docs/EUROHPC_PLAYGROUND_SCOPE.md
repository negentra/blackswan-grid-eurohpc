# EuroHPC Playground scope

The proposed calibration target is Leonardo Booster with NVIDIA A100 64 GB GPUs. That target is a plan; **no A100 benchmark has been measured**. The [14-row matrix](../results/playground_calibration_matrix_v2.csv) varies graph size, temporal length, batch and FP32/BF16 where technically acceptable. It is not a scientific training or safety-evaluation programme. The complete matrix has not been executed.

The local RTX 3080 small CUDA smoke establishes only finite forward/loss/gradients/inference and memory at batch 1, two time steps for S1/S2/S3. Target-GPU timing, memory limits, placement, billing, account, partition and software stack need host assessment and measurement. The Slurm files are unsubmitted examples. A later Fast Lane request remains conditional on measured A100 workload scaling.
