#!/usr/bin/env bash
set -euo pipefail

model_names='max_mil mean_mil att_mil trans_mil s4model mamba_mil'
backbone="resnet50"
encoder="UNI"
declare -A in_dim
in_dim["resnet50"]=1024   # conch 768, UNI 1024

declare -A gpus
gpus["mean_mil"]=4
gpus["max_mil"]=4
gpus["att_mil"]=4
gpus["trans_mil"]=4 
gpus["s4model"]=4      # 修正键名
gpus["mamba_mil"]=4

lr='2e-4'
mambamil_rate='5'
mambamil_layer='2'
mambamil_type='SRMamba'

task="TCGA-BRCA_survival"
data_root_dir="/home/baizhiwang/Badge4/Uncertainty/dataset/TCGA-BRCA/${encoder}"
results_dir="/home/baizhiwang/Badge4/Uncertainty/experiments/result/${task}/${encoder}"
preloading="no"
patch_size="256"

mkdir -p "${results_dir}"
log_dir="/home/baizhiwang/Badge4/Uncertainty/experiments/log"
mkdir -p "${log_dir}"

k_start=-1
k_end=-1

pids=()

for model in ${model_names}; do
  exp="${model}/${backbone}"
  gpu="${gpus[$model]:-}"
  [[ -z "${gpu}" ]] && { echo "⚠️ 未为 ${model} 指定 GPU，跳过"; continue; }
  log="${log_dir}/train_${task}_${model}.log"

  echo "${exp}, GPU is: ${gpu}"

  # 每个训练作为独立后台任务，并各自输出到独立日志
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python /home/baizhiwang/Badge4/Uncertainty/UAFRE-Github/MIL/main_survival.py \
      --drop_out 0.25 \
      --early_stopping \
      --lr "${lr}" \
      --k 5 \
      --k_start "${k_start}" \
      --k_end "${k_end}" \
      --label_frac 1.0 \
      --exp_code "${exp}" \
      --patch_size "${patch_size}" \
      --batch_size 1 \
      --weighted_sample \
      --bag_loss nll_surv \
      --task "${task}" \
      --backbone "${backbone}" \
      --results_dir "${results_dir}" \
      --model_type "${model}" \
      --log_data \
      --split_dir "/home/baizhiwang/Badge/abMIL/CLAM/Badge/TCGA-BRCA-result/split/BRCA_subtyping_100" \
      --data_root_dir "${data_root_dir}" \
      --preloading "${preloading}" \
      --in_dim "${in_dim[$backbone]}" \
      --k_fold True \
      --mambamil_rate "${mambamil_rate}" \
      --mambamil_layer "${mambamil_layer}" \
      --mambamil_type "${mambamil_type}"
  ) > "${log}" 2>&1 &

  pids+=($!)   # 记录该后台任务 PID
done

# 等所有后台任务结束
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "✅ 全部并行训练完成。日志在：${log_dir}"
