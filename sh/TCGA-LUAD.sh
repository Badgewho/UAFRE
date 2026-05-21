model_names='max_mil mean_mil att_mil trans_mil s4model mamba_mil'
backbones="resnet50"

declare -A in_dim
in_dim["resnet50"]=768

declare -A gpus
gpus["mean_mil"]=0
gpus["max_mil"]=0
gpus["att_mil"]=1
gpus["trans_mil"]=1
gpus["s4model"]=4
gpus["mamba_mil"]=5

task="LUAD_subtyping"

model_size="small"
preloading="no"
patch_size="256"
dataset=LUAD
encoder=CONCH
lr='2e-4'
mambamil_rate='5'
mambamil_layer='2'
mambamil_type='SRMamba'

# 多个label（空格分隔）
labels="EGFR KRAS LRP1B prior_malignancy STK11 TP53"

log_dir="/home/baizhiwang/Badge4/Uncertainty/experiments/log"
mkdir -p "$log_dir"

for label in $labels
do
    echo "=============================="
    echo "🚀 Start label: $label"
    echo "=============================="

    results_dir=/home/baizhiwang/Badge4/Uncertainty/experiments/result/"$dataset"/"$encoder"/"$label"

    # 可选：限制同时跑的最大任务数（防止你以后加模型/加backbone）
    # max_jobs=6

    for model in $model_names
    do
        for backbone in $backbones
        do
            exp=$model"/"$backbone
            gpu=${gpus[$model]}
            echo "$exp, GPU is: $gpu"

            k_start=-1
            k_end=-1

            # ✅ 每个model后台并行；用 subshell 保证 CUDA_VISIBLE_DEVICES 不互相污染
            (
                export CUDA_VISIBLE_DEVICES=$gpu
                python /home/baizhiwang/Badge4/Uncertainty/UAFRE-Github/MIL/main.py\
                    --drop_out 0 \
                    --early_stopping \
                    --lr $lr \
                    --k 5 \
                    --k_start $k_start \
                    --k_end $k_end \
                    --label_frac 1.0 \
                    --max_epochs 50 \
                    --exp_code $exp \
                    --patch_size $patch_size \
                    --weighted_sample \
                    --task $task \
                    --backbone $backbone \
                    --results_dir $results_dir \
                    --model_type $model \
                    --log_data \
                    --label_col $label \
                    --split_dir /home/baizhiwang/prov-data/splits/LUAD_subtyping_100 \
                    --data_root_dir /home/baizhiwang/Badge4/Uncertainty/dataset/"$dataset"/"$encoder" \
                    --data_dir /home/baizhiwang/Badge4/Uncertainty/dataset/"$dataset"/"$encoder" \
                    --preloading $preloading \
                    --in_dim ${in_dim[$backbone]} \
                    --mambamil_rate $mambamil_rate \
                    --mambamil_layer $mambamil_layer \
                    --mambamil_type $mambamil_type \
                    > "$log_dir/train_${task}_${model}_${backbone}_${label}shot.log" 2>&1
            ) &

            # 如果你想限制并发数量（可选）
            # while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
            #     sleep 2
            # done

        done
    done

    # ✅ 等这个label下所有model都跑完，再进入下一个label
    wait
    echo "✅ Finished label: $label"
done
