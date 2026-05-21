model_names='max_mil mean_mil att_mil trans_mil s4model mamba_mil'
labels='ATM FGFR3 KDM6A PIK3CA RB1 TP53'
backbone="resnet50"
# model="att_mil"
# num=2

declare -A in_dim
in_dim["resnet50"]=768

declare -A gpus
gpus["mean_mil"]=0
gpus["max_mil"]=0
gpus["att_mil"]=0
gpus["trans_mil"]=2
gpus['s4model']=1
gpus['mamba_mil']=3

task="BLCA_subtyping"
# results_dir="/jhcnas3/Pathology/experiments/train_vl/"$task
encoder="CHIEF_tile"
model_size="small" # since the dim of feature of vit-base is 768    
preloading="no"
patch_size="256"
dataset=BLCA

lr='2e-4'
mambamil_rate='5'
mambamil_layer='2'
mambamil_type='SRMamba'

for model in $model_names
do
    for label in $labels
    do
        results_dir=/home/baizhiwang/Badge4/OWSIA/data_fusion/new_result/"$dataset"/"$encoder"/"$label"
        exp=$model"/"$backbone
        echo $exp", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}
        # k_start and k_end, only for resuming, default is -1
        k_start=-1
        k_end=-1
        python /home/baizhiwang/Badge4/Uncertainty/UAFRE-Github/MIL/main.py \
            --drop_out 0\
            --early_stopping \
            --lr $lr \
            --k 5 \
            --k_start $k_start \
            --k_end $k_end \
            --label_frac 1.0 \
            --exp_code $exp \
            --patch_size $patch_size \
            --weighted_sample \
            --task $task \
            --backbone $backbone \
            --results_dir $results_dir \
            --model_type $model \
            --log_data \
            --label_col $label\
            --split_dir /home/baizhiwang/prov-data/splits/BLCA_subtyping_100\
            --data_root_dir /home/baizhiwang/prov-data/embedding/"$dataset"/"$encoder"\
            --data_dir /home/baizhiwang/prov-data/embedding/"$dataset"/"$encoder"\
            --preloading $preloading \
            --in_dim ${in_dim[$backbone]} \
            --mambamil_rate $mambamil_rate \
            --mambamil_layer $mambamil_layer \
            --mambamil_type $mambamil_type
    done > "/home/baizhiwang/Badge4/OWSIA/experiments/log/train_${task}_${model}_${label}shot" 2>&1 &done

