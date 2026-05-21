model_names='max_mil mean_mil att_mil trans_mil s4model mamba_mil'
# model_names='s4model'
backbones="resnet50"


declare -A in_dim
in_dim["resnet50"]=1536

num=4

declare -A gpus
gpus["mean_mil"]=$num
gpus["max_mil"]=$num
gpus["att_mil"]=$num
gpus["trans_mil"]=$num
gpus['s4model']=$num
gpus['mamba_mil']=$num

task="BRCA_subtyping"

# results_dir="/home/baizhiwang/Badge/MambaMIL/Badge/BRCA-gigapath_subtyping"
model_size="small" # since the dim of feature of vit-base is 768    
preloading="no"
patch_size="256"
dataset=TCGA-BRCA
encoder=Gigapath_tile
label=IDC
lr='2e-4' 
mambamil_rate='5'
mambamil_layer='2'
mambamil_type='SRMamba'
results_dir=/home/baizhiwang/Badge4/Uncertainty/experiments/result/"$dataset"/"$encoder"/"$label"

for model in $model_names
do
    for backbone in $backbones
    do
        exp=$model"/"$backbone
        echo $exp", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}
        # k_start and k_end, only for resuming, default is -1
        
        k_start=-1
        k_end=-1
        python /home/baizhiwang/Badge4/Uncertainty/UAFRE-Github/MIL/main.py\
            --drop_out 0\
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
            --label_col $label\
            --split_dir /home/baizhiwang/Badge/abMIL/CLAM/Badge/TCGA-BRCA-result/split/BRCA_subtyping_100 \
            --data_root_dir /home/baizhiwang/Badge4/Uncertainty/dataset/"$dataset"/"$encoder"\
            --data_dir /home/baizhiwang/Badge4/Uncertainty/dataset/"$dataset"/"$encoder"\
            --preloading $preloading \
            --in_dim ${in_dim[$backbone]} \
            --mambamil_rate $mambamil_rate \
            --mambamil_layer $mambamil_layer \
            --mambamil_type $mambamil_type
    done > "/home/baizhiwang/Badge4/Uncertainty/experiments/log/train_${task}_${model}_${label}shot" 2>&1 &done

