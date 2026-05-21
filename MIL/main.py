from __future__ import print_function

import argparse
import os

# internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from dataset.dataset_generic import Generic_MIL_Dataset

# pytorch imports
import torch
import pandas as pd
import numpy as np
import wandb
# torch.use_deterministic_algorithms(True)
def main(args):

    # wandb.init(project=args.task)
    # wandb.config.update(args)
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(args.backbone, args.patch_size, from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        if args.preloading == 'yes':
            for d in datasets:
                d.pre_loading()
            
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary'+args.label_col+'.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    mean_auc_test = final_df['test_auc'].mean()
    std_auc_test = final_df['test_auc'].std()
    mean_auc_val = final_df['val_auc'].mean()
    std_auc_val = final_df['val_auc'].std()
    mean_acc_test = final_df['test_acc'].mean()
    std_acc_test = final_df['test_acc'].std()
    mean_acc_val = final_df['val_acc'].mean()
    std_acc_val = final_df['val_acc'].std()

    # wandb.log({"mean_auc_test": mean_auc_test, "std_auc_test": std_auc_test, "mean_auc_val": mean_auc_val, "std_auc_val": std_auc_val})
    df_append = pd.DataFrame({
        'folds': ['mean', 'std'],
        'test_auc': [mean_auc_test, std_auc_test],
        'val_auc': [mean_auc_val, std_auc_val],
        'test_acc': [mean_acc_test, std_acc_test],
        'val_acc': [mean_acc_val, std_acc_val],
    })
    final_df = pd.concat([final_df, df_append])
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary'+args.label_col+'.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    final_df['folds'] = final_df['folds'].astype(str)
    table = wandb.Table(dataframe=final_df)
    # wandb.log({"summary": table})
    # wandb.log({"mean_auc_test": mean_auc_test, "mean_acc_test": mean_acc_test, "mean_auc_val": mean_auc_val, "mean_acc_val": mean_acc_val})


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=50,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=2, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='enable dropout (p=0.25)')
parser.add_argument('--model_type', type=str, default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str)
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--patch_size', type=str, default='256')
parser.add_argument('--preloading', type=str, default='no')
parser.add_argument('--in_dim', type=int, default=1536)
parser.add_argument('--label_col', type=str, default='ER',  help='label')
parser.add_argument('--data_dir', type=str, default='/home/baizhiwang/Badge/abMIL/dataset/BCNB', help='patch dir')
## mambamil

parser.add_argument('--mambamil_rate',type=int, default=10, help='mambamil_rate')
parser.add_argument('--mambamil_layer',type=int, default=2, help='mambamil_layer')
parser.add_argument('--mambamil_type',type=str, default='SRMamba', choices= ['Mamba', 'BiMamba', 'SRMamba'], help='mambamil_type')


args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Deviece is:', device)

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'model_type': args.model_type,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}


print('\nLoad Dataset')

if args.task == 'LUAD_LUSC':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/LUAD_LUSC.csv',
                            data_dir= None,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=False,
                            ignore=[])
elif args.task == 'BCNB_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/home/baizhiwang/prov-data/label/BCNB_EPH.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat= False,
                            label_col = args.label_col,
                            ignore=[])
elif args.task == 'LUAD_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/home/baizhiwang/prov-data/label/LUAD.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat= False,
                            label_col = args.label_col,
                            ignore=[])
elif args.task == 'IMPRESS_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset (csv_path = '/home/baizhiwang/prov-data/label/IMPRESS.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat= False,
                            label_col = args.label_col,
                            ignore=[])
    
elif args.task == 'LUAD_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset (csv_path = '/home/baizhiwang/prov-data/label/LUAD.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat= False,
                            label_col = args.label_col,
                            ignore=[])
        
elif args.task == 'BRCA_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/home/baizhiwang/prov-data/label/BRCA.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat= False,
                            label_col = args.label_col,
                            ignore=[])
    
elif args.task == 'BLCA_subtyping':
        args.n_classes = 2
        dataset = Generic_MIL_Dataset(csv_path = '/home/baizhiwang/prov-data/label/TCGA-BLCA.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat= False,
                            label_col = args.label_col,
                            ignore=[])
elif args.task == 'BRACS_xi':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/home/baizhiwang/prov-data/label/BRACS.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0,1:1,2:2,3:3,4:4,5:5,6:6},
                            # label_dict = {0:0,1:1,2:2,3:3,4:4,5:5,6:6},
                            patient_strat= False,
                            label_col = args.label_col,
                            ignore=[])
elif args.task == 'BRACS_cu':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = '/home/baizhiwang/prov-data/label/BRACS.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0,1:1,2:2},
                            patient_strat= False,
                            label_col = args.label_col,
                            ignore=[])
    
elif args.task == 'CAMELYON17_subtyping':
    args.n_classes=4
    dataset = Generic_MIL_Dataset(csv_path = '/home/baizhiwang/prov-data/label/camelyon17_stage_4subtyping.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'negative':0, 'macro':1,'micro':2,'itc':3},
                            patient_strat= False,
                            label_col = args.label_col,
                            ignore=[])

elif args.task == 'CAMELYON16_subtyping':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/home/baizhiwang/prov-data/label/camelyon16.csv',
                            data_dir= args.data_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0,1:1},
                            patient_strat= False,
                            label_col = args.label_col,
                            ignore=[])


else:
    raise NotImplementedError
    
# if not os.path.isdir(args.results_dir):
    # os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment.txt', 'w') as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        


# set auto resume 
if args.k_start == -1:
    folds = args.k if args.k_end == -1 else args.k_end
    for i in range(folds):
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        if not os.path.exists(filename):
            args.k_start = i
            break
    print('Training from fold: {}'.format(args.k_start))

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


