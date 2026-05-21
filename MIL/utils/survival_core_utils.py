from argparse import Namespace
from collections import OrderedDict
import os
import pickle 
import pandas as pd
from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored

import torch

from dataset.dataset_generic import save_splits
from utils.survival_utils import *
import wandb

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = val_loss
        # score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class EarlyStopping_cindex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = val_loss
        # score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss



class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None
    if (args.k_fold):
        print('K-fold cross validation')
        train_split, val_split = datasets

    else:
        print('\nInit train/val/test splits...', end=' ')
        train_split, val_split, test_split = datasets
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))

        print('\nInit loss function...', end=' ')
    
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    args.fusion = None if args.fusion == 'None' else args.fusion



    if args.model_type == 'mean_mil':
        from models.Mean_Max_MIL import MeanMIL
        model = MeanMIL(args.in_dim, args.n_classes, survival = True)
    elif args.model_type == 'max_mil':
        from models.Mean_Max_MIL import MaxMIL
        model = MaxMIL(args.in_dim, args.n_classes, survival = True)
    elif args.model_type == 'att_mil':
        from models.ABMIL import DAttention
        model = DAttention(args.in_dim, args.n_classes, dropout = args.drop_out, act='relu', survival = True)
    elif args.model_type == 'trans_mil':
        from models.TransMIL import TransMIL
        model = TransMIL(args.in_dim, args.n_classes, dropout = args.drop_out, act='relu', survival = True)
    elif args.model_type == 's4model':
        from models.S4MIL import S4Model
        model = S4Model(in_dim = args.in_dim, n_classes = args.n_classes, act = 'gelu', dropout = args.drop_out, survival = True)
    elif args.model_type == 'mamba_mil':
        from models.MambaMIL import MambaMIL
        model = MambaMIL(in_dim = args.in_dim, n_classes=args.n_classes, dropout=args.drop_out, act='gelu', survival = True, layer = args.mambamil_layer, rate = args.mambamil_rate, type = args.mambamil_type)
    
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    if not args.k_fold:
        test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        if args.k_fold:
            early_stopping = EarlyStopping_cindex(warmup=0, patience=20, stop_epoch=40, verbose = True)
        else:
            early_stopping = EarlyStopping(warmup=0, patience=20, stop_epoch=40, verbose = True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.task_type == 'survival':
            train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
            stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args.k_fold)

        if stop:
            break
            


    print('Done!')
    # 训练结束后加载最佳权重
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))

    # 保存验证集生存曲线所需数据
    fold_dir = os.path.join(args.results_dir, f"fold_{cur}")
    _, val_cindex, val_paths = summary_survival(model, val_loader, args.n_classes, save_dir=fold_dir, split="val")
    print('Val c-Index: {:.4f}'.format(val_cindex))
    if val_paths:
        print(f"[VAL] KM数据: {val_paths['km_csv']}")
        print(f"[VAL] 预测曲线: {val_paths['pred_npz']}")

    if (not args.k_fold):
        results_dict, test_cindex, test_paths = summary_survival(model, test_loader, args.n_classes, save_dir=fold_dir, split="test")
        print('Test c-Index: {:.4f}'.format(test_cindex))
        if test_paths:
            print(f"[TEST] KM数据: {test_paths['km_csv']}")
            print(f"[TEST] 预测曲线: {test_paths['pred_npz']}")
        writer.close()
        return results_dict, test_cindex, val_cindex

    writer.close()
    return val_cindex


def train_loop_survival(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, batch in enumerate(loader):
        


        data_WSI, data_omic, label, event_time, c = batch
        data_WSI, data_omic = data_WSI.to(device, non_blocking = True), data_omic.to(device, non_blocking = True)
        label = label.to(device, non_blocking = True)
        c = c.to(device, non_blocking=True)
        # hazards, S, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic) # return hazards, S, Y_hat, A_raw, results_dict
        hazards, S, Y_hat, _, _ = model(data_WSI)
        # hazards = torch.sigmoid(hazards)
        # S = torch.cumprod(1 - hazards, dim=1)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk), data_WSI.size(0)))
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, k_fold=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            # hazards, S, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic) # return hazards, S, Y_hat, A_raw, results_dict
            hazards, S, Y_hat, _, _ = model(data_WSI) # return hazards, S, Y_hat, A_raw, results_dict
            # hazards = torch.sigmoid(hazards)
            # S = torch.cumprod(1 - hazards, dim=1)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss_surv, val_loss, c_index))
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        if k_fold:
            early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        else:
            early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_survival(model, loader, n_classes, save_dir: str = None, split: str = None):
    """
    汇总并(可选)保存用于绘制生存曲线的数据。
    保存内容：
      - {save_dir}/{split}_km_data.csv               # KM 曲线需要的数据（真实标签）
      - {save_dir}/{split}_pred_survival.npz         # 模型预测生存曲线：pred_surv [N,T]、slide_ids [N]、time_bins [T]、hazards [N,T]
    返回：
      patient_results, c_index, saved_paths(dict或None)
    """
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids_series = loader.dataset.slide_data['slide_id']
    patient_results = {}

    # 额外收集：每个样本的预测 survival(t) 和 hazards(t)
    pred_surv_list, hazards_list, id_list, labels_list = [], [], [], []

    for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        slide_id = slide_ids_series.iloc[batch_idx]

        with torch.no_grad():
            hazards, survival, Y_hat, _, _ = model(data_WSI)   # survival/hazards: [B,T]

        # —— 风险分数（越大风险越高）——
        risk_arr = -torch.sum(survival, dim=1).detach().cpu().numpy()  # shape (B,)
        risk_scalar = float(np.asarray(risk_arr).ravel()[0])

        # 标量化 event_time / censorship
        event_time_scalar = float(np.asarray(event_time).ravel()[0])
        c_scalar = float(np.asarray(c.detach().cpu().numpy()).ravel()[0])

        all_risk_scores[batch_idx] = risk_scalar
        all_censorships[batch_idx] = c_scalar
        all_event_times[batch_idx] = event_time_scalar

        # 收集预测曲线（假设 batch_size=1；若>1也能工作，会按样本逐条追加）
        pred_surv_list.append(survival.squeeze(0).detach().cpu().numpy())
        hazards_list.append(hazards.squeeze(0).detach().cpu().numpy())
        id_list.append(str(slide_id))
        labels_list.append(int(label.item()) if hasattr(label, "item") else int(label))

        # 兼容你原先的 patient_results（注意这里的 'survival' 字段是事件时间，不是曲线）
        patient_results.update({
            slide_id: {
                'slide_id': np.array(slide_id),
                'risk': risk_scalar,
                'disc_label': int(label.item()) if hasattr(label, "item") else int(label),
                'survival': event_time_scalar,
                'censorship': c_scalar
            }
        })

    # 计算 C-index（与你原逻辑一致）
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool),
        all_event_times,
        all_risk_scores,
        tied_tol=1e-08
    )[0]

    # —— 可选落盘 —— #
    saved_paths = None
    if save_dir and split:
        # 1) KM 曲线数据（真实标签）
        km_df = pd.DataFrame({
            "slide_id": id_list,
            "event_time": all_event_times,
            "event": (1 - all_censorships).astype(int),  # 1=发生事件，0=右删失
            "censorship": all_censorships.astype(int),
            "risk": all_risk_scores,
            "disc_label": labels_list
        })
        km_csv_path = os.path.join(save_dir, f"{split}_km_data.csv")
        km_df.to_csv(km_csv_path, index=False)

        # 2) 模型预测生存曲线
        pred_surv = np.vstack(pred_surv_list) if pred_surv_list else np.empty((0, 0))
        hazards_arr = np.vstack(hazards_list) if hazards_list else np.empty((0, 0))
        time_bins = np.arange(pred_surv.shape[1]) if pred_surv.size > 0 else np.array([])

        pred_npz_path = os.path.join(save_dir, f"{split}_pred_survival.npz")
        np.savez_compressed(
            pred_npz_path,
            pred_surv=pred_surv,          # [N, T] 每样本的 S(t)
            hazards=hazards_arr,          # [N, T] 每样本的 h(t)
            slide_ids=np.array(id_list),  # [N]
            time_bins=time_bins           # [T] 目前为离散索引，如有真实时间再替换
        )

        saved_paths = {"km_csv": km_csv_path, "pred_npz": pred_npz_path}

    return patient_results, c_index, saved_paths
