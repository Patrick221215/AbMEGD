import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from AbMEGD.datasets import get_dataset
from AbMEGD.models import get_model
from AbMEGD.utils.misc import *
from AbMEGD.utils.data import *
from AbMEGD.utils.train import *


import torch.nn as nn

class DynamicLossWeights(nn.Module):
    def __init__(self, initial_weights):
        super(DynamicLossWeights, self).__init__()
        # 初始化每个损失项的可学习不确定性参数（log scale）
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor([torch.log(torch.tensor(v))], dtype=torch.float32))
            for name, v in initial_weights.items()
        })

    def forward(self, losses):
        total_loss = 0
        weighted_losses = {}
        for name, loss in losses.items():
            log_var = self.log_vars[name]
            weight = torch.exp(-2 * log_var)  # 对不确定性取反比
            weighted_loss = weight * loss + log_var
            total_loss += weighted_loss
            weighted_losses[name] = weighted_loss
        return total_loss, weighted_losses



import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize

class DynamicLossWeights(nn.Module):
    def __init__(self, initial_weights, beta=1.0, total_weight=3.0, weight_min=0.5, weight_max=1.5):
        super(DynamicLossWeights, self).__init__()
        self.beta = beta
        self.total_weight = total_weight
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.loss_names = list(initial_weights.keys())
        self.initial_weights = torch.tensor(
            [initial_weights[name] for name in self.loss_names],
            dtype=torch.float32
        )
        self.epsilon = 1e-8  # 防止除零错误

        # 初始化每个损失项的可学习不确定性参数（对数尺度）
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor([0.0], dtype=torch.float32))  # 初始值为 0，对应 log(1)
            for name in initial_weights.keys()
        })

    def forward(self, losses):
        # 获取损失值并转换为张量
        loss_values = torch.tensor([losses[name].item() for name in self.loss_names])
        # 防止损失值为零或负数
        loss_values = torch.abs(loss_values) + self.epsilon

        # Step 1: 计算调整因子（考虑不确定性）
        # 计算当前的不确定性
        log_vars = torch.tensor([self.log_vars[name].item() for name in self.loss_names])
        uncertainties = torch.exp(log_vars)  # 不确定性 sigma

        # 计算调整因子
        adjustment = (loss_values / (self.initial_weights * uncertainties + self.epsilon)) ** self.beta

        # Step 2: 更新权重
        weights = self.initial_weights * adjustment

        # Step 3: 调整权重，使其满足总和和范围约束
        weights = self.adjust_weights(weights, self.weight_min, self.weight_max, self.total_weight)

        # 构建权重字典
        weight_dict = {name: weights[i].item() for i, name in enumerate(self.loss_names)}

        # 保存当前权重以便日志记录
        self.current_weights = weight_dict

        # 计算加权损失
        total_loss = 0
        weighted_losses = {}
        for i, name in enumerate(self.loss_names):
            weight = weights[i]
            loss = losses[name]
            log_var = self.log_vars[name]
            # 加权损失项，包含不确定性正则项
            weighted_loss = weight * loss / (2 * torch.exp(log_var) ** 2) + log_var
            total_loss += weighted_loss
            weighted_losses[name] = weighted_loss

        return total_loss, weighted_losses

    def adjust_weights(self, weights, weight_min, weight_max, total_weight):
        """
        调整权重，使其满足总和为 total_weight，且每个权重在 [weight_min, weight_max] 范围内
        """
        n = len(weights)
        weights_np = weights.detach().cpu().numpy()

        # 定义优化目标函数
        def objective(w):
            return 0.5 * np.sum((w - weights_np) ** 2)

        # 总和约束
        def constraint_sum(w):
            return np.sum(w) - total_weight

        # 变量的范围约束
        bounds = [(weight_min, weight_max) for _ in range(n)]

        # 定义约束条件
        constraints = ({
            'type': 'eq',
            'fun': constraint_sum
        },)

        # 使用 scipy.optimize.minimize 进行优化
        result = minimize(
            objective,
            weights_np,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            adjusted_weights = torch.tensor(result.x, dtype=weights.dtype)
        else:
            # 如果优化失败，使用简单的重新分配方法
            adjusted_weights = self.redistribute_weights(weights, weight_min, weight_max, total_weight)

        return adjusted_weights

    def redistribute_weights(self, weights, weight_min, weight_max, total_weight):
        """
        如果优化失败，简单地调整权重，使其满足约束
        """
        adjusted_weights = weights.clone()
        n = len(adjusted_weights)

        # 先将权重裁剪到范围内
        adjusted_weights = torch.clamp(adjusted_weights, weight_min, weight_max)

        # 计算当前权重之和
        current_sum = adjusted_weights.sum().item()

        # 如果权重之和与 total_weight 不同，按比例缩放
        if abs(current_sum - total_weight) > self.epsilon:
            adjusted_weights = adjusted_weights / current_sum * total_weight

            # 再次裁剪，防止因缩放导致超出范围
            adjusted_weights = torch.clamp(adjusted_weights, weight_min, weight_max)

            # 重新计算权重之和
            current_sum = adjusted_weights.sum().item()

            # 如果仍然不满足 total_weight，分配剩余的权重
            diff = total_weight - current_sum
            for i in range(n):
                if diff > 0:
                    max_increase = min(diff, weight_max - adjusted_weights[i].item())
                    adjusted_weights[i] += max_increase
                    diff -= max_increase
                elif diff < 0:
                    max_decrease = min(-diff, adjusted_weights[i].item() - weight_min)
                    adjusted_weights[i] -= max_decrease
                    diff += max_decrease
                if abs(diff) < self.epsilon:
                    break

        return adjusted_weights



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:1')
    #parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)
    train_iterator = inf_iterator(DataLoader(
        train_dataset, 
        batch_size=config.train.batch_size, 
        collate_fn=PaddingCollate(), 
        shuffle=True,
        num_workers=args.num_workers
    ))
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    dynamic_loss_weights = DynamicLossWeights(config.train.loss_weights).to(args.device)
    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    # 将 dynamic_loss_weights 的参数加入优化器
    optimizer.add_param_group({'params': dynamic_loss_weights.parameters()})
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1
    
    
    # Resume
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])

    # Train
    def train(it):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(train_iterator), args.device)

        # Forward
        # if args.debug: torch.set_anomaly_enabled(True)
        loss_dict = model(batch)
        #loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        #loss_dict['overall'] = loss
        # 使用动态损失加权计算总损失
        loss, weighted_losses = dynamic_loss_weights(loss_dict)
        weighted_losses['overall'] = loss  # 添加总损失到字典中
        
        time_forward_end = current_milli_time()

        # Backward
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        log_losses(weighted_losses, it, 'train', logger, writer, others={
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        }, dynamic_loss_weights=dynamic_loss_weights)
            
        if not torch.isfinite(loss):
            logger.error('NaN or Inf detected.')
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': it,
                'batch': recursive_to(batch, 'cpu'),
            }, os.path.join(log_dir, 'checkpoint_nan_%d.pt' % it))
            raise KeyboardInterrupt()

    # Validate
    def validate(it):
        loss_tape = ValidationLossTape()
        weighted_losses = []
        with torch.no_grad():
            model.eval()
            #for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
            for i, batch in enumerate(val_loader):
                # Prepare data
                batch = recursive_to(batch, args.device)
                # Forward
                loss_dict = model(batch)
                # loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                # loss_dict['overall'] = loss

                # loss_tape.update(loss_dict, 1)

                # 使用动态损失加权计算总损失
                loss, weighted_losses = dynamic_loss_weights(loss_dict)
                weighted_losses['overall'] = loss
                
                loss_tape.update(weighted_losses, 1)
        avg_loss = loss_tape.log(it, logger, writer, 'val', dynamic_loss_weights=dynamic_loss_weights)
        
        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss

# from tqdm import tqdm
# for it in tqdm(range(it_first, config.train.max_iters + 1)):
    try:
        #for it in range(it_first, config.train.max_iters + 1):
        for it in tqdm(range(it_first, config.train.max_iters + 1), desc='Train', dynamic_ncols=True):
            # if it == 160:
            #     import ipdb; ipdb.set_trace()
            train(it)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                if not args.debug:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
        
