import numpy as np
import torch
from easydict import EasyDict

from .misc import BlackHole


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    elif cfg.type is None:
        return BlackHole()
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def get_warmup_sched(cfg, optimizer):
    if cfg is None: return BlackHole()
    lambdas = [lambda it : (it / cfg.max_iters) if it <= cfg.max_iters else 1 for _ in optimizer.param_groups]
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return warmup_sched


def log_losses(out, it, tag, logger=BlackHole(), writer=BlackHole(), others={}, dynamic_loss_weights=None):
    logstr = '[%s] Iter %05d' % (tag, it)
    logstr += ' | loss %.4f' % out['overall'].item()
    for k, v in out.items():
        if k == 'overall': continue
        # 如果启用了动态损失权重，则获取该损失项的权重
        if dynamic_loss_weights is not None and hasattr(dynamic_loss_weights, 'current_weights') and k in dynamic_loss_weights.current_weights:
            weight = dynamic_loss_weights.current_weights[k]
            logstr += ' | %s(%.4f) %.4f' % (k, weight, v.item())
        else:
            logstr += ' | %s %.4f' % (k, v.item())
    for k, v in others.items():
       logstr += ' | %s %2.4f' % (k, v)
    logger.info(logstr)


    for k, v in out.items():
       if k == 'overall':
            writer.add_scalar('%s/loss' % tag, v, it)
       else:
           writer.add_scalar('%s/loss_%s' % (tag, k), v, it)
        # 记录权重和不确定性到 TensorBoard
           if dynamic_loss_weights is not None and hasattr(dynamic_loss_weights, 'current_weights') and k in dynamic_loss_weights.current_weights:
               weight = dynamic_loss_weights.current_weights[k]
               log_var = dynamic_loss_weights.log_vars[k].item()
               writer.add_scalar('%s/weight_%s' % (tag, k), weight, it)
               writer.add_scalar('%s/log_var_%s' % (tag, k), log_var, it)
    for k, v in others.items():
        writer.add_scalar('%s/%s' % (tag, k), v, it)
    
    writer.flush()


class ValidationLossTape(object):

    def __init__(self):
        super().__init__()
        self.accumulate = {}
        self.others = {}
        self.total = 0

    def update(self, out, n, others={}):
        self.total += n
        for k, v in out.items():
            if k not in self.accumulate:
                self.accumulate[k] = v.clone().detach()
            else:
                self.accumulate[k] += v.clone().detach()

        for k, v in others.items():
            if k not in self.others:
                self.others[k] = v.clone().detach()
            else:
                self.others[k] += v.clone().detach()
        

    def log(self, it, logger=BlackHole(), writer=BlackHole(), tag='val', dynamic_loss_weights=None):
        avg = EasyDict({k:v / self.total for k, v in self.accumulate.items()})
        avg_others = EasyDict({k:v / self.total for k, v in self.others.items()})
        log_losses(avg, it, tag, logger, writer, others=avg_others, dynamic_loss_weights=dynamic_loss_weights)
        return avg['overall']


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        if device == 'cpu':
            return obj.cpu()
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def reweight_loss_by_sequence_length(length, max_length, mode='sqrt'):
    if mode == 'sqrt':
        w = np.sqrt(length / max_length)
    elif mode == 'linear':
        w = length / max_length
    elif mode is None:
        w = 1.0
    else:
        raise ValueError('Unknown reweighting mode: %s' % mode)
    return w


def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
