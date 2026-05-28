import os
import time
import random
import logging
from typing import OrderedDict
import torch
import torch.linalg
import numpy as np
import yaml
from easydict import EasyDict
from glob import glob


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class Counter(object):
    def __init__(self, start=0):
        super().__init__()
        self.now = start

    def step(self, delta=1):
        prev = self.now
        self.now += delta
        return prev


# def get_logger(name, log_dir=None):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)
#     formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

#     stream_handler = logging.StreamHandler()
#     stream_handler.setLevel(logging.DEBUG)
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)


#     if log_dir is not None:
#         file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
#         file_handler.setLevel(logging.DEBUG)
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)

#     return logger

def get_logger(name, log_dir=None):
    """
    约定：
    - 所有真正的 handler（控制台 / 文件）只挂在 root logger 上。
    - get_logger 只负责返回一个带名字的 logger，并让它的日志往 root 传（propagate=True）。
    这样：
      * debug 模式（log_dir=None）：root -> 控制台
      * 正常训练（log_dir 不为 None）：root -> log.txt，不往控制台打
    """
    # -------- 配置 root logger --------
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 清空之前的 handler，避免重复添加 / 双重输出
    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        '[%(asctime)s::%(name)s::%(levelname)s] %(message)s'
    )

    if log_dir is None:
        # debug 模式：只往控制台输出
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    else:
        # 训练模式：只写文件，不往控制台输出
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # -------- 返回子 logger（如 'train'） --------
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # 让它把消息交给 root 处理，不自己挂 handler
    logger.propagate = True
    logger.handlers = []   # 确保没有多余 handler

    return logger




def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def get_checkpoint_path(folder, it=None):
    if it is not None:
        return os.path.join(folder, '%d.pt' % it), it
    all_iters = list(map(lambda x: int(os.path.basename(x[:-3])), glob(os.path.join(folder, '*.pt'))))
    all_iters.sort()
    return os.path.join(folder, '%d.pt' % all_iters[-1]), all_iters[-1]


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name


def extract_weights(weights: OrderedDict, prefix):
    extracted = OrderedDict()
    for k, v in weights.items():
        if k.startswith(prefix):
            extracted.update({
                k[len(prefix):]: v
            })
    return extracted


def current_milli_time():
    return round(time.time() * 1000)
