'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-09-12 07:47:11
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-09-25 10:01:36
FilePath: /cjm/project/AbMEGD/AbMEGD/modules/common/topology.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn.functional as F


def get_consecutive_flag(chain_nb, res_nb, mask):
    """
    Args:
        chain_nb, res_nb
    Returns:
        consec: A flag tensor indicating whether residue-i is connected to residue-(i+1), 
                BoolTensor, (B, L-1)[b, i].
    """
    d_res_nb = (res_nb[:, 1:] - res_nb[:, :-1]).abs()   # (B, L-1)
    same_chain = (chain_nb[:, 1:] == chain_nb[:, :-1])
    consec = torch.logical_and(d_res_nb == 1, same_chain)
    consec = torch.logical_and(consec, mask[:, :-1])
    return consec


def get_terminus_flag(chain_nb, res_nb, mask):
    consec = get_consecutive_flag(chain_nb, res_nb, mask)
    N_term_flag = F.pad(torch.logical_not(consec), pad=(1, 0), value=1)
    C_term_flag = F.pad(torch.logical_not(consec), pad=(0, 1), value=1)
    return N_term_flag, C_term_flag
