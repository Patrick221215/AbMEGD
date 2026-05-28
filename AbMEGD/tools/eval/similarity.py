import numpy as np
from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.PDB import Selection
from Bio.PDB.Polypeptide import three_to_one

try:
    from .base import EvalTask
except ImportError:
    from base import EvalTask


BLOSUM62 = substitution_matrices.load("BLOSUM62")


def reslist_rmsd(res_list1, res_list2):
    res_short, res_long = (res_list1, res_list2) if len(res_list1) < len(res_list2) else (res_list2, res_list1)
    M, N = len(res_short), len(res_long)
    if M == 0 or N == 0:
        return np.nan

    def d(i, j):
        coord_i = np.array(res_short[i]["CA"].get_coord())
        coord_j = np.array(res_long[j]["CA"].get_coord())
        return ((coord_i - coord_j) ** 2).sum()

    SD = np.full([M, N], np.inf)
    for i in range(M):
        j = N - (M - i)
        SD[i, j] = sum(d(i + k, j + k) for k in range(N - j))

    for j in range(N):
        SD[M - 1, j] = d(M - 1, j)

    for i in range(M - 2, -1, -1):
        for j in range((N - (M - i)) - 1, -1, -1):
            SD[i, j] = min(d(i, j) + SD[i + 1, j + 1], SD[i, j + 1])

    min_SD = SD[0, : N - M + 1].min()
    return float(np.sqrt(min_SD / M))


def entity_to_seq(entity):
    seq = ""
    mapping = []
    for res in Selection.unfold_entities(entity, "R"):
        try:
            seq += three_to_one(res.get_resname())
            mapping.append(res.get_id())
        except KeyError:
            pass
    return seq, mapping


def align_sequences(sequence_A, sequence_B):
    alns = pairwise2.align.globalds(
        sequence_A,
        sequence_B,
        BLOSUM62,
        -10.0,
        -0.5,
        penalize_end_gaps=(False, False),
    )
    if not alns:
        return (sequence_A, sequence_B), 0.0
    aligned_A, aligned_B, _, _, _ = alns[0]
    sl = len(aligned_A)
    if sl == 0:
        return (aligned_A, aligned_B), 0.0
    matches = [(aligned_A[i] == aligned_B[i]) and aligned_A[i] != "-" for i in range(sl)]
    aar = 100.0 * sum(matches) / sl
    return (aligned_A, aligned_B), float(aar)


def extract_reslist(model, residue_first, residue_last):
    if residue_first is None or residue_last is None:
        return []
    if residue_first[0] != residue_last[0]:
        return []

    chain_id = residue_first[0]
    pos_first, pos_last = tuple(residue_first[1:]), tuple(residue_last[1:])
    chain = model[chain_id]
    reslist = []
    for res in Selection.unfold_entities(chain, "R"):
        pos_current = (res.id[1], res.id[2])
        if pos_first <= pos_current <= pos_last:
            reslist.append(res)
    return reslist


def eval_region_similarity(task: EvalTask):
    model_gen = task.get_gen_biopython_model()
    model_ref = task.get_ref_biopython_model()

    reslist_gen = extract_reslist(model_gen, task.residue_first, task.residue_last)
    reslist_ref = extract_reslist(model_ref, task.residue_first, task.residue_last)

    seq_gen, _ = entity_to_seq(reslist_gen)
    seq_ref, _ = entity_to_seq(reslist_ref)
    _, aar_region = align_sequences(seq_gen, seq_ref)

    task.scores.update(
        {
            "RMSD_region": reslist_rmsd(reslist_gen, reslist_ref),
            "AAR_region": aar_region,
        }
    )
    return task
