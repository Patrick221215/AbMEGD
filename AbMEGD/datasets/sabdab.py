import os
import random
import logging
import datetime
import pandas as pd
import joblib
import pickle
import lmdb
import subprocess
import torch
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..utils.protein import parsers, constants
from ._base import register_dataset


ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}

RESOLUTION_THRESHOLD = 4.0

TEST_ANTIGENS = [
    'sars-cov-2 receptor binding domain',
    'hiv-1 envelope glycoprotein gp160',
    'mers s',
    'influenza a virus',
    'cd27 antigen',
]


# ===== 只为 RAbD 测试集做的最小兼容修复 =====
# (pdbcode, 抗原链字符串) -> (H_id, L_id, Ag_id) 用在 entry_id 上
RABD_TEST_NAME_MAP = {
    ("3h3b", "B"): ("D", "E", "B"),  # 3h3b_*_*_B  -> 3h3b_D_E_B
    ("2ghw", "A"): ("B", "E", "A"),  # 2ghw_*_*_A  -> 2ghw_B_E_A
    ("3uzq", "B"): ("A", "C", "B"),  # 3uzq_*_*_B  -> 3uzq_A_C_B
}

# 这两个本来被 ag_type 过滤掉，但我们强行保留
FORCE_KEEP_ENTRY_IDS = {
    "1w72_I_M_DF",
    "3ffd_A_B_P",
}

# 这三条是 RAbD 测试集中需要强制保留 CDR 的 canonical id
RABD_CANONICAL_IDS = {"3h3b_D_E_B", "2ghw_B_E_A", "3uzq_A_C_B"}

def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    else:
        return val


def nan_to_none(val):
    if val != val or not val:
        return None
    else:
        return val


def split_sabdab_delimited_str(val):
    if not val:
        return []
    else:
        return [s.strip() for s in val.split('|')]


def parse_sabdab_resolution(val):
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    else:
        return float(val)


def _aa_tensor_to_sequence(aa):
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])


def _label_heavy_chain_cdr(data, seq_map, cdr_range_cls, max_cdr3_length=30):
    """
    根据给定的 CDR range 类（ChothiaCDRRange 或 IMGTCDRRange）给 heavy chain 打标签。
    """
    if data is None or seq_map is None:
        return data, seq_map

    # Add CDR labels
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = cdr_range_cls.to_cdr('H', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    # Add CDR sequence annotations
    data['H1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H1] )
    data['H2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H2] )
    data['H3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H3] )

    cdr3_length = (cdr_flag == constants.CDR.H3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.H3] = 0
        logging.warning(f'CDR-H3 too long {cdr3_length}. Removed.')
        return None, None

    # Filter: ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDR-H3 found in the heavy chain.')
        return None, None

    return data, seq_map


def _label_light_chain_cdr(data, seq_map, cdr_range_cls, max_cdr3_length=30):
    """
    根据给定的 CDR range 类（ChothiaCDRRange 或 IMGTCDRRange）给 light chain 打标签。
    """
    if data is None or seq_map is None:
        return data, seq_map

    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = cdr_range_cls.to_cdr('L', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    data['L1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L1] )
    data['L2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L2] )
    data['L3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L3] )

    cdr3_length = (cdr_flag == constants.CDR.L3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.L3] = 0
        logging.warning(f'CDR-L3 too long {cdr3_length}. Removed.')
        return None, None

    # Ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDRs found in the light chain.')
        return None, None

    return data, seq_map

def preprocess_sabdab_structure(task):
    entry = task['entry']
    pdb_path = task['pdb_path']

    numbering_scheme = task.get('numbering_scheme', 'chothia').lower()
    cdr_range_cls = constants.get_cdr_scheme(numbering_scheme)

    parser = PDB.PDBParser(QUIET=True)
    try:
        # 用 entry['id'] 作为结构 id
        model = parser.get_structure(entry['id'], pdb_path)[0]
    except ValueError as e:
        # Bio.PDB 在解析奇怪的 resseq（比如 'X'）时会抛这个错
        logging.warning('[{}] {}: {}'.format(
            entry['id'],
            e.__class__.__name__,
            str(e),
        ))
        return None

    parsed = {
        'id': entry['id'],
        'heavy': None,
        'heavy_seqmap': None,
        'light': None,
        'light_seqmap': None,
        'antigen': None,
        'antigen_seqmap': None,
    }

    try:
        # --------- Heavy chain ----------
        raw_heavy = None
        raw_heavy_seqmap = None
        if entry['H_chain'] is not None:
            H_chain = entry['H_chain']
            try:
                heavy_data, heavy_seqmap = parsers.parse_biopython_structure(
                    model[H_chain],
                    max_resseq=113 if numbering_scheme == 'chothia' else 128,
                )
                raw_heavy, raw_heavy_seqmap = heavy_data, heavy_seqmap

                # 正常按 CDR 规则打标签
                parsed['heavy'], parsed['heavy_seqmap'] = _label_heavy_chain_cdr(
                    heavy_data, heavy_seqmap, cdr_range_cls
                )

                # 只对 RAbD 的三条特例：如果被 CDR 规则筛掉，就直接保留原始 heavy
                if parsed['heavy'] is None and raw_heavy is not None \
                        and entry['id'] in RABD_CANONICAL_IDS:
                    parsed['heavy'], parsed['heavy_seqmap'] = raw_heavy, raw_heavy_seqmap

            except Exception as e:
                logging.warning(
                    "[%s] HEAVY chain=%r parse FAILED: %s: %s",
                    entry['id'], H_chain, e.__class__.__name__, str(e)
                )
                parsed['heavy'], parsed['heavy_seqmap'] = None, None

        # --------- Light chain ----------
        raw_light = None
        raw_light_seqmap = None
        if entry['L_chain'] is not None:
            L_chain = entry['L_chain']
            try:
                light_data, light_seqmap = parsers.parse_biopython_structure(
                    model[L_chain],
                    max_resseq=106 if numbering_scheme == 'chothia' else 128,
                )
                raw_light, raw_light_seqmap = light_data, light_seqmap

                parsed['light'], parsed['light_seqmap'] = _label_light_chain_cdr(
                    light_data, light_seqmap, cdr_range_cls
                )

                if parsed['light'] is None and raw_light is not None \
                        and entry['id'] in RABD_CANONICAL_IDS:
                    parsed['light'], parsed['light_seqmap'] = raw_light, raw_light_seqmap

            except Exception as e:
                logging.warning(
                    "[%s] LIGHT chain=%r parse FAILED: %s: %s",
                    entry['id'], L_chain, e.__class__.__name__, str(e)
                )
                parsed['light'], parsed['light_seqmap'] = None, None

        # 至少要有一条链
        if parsed['heavy'] is None and parsed['light'] is None:
            raise ValueError('Neither valid H-chain or L-chain is found.')

        # 抗原链照旧
        if len(entry['ag_chains']) > 0:
            chains = [model[c] for c in entry['ag_chains']]
            parsed['antigen'], parsed['antigen_seqmap'] = parsers.parse_biopython_structure(
                chains
            )

    except (
        PDBExceptions.PDBConstructionException,
        parsers.ParsingException,
        KeyError,
        ValueError,
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            entry['id'],
            e.__class__.__name__,
            str(e),
        ))
        return None

    return parsed


class SAbDabDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(
        self, 
        summary_path = './data/sabdab_summary_all.tsv', 
        chothia_dir = './data/all_structures/chothia', 
        processed_dir = './data/processed',
        split = 'train',
        split_seed = 42,
        transform = None,
        reset = False,
        idx_path = None, 
        numbering_scheme: str = 'chothia',
    ):
        super().__init__()
        self.summary_path = summary_path
        self.chothia_dir = chothia_dir
        if not os.path.exists(chothia_dir):
            raise FileNotFoundError(
                f"SAbDab structures not found in {chothia_dir}. "
                "Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
            )

        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        self.numbering_scheme = numbering_scheme.lower()
        self.cdr_range_cls = constants.get_cdr_scheme(self.numbering_scheme)

        self.sabdab_entries = None
        self.db_conn = None
        self.db_ids = None
        self.clusters = None
        self.id_to_cluster = None
        self.ids_in_split = None

        # 关键：先定义，后面 get_structure 才能安全访问
        self.id_map = None

        self._load_sabdab_entries()
        self._load_structures(reset)

        if idx_path is None:
            self._load_clusters(reset)
        else:
            self.clusters = {}
            self.id_to_cluster = {}

        self._load_split(split, split_seed, idx_path)

        self.transform = transform
        

    def _load_sabdab_entries(self):
        df = pd.read_csv(self.summary_path, sep='\t')
        entries_all = []
        
        max_entries = 162  # 设置要处理的最大条目数
        for i, row in tqdm(
            df.iterrows(), 
            dynamic_ncols=True, 
            desc='Loading entries',
            total=len(df),
        ):
            # if i >= max_entries:
            #     break

            pdbcode = row['pdb']          
            # 还是用原来的规则生成 H/L/Ag 字符串，只是在 entry_id 上对少数 RAbD 特例修正
            H_raw = nan_to_empty_string(row['Hchain'])
            L_raw = nan_to_empty_string(row['Lchain'])
            ag_chains = split_sabdab_delimited_str(
                nan_to_empty_string(row['antigen_chain'])
            )
            ag_str = ''.join(ag_chains)

            # 对 3h3b / 2ghw / 3uzq 这三个 test case 做链名重写，其它保持原样
            if (pdbcode, ag_str) in RABD_TEST_NAME_MAP:
                H_id, L_id, Ag_id = RABD_TEST_NAME_MAP[(pdbcode, ag_str)]
            else:
                H_id, L_id, Ag_id = H_raw, L_raw, ag_str

            entry_id = f"{pdbcode}_{H_id}_{L_id}_{Ag_id}"

            resolution = parse_sabdab_resolution(row['resolution'])
            entry = {
                'id': entry_id,
                'pdbcode': pdbcode,
                # 注意这里仍然用真实的 H/L 链来解析结构，不动原始过滤语义
                'H_chain': nan_to_none(row['Hchain']),
                'L_chain': nan_to_none(row['Lchain']),
                'ag_chains': ag_chains,
                'ag_type': nan_to_none(row['antigen_type']),
                'ag_name': nan_to_none(row['antigen_name']),
                'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'),
                'resolution': resolution,
                'method': row['method'],
                'scfv': row['scfv'],
            }

            # ===== Filtering：保留原有规则，只对 1w72 / 3ffd 开一个白名单后门 =====
            is_allowed_type = (
                entry['ag_type'] in ALLOWED_AG_TYPES or entry['ag_type'] is None
            )
            is_good_resolution = (
                entry['resolution'] is not None
                and entry['resolution'] <= RESOLUTION_THRESHOLD
            )
            force_keep = entry_id in FORCE_KEEP_ENTRY_IDS

            if (is_allowed_type and is_good_resolution) or (force_keep and is_good_resolution):
                if force_keep and not is_allowed_type:
                    logging.warning(
                        f"[SAbDabDataset] FORCE-KEEP entry_id={entry_id} "
                        f"with ag_type='{entry['ag_type']}'"
                    )
                entries_all.append(entry)

        self.sabdab_entries = entries_all


    def _load_structures(self, reset):
        if not os.path.exists(self._structure_cache_path) or reset:
            if os.path.exists(self._structure_cache_path):
                os.unlink(self._structure_cache_path)
            self._preprocess_structures()

        with open(self._structure_cache_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)
        self.sabdab_entries = list(
            filter(
                lambda e: e['id'] in self.db_ids,
                self.sabdab_entries
            )
        )

    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')
        
    def _preprocess_structures(self):
        tasks = []
        for entry in self.sabdab_entries:
            pdb_path = os.path.join(self.chothia_dir, '{}.pdb'.format(entry['pdbcode']))
            if not os.path.exists(pdb_path):
                logging.warning(f"PDB not found: {pdb_path}")
                continue
            tasks.append({
                'id': entry['id'],
                'entry': entry,
                'pdb_path': pdb_path,
                'numbering_scheme': self.numbering_scheme,  # ★★★ 新增：把当前编号方案传下去
            })

        data_list = joblib.Parallel(
            n_jobs = max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(preprocess_sabdab_structure)(task)
            for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
        )

        db_conn = lmdb.open(
            self._structure_cache_path,
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:
                    continue
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)

    @property
    def _cluster_path(self):
        return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')

    def _load_clusters(self, reset):
        if not os.path.exists(self._cluster_path) or reset:
            self._create_clusters()

        clusters, id_to_cluster = {}, {}
        with open(self._cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name
        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

    def _create_clusters(self):
        cdr_records = []
        for id in self.db_ids:
            structure = self.get_structure(id)
            if structure['heavy'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['heavy']['H3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
            elif structure['light'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['light']['L3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
        fasta_path = os.path.join(self.processed_dir, 'cdr_sequences.fasta')
        SeqIO.write(cdr_records, fasta_path, 'fasta')

        cmd = ' '.join([
            'mmseqs', 'easy-cluster',
            os.path.realpath(fasta_path),
            'cluster_result', 'cluster_tmp',
            '--min-seq-id', '0.5',
            '-c', '0.8',
            '--cov-mode', '1',
        ])
        subprocess.run(cmd, cwd=self.processed_dir, shell=True, check=True)

    def _load_split(self, split, split_seed, idx_path=None):
        """
        如果给了 idx_path，就直接从 idx 文件读取样本 id；
        否则使用原来的 TEST_ANTIGENS + cluster 划分方式。
        """
        assert split in ('train', 'val', 'test')

        # ===== 1) 优先使用外部 idx 文件 =====
        if idx_path is not None:
            if not os.path.exists(idx_path):
                raise FileNotFoundError(f"idx_path 不存在: {idx_path}")
            with open(idx_path) as f:
                ids = [l.strip() for l in f if l.strip()]

            db_ids_set = set(self.db_ids)

            # ---- 1.1 train / val：严格模式，完全复刻“之前版本”的逻辑 ----
            if split in ('train', 'val'):
                ok = [i for i in ids if i in db_ids_set]
                missing = [i for i in ids if i not in db_ids_set]

                if missing:
                    logging.warning(
                        f"[SAbDabDataset] (strict idx, split={split}) "
                        f"{len(missing)} 个 id 在 LMDB 中找不到，将被忽略，例子: {missing[:5]}"
                    )
                if len(ok) == 0:
                    raise RuntimeError(
                        f"[SAbDabDataset] (strict idx, split={split}) "
                        f"idx_path={idx_path} 里没有任何 id 能在 LMDB 中找到，"
                        f"请检查命名是否和 sabdab entry_id 一致。"
                    )

                self.ids_in_split = ok
                # train / val 不做任何映射，保持和旧版本完全一致
                self.id_map = None
                logging.info(
                    f"[SAbDabDataset] (strict idx, split={split}) "
                    f"使用外部 idx_path={idx_path}，有效样本数={len(ok)}"
                )
                return

            # ---- 1.2 test：允许 RAbD 的 3 个 alias → canonical 映射，其它仍然严格 ----
            logical_ids = []
            id_map = {}
            missing = []

            for idx_id in ids:
                real_id = None

                # 1) 完全匹配：和旧逻辑一样
                if idx_id in db_ids_set:
                    real_id = idx_id
                else:
                    # 2) 只对 RAbD 的 3 个特殊 case 做 alias 映射
                    parts = idx_id.split('_')
                    if len(parts) == 4:
                        pdbcode, H, L, Ag = parts
                        key = (pdbcode, Ag)
                        if key in RABD_TEST_NAME_MAP:
                            H_id, L_id, Ag_id = RABD_TEST_NAME_MAP[key]
                            canonical_id = f"{pdbcode}_{H_id}_{L_id}_{Ag_id}"
                            if canonical_id in db_ids_set:
                                real_id = canonical_id
                                logging.info(
                                    f"[SAbDabDataset] (test alias) "
                                    f"{idx_id} -> {canonical_id}"
                                )

                if real_id is None:
                    missing.append(idx_id)
                else:
                    logical_ids.append(idx_id)
                    # 只有 alias 映射的才需要记录到 id_map
                    if real_id != idx_id:
                        id_map[idx_id] = real_id

            if missing:
                logging.warning(
                    f"[SAbDabDataset] (test idx) {len(missing)} 个 id 在 LMDB 中找不到，将被忽略，"
                    f"例子: {missing[:5]}"
                )
            if len(logical_ids) == 0:
                raise RuntimeError(
                    f"[SAbDabDataset] (test idx) idx_path={idx_path} 里没有任何 id 能在 LMDB 中找到 "
                    f"（包括 RAbD alias 尝试），请检查命名是否和 sabdab entry_id 一致。"
                )

            self.ids_in_split = logical_ids      # 对外仍然是 idx 文件里的“逻辑 id”
            self.id_map = id_map if len(id_map) > 0 else None
            logging.info(
                f"[SAbDabDataset] (test idx) 使用外部 idx_path={idx_path}，有效样本数={len(logical_ids)}，"
                f"其中 alias 映射条数={len(id_map)}"
            )
            return

        # ===== 2) 没有提供 idx_path -> 使用原来的 DiffAb 划分逻辑 =====
        self.id_map = None  # 不需要映射，直接用 entry['id']

        ids_test = [
            entry['id']
            for entry in self.sabdab_entries
            if entry['ag_name'] in TEST_ANTIGENS
        ]
        test_relevant_clusters = set([self.id_to_cluster[id] for id in ids_test])

        ids_train_val = [
            entry['id']
            for entry in self.sabdab_entries
            if self.id_to_cluster[entry['id']] not in test_relevant_clusters
        ]
        random.Random(split_seed).shuffle(ids_train_val)
        if split == 'test':
            self.ids_in_split = ids_test
        elif split == 'val':
            self.ids_in_split = ids_train_val[:20]
        else:
            self.ids_in_split = ids_train_val[20:]


    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get_structure(self, id):
        """
        id: 逻辑 id（可能来自 idx 文件）。
        如果存在 self.id_map（仅在 test + alias 场景下），先映射到实际 LMDB key。
        """
        self._connect_db()
        real_id = self.id_map.get(id, id) if self.id_map is not None else id
        with self.db_conn.begin() as txn:
            raw = txn.get(real_id.encode())
            if raw is None:
                raise KeyError(f"Structure id '{real_id}' not found in LMDB.")
            return pickle.loads(raw)

    def __len__(self):
        return len(self.ids_in_split)

    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.get_structure(id)
        if self.transform is not None:
            data = self.transform(data)
        return data


@register_dataset('sabdab')
def get_sabdab_dataset(cfg, transform):
    return SAbDabDataset(
        summary_path = cfg.summary_path,
        chothia_dir = cfg.chothia_dir,
        processed_dir = cfg.processed_dir,
        split = cfg.split,
        split_seed = cfg.get('split_seed', 42),
        transform = transform,
        idx_path = cfg.get('idx_path', None),
        numbering_scheme = getattr(cfg, 'numbering_scheme', 'chothia'), 
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='./data/processed')
    parser.add_argument('--numbering_scheme', type=str, default='chothia')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()
    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()
    dataset = SAbDabDataset(
        processed_dir=args.processed_dir,
        split=args.split, 
        reset=args.reset,
        numbering_scheme = args.numbering_scheme,
    )
    print(dataset[0])
    print(len(dataset), len(dataset.clusters))
