from functools import lru_cache
import numpy as np
import torch, os
import torch.nn.functional as F
from diffusion.RGDM_diffusion import RGDiffusionTrainer
from diffusion.DDPM_diffusion import DDPM_DiffusionTrainer
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from .logging import get_logger
logger = get_logger(__name__)
from .pdb import pdb_to_npy, pdb_to_npy_interpolate

class ResidueDataset(Dataset):
    def __init__(self, args, split, **kwargs):
        super(ResidueDataset, self).__init__(**kwargs)
        self.split = split

        embeddings_arg_keys = ['omegafold_num_recycling']
        embeddings_suffix = get_args_suffix(embeddings_arg_keys, args) + '.npz'
        self.embeddings_suffix = embeddings_suffix

        self.lm_edge_dim = args.lm_edge_dim
        self.lm_node_dim = args.lm_node_dim
        self.args = args

    def get(self, idx):
        # get protein information
        row = self.split.iloc[idx]

        # make torch_geometric.data.Heterodata (=data)
        data = HeteroData()
        data.skip = False
        data['resi'].num_nodes = row.seqlen
        data['resi'].edge_index = get_dense_edges(row.seqlen)
        data.path = pdb_path = os.path.join(self.args.pdb_dir, row.name[:2], row.name)
        data.info = row

        # load protein structure deposited in PDB
        try:
            pos, mask = pdb_to_npy(pdb_path, seqres=row.seqres)
            pos[~mask,0] = pdb_to_npy_interpolate(row.seqlen, mask, pos[mask,0])
            data['resi'].pos = pos[:, 0].astype(np.float32)
        except Exception as e:
            logger.warning(f"Error in loading positions from PDB: {pdb_path}")
            data.skip = True
            return data

        # load OmegaFold embeddings
        embeddings_name = row.__getattr__(self.args.embeddings_key)
        embeddings_path = os.path.join(self.args.embeddings_dir, embeddings_name[:2], embeddings_name) + '.' + self.embeddings_suffix
        # save embeddings to 'data'
        try:
            embeddings_dict = dict(np.load(embeddings_path)) 
            node_repr, edge_repr = embeddings_dict['node_repr'], embeddings_dict['edge_repr']
            
            # check the tensor shape
            if node_repr.shape[0] != data['resi'].num_nodes:
                raise ValueError("LM dim error when reading embeddings")
            data['resi'].node_attr = torch.tensor(node_repr)
            edge_repr = torch.tensor(edge_repr)
            src, dst = data['resi'].edge_index[0], data['resi'].edge_index[1]
            data['resi'].edge_attr_ = torch.cat([edge_repr[src, dst], edge_repr[dst, src]], -1)
        except Exception as e:
            logger.warning(f"Error loading embeddings: {embeddings_path}")
            data.skip = True
            return data
        
        return data

    def len(self):
        return len(self.split)

    
def RGDM_get_loader(args, splits, mode='train', shuffle=True):

    # Modify splits as needed
    try:
        split = splits[splits.split == mode]
    except Exception as e:
        split = splits
        logger.warning("Not splitting based on split")
    if args.limit_mols:
        split = split[:args.limit_mols]
    if 'seqlen' not in split.columns:
        split['seqlen'] = [len(s) for s in split.seqres]
    split = split[split.seqlen <= args.max_len]
    
    transform = RGDiffusionTrainer(args)
    
    dataset = ResidueDataset(args, split, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=0 # args.num_workers
    )
    
    # log information
    logger.info(f"Initialized {mode if mode else ''} loader with {len(dataset)} entries")

    return loader


def DDPM_get_loader(args, splits, mode='train', shuffle=True):

    # Modify splits as needed
    try:
        split = splits[splits.split == mode]
    except Exception as e:
        split = splits
        logger.warning("Not splitting based on split")
    if args.limit_mols:
        split = split[:args.limit_mols]
    if 'seqlen' not in split.columns:
        split['seqlen'] = [len(s) for s in split.seqres]
    split = split[split.seqlen <= args.max_len]
    
    # 'loader' loads data in ResidueDataset
    # In each step, 'transform.__call__()' is applied to data
    transform = DDPM_DiffusionTrainer(args)
    dataset = ResidueDataset(args, split, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=args.num_workers
    )
    
    # log information
    logger.info(f"Initialized {mode if mode else ''} loader with {len(dataset)} entries")

    return loader


def get_args_suffix(arg_keys, args):
    cache_name = []
    for k in arg_keys: cache_name.extend([k, args.__dict__[k]])
    return '.'.join(map(str, cache_name))


def get_dense_edges(n):
    atom_ids = np.arange(n)
    src, dst = np.repeat(atom_ids, n), np.tile(atom_ids, n)
    mask = src != dst; src, dst = src[mask], dst[mask]
    edge_idx = np.stack([src, dst])
    return torch.tensor(edge_idx)
