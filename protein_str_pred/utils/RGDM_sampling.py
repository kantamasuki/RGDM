import copy, torch, os
from .pdb import pdb_to_npy, pdb_to_npy_interpolate, PDBFile, tmscore
from .logging import get_logger
from diffusion.RGDM_diffusion import RGDiffusionSampler
import pandas as pd
from tqdm import tqdm
import numpy as np

logger = get_logger(__name__)


def RGDM_sampling_epoch(args, model, dataset, device='cpu', pdbs=False, elbo=None, model_dir=None, inf_name=None, csv_path=None):
    
    model.eval()
    
    # N is the maximum number of proteins to perform sampling
    N = len(dataset)
    # num_samples is the number of predictions for each protein
    # We choose the best structure among predicted structures based on RMSD score. 
    num_samples = args.num_samples
    # Path to directory to save predicted samples
    if not os.path.exists(os.path.join(model_dir, inf_name)):
        os.mkdir(os.path.join(model_dir, inf_name))
    # Make a file to write proteins that could not be generated.
    filename = os.path.join(model_dir, inf_name, 'failed_protein.txt')
    with open(filename, "w", encoding="utf-8") as f:
        f.write("proteins that could not be generated\n")
    
    # Run sampling
    logger.info(f"Sampling start, #{N}-proteins, #{num_samples}-samples for each protein")
    pred_fail_count = 0
    for i in tqdm(range(N), desc="proteins"):
        
        # Load i-th data in dataset without performing the transformer.
        # (See ./dataset.py for the definition of transformer)
        data_ = dataset.get(i)
        if data_.skip:
            logger.warning('Skipping inference mol')
            continue

        # Modify/Add protein information to data_
        # data_.pdb is a dataframe to save sampled structure
        molseq = data_.info.seqres
        seqlen = data_['resi'].num_nodes
        try:
            data_['resi'].pos = torch.from_numpy(data_['resi'].pos)
        except:
            data_['resi'].pos = torch.zeros(data_['resi'].num_nodes, 3)
        data_.pdb = PDBFile(molseq)
        data_ = data_.to(device)

        # Define the sampler
        Sampler = RGDiffusionSampler(args, seqlen).to(device)

        # Predict protein structures
        datas = []
        for j in range(num_samples):
            data = copy.deepcopy(data_)
            
            # This block performs the sampling
            try:
                data.Y, data.lnP = Sampler(model, data, elbo)
                data.copy = j; datas.append(data)

            except Exception as e:
                if type(e) is KeyboardInterrupt: raise e
                logger.error('Skipping inference mol due to exception ' + str(e))
                raise e
        # Calculate score for each sample and pick up the best one
        best_sample = -1
        rmsd = np.inf
        working_path = os.path.join(model_dir, inf_name, 'work.pdb')
        for i in range(len(datas)):
            samp = datas[i]
            samp.pdb.clear().add(samp.Y).write(working_path)
            try:
                pos, _ = pdb_to_npy(working_path, seqres=molseq)
                pos = pos[:, 0].astype(np.float32)
                res = tmscore(samp.path, pos, molseq)
                samp.__dict__.update(res)
                if samp.rmsd < rmsd:
                    rmsd = samp.rmsd
                    best_sample = i
            except:
                continue

        # If all samples fail, write the protein name to the log file 
        if best_sample == -1:
            filename = os.path.join(model_dir, inf_name, 'failed_protein.txt')
            with open(filename, "a", encoding="utf-8") as f:
                f.write(datas[0].path.split('/')[-1] + ".0.pdb" + "\n")
            continue
        else:
            # save structure
            samp = datas[best_sample]
            saving_path = os.path.join(
                model_dir, inf_name, samp.path.split('/')[-1] + ".0.pdb")
            samp.pdb.clear().add(samp.Y).write(saving_path)
            logs = {
                'path': [samp.path],
                'copy': [samp.copy],
                'lnP': [samp.lnP],
                'rmsd': [samp.rmsd],
                'gdt_ts': [samp.gdt_ts],
                'gdt_ha': [samp.gdt_ha],
                'tm': [samp.tm],
                }
            # save score in CSV-format
            pd_logs = pd.DataFrame(logs)
            if os.path.exists(csv_path):
                pd_logs.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                pd_logs.to_csv(csv_path, index=False)
