import copy
import os
import torch
import shutil
import warnings
warnings.filterwarnings("ignore")

import time
from argparse import ArgumentParser, Namespace, FileType
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
import scipy
from Bio.PDB import PDBParser

from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, AddHs
from rdkit import Chem

import torch
# Set torch threads based on num_workers - will be updated after args parsing
torch.multiprocessing.set_sharing_strategy('file_system')



from torch_geometric.loader import DataLoader


from datasets.process_mols import read_molecule, generate_conformer, write_mol_with_coords
from datasets.pdbbind import PDBBind,PDBBindScoring
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule, set_time
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import LigandToPDB, modify_pdb, receptor_to_pdb, save_protein
from utils.clash import compute_side_chain_metrics
# from utils.relax import openmm_relax
from tqdm import tqdm
import datetime
from contextlib import contextmanager

from multiprocessing import Pool as ThreadPool

import random
import pickle
# pool = ThreadPool(8)

@contextmanager
def Timer(title):
    'timing function'
    t0 = datetime.datetime.now()
    yield
    print("%s - done in %is"%(title, (datetime.datetime.now() - t0).seconds))
    return None

RDLogger.DisableLog('rdApp.*')
import yaml
parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path and --ligand parameters')
parser.add_argument('--protein_path', type=str, default='data/dummy_data/1a0q_protein.pdb', help='Path to the protein .pdb file')
parser.add_argument('--ligand', type=str, default='COc(cc1)ccc1C#N', help='Either a SMILES string or the path to a molecule file that rdkit can read')
parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
parser.add_argument('--esm_embeddings_path', type=str, default='data/esm2_output', help='If this is set then the LM embeddings at that path will be used for the receptor features')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')
parser.add_argument('--savings_per_complex', type=int, default=1, help='Number of samples to save')
parser.add_argument('--seed', type=int, default=42, help='Number of samples to generate')

parser.add_argument('--model_dir', type=str, default='workdir/paper_score_model', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')

parser.add_argument('--batch_size', type=int, default=32, help='Number of complexes loaded at once')
parser.add_argument('--sample_batch_size', type=int, default=32, help='Number of complex-sample pairs passed to the model simultaneously')
parser.add_argument('--cache_path', type=str, default='data/cache', help='Folder from where to load/restore cached dataset')
parser.add_argument('--no_random', action='store_true', default=False, help='Use no randomness in reverse diffusion')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--ode', action='store_true', default=False, help='Use ODE formulation for inference')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for creating the dataset')
parser.add_argument('--sigma_schedule', type=str, default='expbeta', help='')
parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')
parser.add_argument('--keep_local_structures', action='store_true', default=False, help='Keeps the local structure when specifying an input with 3D coordinates instead of generating them with RDKit')
parser.add_argument('--protein_dynamic', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--relax', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--use_existing_cache', action='store_true', default=False, help='Use existing cache file, if they exist.')
parser.add_argument('--device', type=int, default=0, help='GPU device to use')


args = parser.parse_args()

# Set torch threads to match num_workers for maximum parallelism
# Note: This may cause oversubscription but can be faster for GPU-bound workloads
import multiprocessing
import os

allocated_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
torch_threads = args.num_workers  # Use same as data workers for maximum parallelism
torch.set_num_threads(torch_threads)
print(f"[INFO] Using {args.num_workers} workers and {torch_threads} torch threads (allocated cores: {allocated_cores})")

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

Seed_everything(seed=args.seed)
if args.config:
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value

os.makedirs(args.out_dir, exist_ok=True)

with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))

if args.confidence_model_dir is not None:
    with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))

# Select computation device – respect the --device CLI flag
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(args.device)
        device = torch.device(f'cuda:{args.device}')
        print(f"[INFO] Using GPU device {args.device}: {torch.cuda.get_device_name(device)}")
    except Exception as e:
        print(f"[WARNING] Could not set CUDA device {args.device} ({e}). Falling back to default CUDA device.")
        device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("[INFO] CUDA not available – running on CPU.")

if args.protein_ligand_csv is not None:
    df = pd.read_csv(args.protein_ligand_csv)
    # df = df[:10]
    if 'crystal_protein_path' not in df.columns:
        df['crystal_protein_path'] = df['protein_path']

    protein_path_list = df['protein_path'].tolist()
    ligand_descriptions = df['ligand'].tolist()

    # Preserve provided mutant names if present and valid; otherwise fall back to autogenerated idx_X labels
    print(f"[DEBUG] CSV has {len(df)} rows")
    print(f"[DEBUG] 'name' column exists: {'name' in df.columns}")
    if 'name' in df.columns:
        print(f"[DEBUG] Null names: {df['name'].isnull().sum()}")
        print(f"[DEBUG] Unique names: {df['name'].nunique()}")
        print(f"[DEBUG] Sample names: {df['name'].head().tolist()}")
    
    if 'name' not in df.columns or df['name'].isnull().any():
        print("[DEBUG] Generating idx_ names because 'name' column missing or has nulls")
        df['name'] = [f'idx_{i}' for i in range(df.shape[0])]
    else:
        print("[DEBUG] Using provided mutant names from CSV")
    name_list = df['name'].tolist()
else:
    protein_path_list = [args.protein_path]
    ligand_descriptions = [args.ligand]

test_dataset = PDBBindScoring(transform=None, root='', name_list=name_list, protein_path_list=protein_path_list, ligand_descriptions=ligand_descriptions,
                       receptor_radius=score_model_args.receptor_radius, cache_path=args.cache_path,
                       remove_hs=score_model_args.remove_hs, max_lig_size=None,
                       c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors, matching=False, keep_original=False,
                       popsize=score_model_args.matching_popsize, maxiter=score_model_args.matching_maxiter,center_ligand=True,
                       all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                       atom_max_neighbors=score_model_args.atom_max_neighbors,
                       esm_embeddings_path= args.esm_embeddings_path if score_model_args.esm_embeddings_path is not None else None,
                       require_ligand=True,require_receptor=True, num_workers=args.num_workers, keep_local_structures=args.keep_local_structures, use_existing_cache=args.use_existing_cache)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
# Load the checkpoint directly onto the target device (GPU if available) to avoid subsequent device mismatches
state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=device)
model.load_state_dict(state_dict, strict=True)
# Ensure every parameter/buffer is on the desired device
model = model.to(device)
model.eval()

if args.confidence_model_dir is not None:
    if confidence_args.transfer_weights:
        with open(f'{confidence_args.original_model_dir}/model_parameters.yml') as f:
            confidence_model_args = Namespace(**yaml.full_load(f))
    else:
        confidence_model_args = confidence_args
    confidence_model = get_model(confidence_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
    state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=device)
    confidence_model.load_state_dict(state_dict, strict=True)
    confidence_model = confidence_model.to(device)
    confidence_model.eval()
else:
    confidence_model = None
    confidence_args = None
    confidence_model_args = None

tr_schedule = get_t_schedule(inference_steps=args.inference_steps)
rot_schedule = tr_schedule
tor_schedule = tr_schedule
res_tr_schedule = tr_schedule
res_rot_schedule = tr_schedule
res_chi_schedule = tr_schedule
print('common t schedule', tr_schedule)

failures, skipped, confidences_list, names_list, run_times, min_self_distances_list = 0, 0, [], [], [], []
N = args.samples_per_complex
print('Size of test dataset: ', len(test_dataset))

affinity_pred = {}
all_complete_affinity = []
print(f"[DEBUG] Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
for idx, orig_complex_graph_batch in tqdm(enumerate(test_loader)):
    # Convert loader output (Batch) to individual complex graphs
    complex_graphs = (orig_complex_graph_batch.to_data_list()
                      if hasattr(orig_complex_graph_batch, 'to_data_list')
                      else [orig_complex_graph_batch])

    # ------------------------------------------------------------------
    # Process each complex individually (mirrors logic in inference.py)
    # ------------------------------------------------------------------
    for cg in complex_graphs:
        complex_name = (cg.name[0] if hasattr(cg.name, '__len__') and not isinstance(cg.name, str)
                        else cg.name)

        # Replicate the complex N (=samples_per_complex) times
        data_list = [cg.clone() for _ in range(N)]

        # Randomise initial positions in-place
        randomize_position(
            data_list,
            score_model_args.no_torsion,
            args.no_random,
            score_model_args.tr_sigma_max,
            score_model_args.rot_sigma_max,
            score_model_args.tor_sigma_max,
            score_model_args.res_tr_sigma_max,
            score_model_args.res_rot_sigma_max,
        )

        steps = args.actual_steps if args.actual_steps is not None else args.inference_steps
        try:
            final_samples, _, lddt_pred_all, affinity_pred_all = sampling(
                data_list=data_list,
                model=model,
                inference_steps=steps,
                tr_schedule=tr_schedule,
                rot_schedule=rot_schedule,
                tor_schedule=tor_schedule,
                res_tr_schedule=res_tr_schedule,
                res_rot_schedule=res_rot_schedule,
                res_chi_schedule=res_chi_schedule,
                device=device,
                t_to_sigma=t_to_sigma,
                model_args=score_model_args,
                no_random=args.no_random,
                ode=args.ode,
                visualization_list=None,
                batch_size=args.sample_batch_size,
                no_final_step_noise=args.no_final_step_noise,
                protein_dynamic=args.protein_dynamic,
            )
        except Exception as e:
            print("Failed on", complex_name, e)
            traceback.print_exc()
            try:
                print(f"[DEBUG] ligand.pos device: {cg['ligand'].pos.device}, receptor.pos device: {cg['receptor'].pos.device}")
            except Exception as dev_e:
                print(f"[DEBUG] Could not print device info: {dev_e}")
            failures += 1
            continue

        # lddt_pred_all / affinity_pred_all have shape (N,)
        lddt_preds = lddt_pred_all.view(-1).cpu().numpy()
        affinity_preds = affinity_pred_all.view(-1).cpu().numpy()
        final_affinity_pred = np.minimum((affinity_preds * lddt_preds).sum() / (lddt_preds.sum() + 1e-12), 15.0)

        affinity_pred[complex_name] = final_affinity_pred

        # Optional detailed per-replicate record (for downstream analysis)
        complete_affinity = pd.DataFrame({
            "name": [complex_name] * N,
            "lddt": lddt_preds,
            "affinity": affinity_preds,
        })
        all_complete_affinity.append(complete_affinity)
        names_list.append(complex_name)

        # ------------------------------------------------------------------
        # NEW: clash-score ranking & per-rank file export (parity with single run)
        # ------------------------------------------------------------------
        try:
            import copy as _cpy
            # 1) construct output directory equivalent to single-inference
            # Use the actual complex (mutant) name for the directory, sanitized for filesystem safety
            safe_name = str(complex_name).replace("/", "-")
            write_dir = os.path.join(args.out_dir, safe_name)
            os.makedirs(write_dir, exist_ok=True)

            # 2) copy original input files for reference (if available)
            if 'protein_path' in df.columns:
                try:
                    src_protein = df.loc[df['name'] == complex_name, 'protein_path'].values[0]
                    if os.path.isfile(src_protein):
                        shutil.copy2(src_protein, write_dir)
                except Exception:
                    pass
            if 'ligand' in df.columns:
                try:
                    src_ligand = df.loc[df['name'] == complex_name, 'ligand'].values[0]
                    if os.path.isfile(src_ligand):
                        shutil.copy2(src_ligand, write_dir)
                except Exception:
                    pass

            # 3) prepare convenience handles
            receptor_pdb = cg.rec_pdb[0] if hasattr(cg, 'rec_pdb') else None
            # cg.mol should already be a single RDKit Mol object; avoid erroneous indexing that triggers
            # "'Mol' object is not subscriptable" when it is not a list.
            lig_rdkit    = cg.mol        if hasattr(cg, 'mol') else None
            if receptor_pdb is None or lig_rdkit is None:
                raise ValueError("receptor_pdb or ligand mol missing – cannot write ranked files")
            pdb_or_cif   = receptor_pdb.get_full_id()[0]

            # 4) collect ligand coordinate sets (global frame)
            ligand_pos = np.asarray([
                sample['ligand'].pos.cpu().numpy() + cg.original_center.cpu().numpy()
                for sample in final_samples
            ])

            ligandFiles, pdbFiles, clash_scores = [], [], []
            max_rank = min(args.samples_per_complex, len(ligand_pos))
            for rank_idx in range(max_rank):
                # write temporary step1_* files first (will be renamed after re-ordering)
                mol_pred = _cpy.deepcopy(lig_rdkit)
                # If hydrogens were removed in the diffusion graph, the coordinate array
                # may contain fewer atoms than the full (hydrogen-included) RDKit Mol.
                # Strip hydrogens to keep counts consistent before writing coordinates.
                if mol_pred.GetNumAtoms() != ligand_pos[rank_idx].shape[0]:
                    mol_pred = RemoveHs(mol_pred, sanitize=False)
                lig_file = os.path.join(
                    write_dir,
                    f"step1_rank{rank_idx+1}_ligand_lddt{lddt_preds[rank_idx]:.2f}_affinity{affinity_preds[rank_idx]:.2f}.sdf",
                )
                write_mol_with_coords(mol_pred, ligand_pos[rank_idx], lig_file)

                new_rec = _cpy.deepcopy(receptor_pdb)
                if args.protein_dynamic:
                    modify_pdb(new_rec, final_samples[rank_idx])
                pdb_file = os.path.join(
                    write_dir,
                    f"step1_rank{rank_idx+1}_receptor_lddt{lddt_preds[rank_idx]:.2f}_affinity{affinity_preds[rank_idx]:.2f}.{pdb_or_cif}",
                )
                save_protein(new_rec, pdb_file)

                ligandFiles.append(lig_file)
                pdbFiles.append(pdb_file)
                try:
                    clash_scores.append(compute_side_chain_metrics(pdb_file, lig_file, verbose=False))
                except Exception:
                    clash_scores.append(999.0)  # large penalty if metric fails

            # 5) compute composite rank: lddt (descending) + clash/2 (ascending)
            # Length-safety: convert clash_scores to np.array before rankdata to avoid size mismatches
            clash_np = np.asarray(clash_scores)
            re_order = np.argsort(
                scipy.stats.rankdata(-lddt_preds[:max_rank]) +
                scipy.stats.rankdata(clash_np)/2.0
            )

            # 6) rename files to final rankX_* pattern
            for new_rank, old_idx in enumerate(re_order):
                os.rename(
                    ligandFiles[old_idx],
                    ligandFiles[old_idx].replace(f"step1_rank{old_idx+1}", f"rank{new_rank+1}")
                )
                os.rename(
                    pdbFiles[old_idx],
                    pdbFiles[old_idx].replace(f"step1_rank{old_idx+1}", f"rank{new_rank+1}")
                )
        except Exception as e:
            print(f"[WARNING] Clash-score ranking/export failed for {complex_name}: {e}")

        # ------------------------------------------------------------------
        # Optional: save final ligand–receptor complex as PDB files
        # ------------------------------------------------------------------
        if args.save_visualisation:
            try:
                from pathlib import Path
                import copy

                # Create a dedicated directory per complex
                complex_dir = Path(args.out_dir) / complex_name
                complex_dir.mkdir(parents=True, exist_ok=True)

                for sample_idx, sample_graph in enumerate(final_samples):
                    # Prefix: sample_0, sample_1, ...
                    prefix = complex_dir / f"sample_{sample_idx}"

                    # 1. Save ligand
                    # Ensure graph data is on CPU for PDB utilities
                    sample_graph_cpu = sample_graph.to('cpu') if hasattr(sample_graph, 'to') else sample_graph

                    ligand_writer = LigandToPDB(sample_graph_cpu.mol if hasattr(sample_graph_cpu, 'mol') else cg.mol)
                    # Shift ligand coordinates back to global frame using the same original_center applied to the receptor
                    ligand_coords_global = sample_graph_cpu['ligand'].pos + sample_graph_cpu.original_center
                    ligand_writer.add(ligand_coords_global, order=0)
                    # ligand_writer.write(f"{prefix}_ligand.pdb")

                    # 2. Save receptor (all atoms)
                    rec_pdb_mod = None
                    if hasattr(sample_graph_cpu, 'rec_pdb'):
                        try:
                            rec_pdb_mod = modify_pdb(copy.deepcopy(sample_graph_cpu.rec_pdb), sample_graph_cpu)
                        except Exception as e:
                            print(f"[WARNING] Could not prepare receptor PDB for {complex_name} sample {sample_idx}: {e}")

                    # 3. Write combined complex (receptor + ligand) like single_inference output
                    if rec_pdb_mod is not None:
                        try:
                            complex_path = f"{prefix}_complex.pdb"
                            # Write receptor first
                            save_protein(rec_pdb_mod, complex_path, ca_only=False)
                            # Append ligand MODEL blocks
                            ligand_block = ligand_writer.write()
                            with open(complex_path, "a") as cf:
                                cf.write(ligand_block)
                        except Exception as e:
                            print(f"[WARNING] Could not write complex PDB for {complex_name} sample {sample_idx}: {e}")
            except Exception as e:
                print(f"[WARNING] Could not write PDB files for {complex_name}: {e}")

    # End per-complex loop

# ----------------------------------------------------------------------
# Post-processing & result serialization (unchanged)
# ----------------------------------------------------------------------

affinity_pred_df = pd.DataFrame({'name':list(affinity_pred.keys()),'affinity':list(affinity_pred.values())})
affinity_pred_df.to_csv(f'{args.out_dir}/affinity_prediction.csv',index=False)

# Only concatenate if we have results to avoid "No objects to concatenate" error
if all_complete_affinity:
    pd.concat(all_complete_affinity).to_csv(f'{args.out_dir}/complete_affinity_prediction.csv',index=False)
else:
    print("Warning: No complete affinity predictions to save - all complexes failed processing.")
    # Create empty file with proper headers
    empty_df = pd.DataFrame(columns=['name', 'lddt', 'affinity'])
    empty_df.to_csv(f'{args.out_dir}/complete_affinity_prediction.csv',index=False)

# min_self_distances = np.array(min_self_distances_list)
# confidences = np.array(confidences_list)
# names = np.array(names_list)
# run_times = np.array(run_times)
# np.save(f'{args.out_dir}/min_self_distances.npy', min_self_distances)
# np.save(f'{args.out_dir}/confidences.npy', confidences)
# np.save(f'{args.out_dir}/run_times.npy', run_times)
# np.save(f'{args.out_dir}/complex_names.npy', np.array(names))

print(f'Results are in {args.out_dir}')
