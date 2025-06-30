#!/usr/bin/env python3
"""
Batch DynamicBind Processing Script for Multiple Protein Mutants vs Same Ligand

This script optimizes DynamicBind for processing thousands of protein mutants
against the same ligand by:
1. Batch ESM embedding preparation for all proteins
2. Batch ESM extraction 
3. High-throughput screening in batches
4. Parallel relaxation

Usage:
python run_batch_protein_screening.py /path/to/mutants_folder ligand.csv --header batch_screen
"""

import numpy as np
import pandas as pd
import os
import sys
import subprocess
import glob
from datetime import datetime
import logging
import argparse
from pathlib import Path
import shutil
from multiprocessing import Pool
import time
from tqdm import tqdm

def setup_cuda_environment(device_id):
    """Setup CUDA environment with proper library paths for HPC systems"""
    cuda_env = os.environ.copy()
    cuda_env['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    # Common CUDA library paths on different HPC systems
    cuda_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-12.1/lib64',
        '/usr/local/cuda-12/lib64',
        '/usr/local/cuda-11/lib64',
        '/opt/cuda/lib64',
        '/apps/cuda/lib64',
        '/sw/cuda/lib64',
        '/cluster/cuda/lib64'
    ]
    
    # Check for CUDA installation via environment modules
    cuda_home = cuda_env.get('CUDA_HOME')
    if cuda_home:
        cuda_paths.insert(0, os.path.join(cuda_home, 'lib64'))
    
    # Check for existing LD_LIBRARY_PATH
    existing_ld_path = cuda_env.get('LD_LIBRARY_PATH', '')
    
    # Find existing CUDA paths
    valid_cuda_paths = [p for p in cuda_paths if os.path.exists(p)]
    
    # Combine all paths
    if valid_cuda_paths:
        if existing_ld_path:
            cuda_env['LD_LIBRARY_PATH'] = ':'.join(valid_cuda_paths + [existing_ld_path])
        else:
            cuda_env['LD_LIBRARY_PATH'] = ':'.join(valid_cuda_paths)
        print(f"Found CUDA libraries in: {valid_cuda_paths}")
    else:
        print("Warning: No CUDA library paths found. GPU may not work.")
    
    return cuda_env

def do(cmd, get=False, show=True):
    """Execute shell command"""
    if get:
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0].decode()
        if show:
            print(out, end="")
        return out
    else:
        return subprocess.Popen(cmd, shell=True).wait()

def clean_single_pdb(args_tuple):
    """Clean a single PDB file - module level function for multiprocessing"""
    pdb_file, cleaned_dir, relax_python, script_folder = args_tuple
    cleaned_file = os.path.join(cleaned_dir, f"cleaned_{os.path.basename(pdb_file)}")
    lock_file = f"{cleaned_file}.lock"
    
    # Skip if already cleaned and valid
    if os.path.exists(cleaned_file) and os.path.getsize(cleaned_file) > 0:
        return pdb_file, cleaned_file
    
    # Use lock file to prevent race conditions
    try:
        # Try to create lock file atomically
        with open(lock_file, 'x') as f:
            f.write(f"Cleaning by PID {os.getpid()}")
        
        # Double-check after acquiring lock (another worker might have finished)
        if os.path.exists(cleaned_file) and os.path.getsize(cleaned_file) > 0:
            os.remove(lock_file)
            return pdb_file, cleaned_file
        
        # Actually clean the file
        cmd = f"{relax_python} {script_folder}/clean_pdb.py {pdb_file} {cleaned_file}"
        result = do(cmd, show=False)
        
        # Remove lock file
        os.remove(lock_file)
        
        if result == 0:
            return pdb_file, cleaned_file
        else:
            print(f"Warning: Failed to clean {pdb_file}")
            return pdb_file, None
            
    except FileExistsError:
        # Another worker is already cleaning this file
        print(f"Skipping {os.path.basename(pdb_file)} - already being cleaned by another worker")
        
        # Wait a bit and check if cleaning completed
        import time
        for _ in range(60):  # Wait up to 60 seconds
            time.sleep(1)
            if os.path.exists(cleaned_file) and os.path.getsize(cleaned_file) > 0:
                return pdb_file, cleaned_file
        
        # If still not done, return None (failed)
        print(f"Warning: Timeout waiting for {pdb_file} to be cleaned")
        return pdb_file, None

def clean_pdb_batch(pdb_files, cleaned_dir, relax_python, script_folder, max_workers=10):
    """Clean multiple PDB files in parallel"""
    print(f"Cleaning {len(pdb_files)} PDB files with {max_workers} workers...")
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Prepare arguments for each PDB file
    args_list = [(pdb_file, cleaned_dir, relax_python, script_folder) for pdb_file in pdb_files]
    
    print(f"Starting parallel cleaning with {max_workers} workers...")
    
    results = []
    with Pool(max_workers) as pool:
        # Use imap for progress tracking
        with tqdm(total=len(args_list), desc="Cleaning PDBs") as pbar:
            for result in pool.imap(clean_single_pdb, args_list):
                results.append(result)
                pbar.update(1)
    
    # Filter successful cleanings
    successful_cleanings = {orig: cleaned for orig, cleaned in results if cleaned is not None}
    print(f"Successfully cleaned {len(successful_cleanings)}/{len(pdb_files)} files")
    
    return successful_cleanings

def create_batch_csv(cleaned_pdb_dict, ligand_info, output_csv):
    """Create a CSV file with all protein-ligand combinations"""
    data = []
    
    for original_pdb, cleaned_pdb in cleaned_pdb_dict.items():
        # Get protein name from filename
        protein_name = os.path.splitext(os.path.basename(original_pdb))[0]
        # Use the protein base name as the unique mutant identifier
        data.append({
            'protein_path': cleaned_pdb,
            'ligand': ligand_info,
            'name': protein_name,  # keeps meaningful mutant name
            'original_protein': original_pdb
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created batch CSV with {len(df)} protein-ligand pairs: {output_csv}")
    
    return df

def run_batch_esm_embedding(batch_csv, output_fasta, python, script_folder):
    """Run ESM embedding preparation for all proteins at once"""
    print("Preparing ESM embeddings for all proteins...")
    cmd = f"{python} {script_folder}/datasets/esm_embedding_preparation.py --protein_ligand_csv {batch_csv} --out_file {output_fasta}"
    result = do(cmd)
    if result != 0:
        raise RuntimeError("ESM embedding preparation failed")
    print(f"ESM embedding preparation complete: {output_fasta}")

def run_batch_esm_extraction(input_fasta, output_dir, python, script_folder, device):
    """Run ESM extraction for all proteins at once"""
    print("Extracting ESM embeddings for all proteins...")
    print(f"GPU Debug: Setting CUDA_VISIBLE_DEVICES={device}")
    
    # Set up CUDA environment
    cuda_env = setup_cuda_environment(device)
    
    cmd = f"{python} {script_folder}/esm/scripts/extract.py esm2_t33_650M_UR50D {input_fasta} {output_dir} --repr_layers 33 --include per_tok --truncation_seq_length 10000 --model_dir {script_folder}/esm_models"
    print(f"GPU Debug: ESM command = {cmd}")
    print(f"GPU Debug: LD_LIBRARY_PATH = {cuda_env.get('LD_LIBRARY_PATH', 'Not set')}")
    
    result = subprocess.run(cmd, shell=True, env=cuda_env).returncode
    if result != 0:
        raise RuntimeError("ESM extraction failed")
    print(f"ESM extraction complete: {output_dir}")

def run_batch_screening(batch_csv, esm_dir, output_dir, args, python, script_folder, model_workdir, ckpt):
    """Run high-throughput screening for all protein-ligand pairs"""
    print(f"Running batch screening for all {len(pd.read_csv(batch_csv))} protein-ligand pairs...")
    
    # GPU debugging
    print(f"GPU Debug: args.device = {args.device} (type: {type(args.device)})")
    print(f"GPU Debug: Setting CUDA_VISIBLE_DEVICES={args.device}")
    
    # Set up CUDA environment
    cuda_env = setup_cuda_environment(args.device)
    
    # Ensure paths are absolute to avoid issues when subprocess changes cwd
    batch_csv = os.path.abspath(batch_csv)
    esm_dir = os.path.abspath(esm_dir)
    output_dir_abs = os.path.abspath(output_dir)
    
    protein_dynamic = "--protein_dynamic" if not args.rigid_protein else ""
    
    cmd = f"{python} {script_folder}/screening.py --seed {args.seed} --device {args.device} --ckpt {ckpt} {protein_dynamic}"
    cmd += f" --save_visualisation --model_dir {model_workdir} --protein_ligand_csv {batch_csv}"
    cmd += f" --esm_embeddings_path {esm_dir} --out_dir {output_dir_abs}"
    cmd += f" --inference_steps {args.inference_steps} --samples_per_complex {args.samples_per_complex}"
    cmd += f" --savings_per_complex {args.savings_per_complex} --batch_size {args.batch_size} --sample_batch_size {args.sample_batch_size}"
    cmd += f" --num_workers {args.num_workers} --actual_steps {args.inference_steps} --no_final_step_noise"
    cmd += f" --cache_path {args.cache_path} --use_existing_cache"  # Add cache path and enable cache usage
    
    print(f"GPU Debug: Screening command = {cmd}")
    print(f"GPU Debug: LD_LIBRARY_PATH = {cuda_env.get('LD_LIBRARY_PATH', 'Not set')}")
    print(f"Cache Debug: Using cache path = {args.cache_path} (will become {args.cache_path}_torsion)")
    
    result = subprocess.run(cmd, shell=True, env=cuda_env).returncode
    if result != 0:
        raise RuntimeError("Batch screening failed")
        
    print("Batch screening complete")

def run_batch_relaxation(output_dir, args, relax_python, script_folder):
    """Run relaxation for all results"""
    if not args.no_relax:
        print("Running batch relaxation...")
        print(f"GPU Debug: Setting CUDA_VISIBLE_DEVICES={args.device} for relaxation")
        
        # Set up CUDA environment
        cuda_env = setup_cuda_environment(args.device)
        
        cmd = f"{relax_python} {script_folder}/relax_final.py --results_path {output_dir} --samples_per_complex {args.samples_per_complex} --num_workers {args.num_workers}"
        print(f"GPU Debug: Relaxation command = {cmd}")
        print(f"GPU Debug: LD_LIBRARY_PATH = {cuda_env.get('LD_LIBRARY_PATH', 'Not set')}")
        
        result = subprocess.run(cmd, shell=True, env=cuda_env).returncode
        if result != 0:
            print("Warning: Batch relaxation failed")
        else:
            print("Batch relaxation complete")

def create_final_summary_csv(combined_results, results_df, batch_df, final_csv_path):
    """Create a final consolidated CSV with id and DB_affinity columns"""
    try:
        # Use DynamicBind's own calculated affinities (from affinity_prediction.csv)
        # This file contains DynamicBind's processed affinity per protein (not individual samples)
        if combined_results is not None:
            final_results = combined_results
        elif results_df is not None:
            final_results = results_df
        else:
            print("Warning: No results available to create final CSV")
            return
        
        # Convert affinity column to numeric to handle potential object dtypes
        if 'affinity' in final_results.columns:
            final_results['affinity'] = pd.to_numeric(final_results['affinity'], errors='coerce')
        
        # Extract mutant names from original batch data
        name_mapping = {}
        for _, row in batch_df.iterrows():
            # Extract mutant name from original protein path
            mutant_name = os.path.basename(row['original_protein']).replace('.pdb', '')
            name_mapping[row['name']] = mutant_name
        
        # Create final CSV using DynamicBind's calculated affinities directly
        final_data = []
        
        for _, row in final_results.iterrows():
            protein_name = row['name']
            mutant_name = name_mapping.get(protein_name, protein_name)
            # Use DynamicBind's affinity calculation directly (no "best" selection)
            db_affinity = row['affinity']
            
            # Handle NaN or invalid affinities
            if pd.isna(db_affinity):
                db_affinity = -999.0  # Use a sentinel value for failed predictions
            
            final_data.append({
                'id': mutant_name,
                'DB_affinity': db_affinity
            })
        
        # Create and save DataFrame
        final_df = pd.DataFrame(final_data)
        
        # Only sort by affinity if we have valid (non-sentinel) values
        valid_affinities = final_df[final_df['DB_affinity'] != -999.0]
        if len(valid_affinities) > 0:
            final_df = final_df.sort_values('DB_affinity', ascending=False)  # Sort by DynamicBind affinity
            highest_affinity = final_df[final_df['DB_affinity'] != -999.0]['DB_affinity'].max()
            average_affinity = final_df[final_df['DB_affinity'] != -999.0]['DB_affinity'].mean()
        else:
            highest_affinity = "N/A (all predictions failed)"
            average_affinity = "N/A (all predictions failed)"
        
        final_df.to_csv(final_csv_path, index=False)
        
        print(f"\n=== Final Results Summary ===")
        print(f"Created consolidated CSV: {final_csv_path}")
        print(f"Total mutants: {len(final_df)}")
        print(f"Valid predictions: {len(valid_affinities)}/{len(final_df)}")
        print(f"Failed predictions: {len(final_df) - len(valid_affinities)}/{len(final_df)}")
        print(f"Using DynamicBind's calculated affinities (from affinity_prediction.csv)")
        print(f"Highest DB affinity: {highest_affinity}")
        print(f"Average DB affinity: {average_affinity}")
        
    except Exception as e:
        print(f"Error creating final CSV: {e}")
        # Create a fallback simple version
        if results_df is not None and len(results_df) > 0:
            try:
                # Convert affinity to numeric if it's not already
                results_df['affinity'] = pd.to_numeric(results_df['affinity'], errors='coerce')
                
                simple_df = pd.DataFrame({
                    'id': results_df['name'],
                    'DB_affinity': results_df['affinity'].fillna(-999.0)  # Fill NaN with sentinel value
                })
                simple_df.to_csv(final_csv_path, index=False)
                print(f"Created fallback CSV: {final_csv_path}")
            except Exception as fallback_e:
                print(f"Fallback CSV creation also failed: {fallback_e}")
                # Create empty CSV with proper headers as last resort
                empty_df = pd.DataFrame(columns=['id', 'DB_affinity'])
                empty_df.to_csv(final_csv_path, index=False)
                print(f"Created empty CSV with headers: {final_csv_path}")
        else:
            # Create empty CSV with proper headers
            empty_df = pd.DataFrame(columns=['id', 'DB_affinity'])
            empty_df.to_csv(final_csv_path, index=False)
            print(f"Created empty CSV (no results available): {final_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch DynamicBind processing for multiple protein mutants vs same ligand")
    
    # Required arguments
    parser.add_argument('mutants_folder', type=str, help='Folder containing protein mutant PDB files')
    parser.add_argument('ligand_csv', type=str, help='CSV file containing ligand information (with "ligand" column)')
    
    # Optional arguments
    parser.add_argument('--header', type=str, default='batch_mutants', help='Name for the batch run')
    parser.add_argument('--results', type=str, default='results', help='Results folder')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--samples_per_complex', type=int, default=10, help='Samples per complex (reduced for speed)')
    parser.add_argument('--savings_per_complex', type=int, default=10, help='Samples to save per complex')
    parser.add_argument('--inference_steps', type=int, default=40, help='Inference steps (reduced for speed)')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of complexes loaded at once in screening')
    parser.add_argument('--sample_batch_size', type=int, default=32, help='Internal batch size of complex-sample pairs passed to model inside screening')
    parser.add_argument('--num_workers', type=int, default=20, help='Workers for relaxation')
    parser.add_argument('--max_clean_workers', type=int, default=10, help='Workers for parallel PDB cleaning')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model', type=int, default=1, help='Model version')
    parser.add_argument('--no_relax', action='store_true', help='Skip relaxation step')
    parser.add_argument('--rigid_protein', action='store_true', help='Use rigid protein')
    parser.add_argument('--ligand_column', type=str, default='ligand', help='Column name for ligand in CSV')
    parser.add_argument('--cache_path', type=str, default='data/cache', help='Cache directory (DynamicBind will append _torsion automatically)')
    
    # Python environments
    parser.add_argument('--python', type=str, default='/nfs/hpc/share/nguyhun2/.conda/envs/dynamicbind/bin/python', help='DynamicBind python')
    parser.add_argument('--relax_python', type=str, default='/nfs/hpc/share/nguyhun2/.conda/envs/relax/bin/python', help='Relax python')
    
    # Processing options
    parser.add_argument('--skip_cleaning', action='store_true', help='Skip PDB cleaning (if already cleaned)')
    parser.add_argument('--skip_esm', action='store_true', help='Skip ESM steps (if already done)')
    parser.add_argument('--only_esm', action='store_true', help='Only run ESM steps')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # GPU Environment Debug Check
    print("=== GPU Environment Debug ===")
    print(f"Requested CUDA device: {args.device}")
    print(f"CUDA_VISIBLE_DEVICES from environment: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Set up CUDA environment for testing
    cuda_env = setup_cuda_environment(args.device)
    
    # Test PyTorch CUDA availability in the DynamicBind environment
    cuda_test_cmd = f"{args.python} -c \"import torch; print(f'PyTorch CUDA available: {{torch.cuda.is_available()}}'); print(f'PyTorch CUDA devices: {{torch.cuda.device_count()}}'); print(f'Current device: {{torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}}'); print(f'CUDA runtime version: {{torch.version.cuda}}');\""
    print("Testing PyTorch CUDA in DynamicBind environment...")
    print(f"Using LD_LIBRARY_PATH: {cuda_env.get('LD_LIBRARY_PATH', 'Not set')}")
    cuda_result = subprocess.run(cuda_test_cmd, shell=True, env=cuda_env, capture_output=True, text=True)
    print(cuda_result.stdout)
    if cuda_result.stderr:
        print(f"STDERR: {cuda_result.stderr}")
    print("=== End GPU Debug ===\n")

    # Setup paths (static names for reusability)
    script_folder = os.path.dirname(os.path.realpath(__file__))
    work_dir = f"batch_work_{args.header}"
    
    logging.info(f"Starting batch processing.")
    logging.info(f"Processing configuration:")
    logging.info(f"• Batch size: {args.batch_size} complexes per batch")
    logging.info(f"• Auto-detection: Enabled (will reuse existing cleaned PDBs, ESM files, and cache)")
    logging.info(f"• Work directory: {work_dir} (reusable across runs)")
    os.makedirs(work_dir, exist_ok=True)
    
    # Find all PDB files in mutants folder
    pdb_pattern = os.path.join(args.mutants_folder, "*.pdb")
    pdb_files = glob.glob(pdb_pattern)
    
    if not pdb_files:
        raise ValueError(f"No PDB files found in {args.mutants_folder}")
    
    logging.info(f"Found {len(pdb_files)} PDB files to process")
    
    # Read ligand information
    ligand_df = pd.read_csv(args.ligand_csv)
    if args.ligand_column not in ligand_df.columns:
        raise ValueError(f"Column '{args.ligand_column}' not found in {args.ligand_csv}")
    
    # For now, use the first ligand (assuming same ligand for all)
    ligand_info = ligand_df[args.ligand_column].iloc[0]
    logging.info(f"Using ligand: {ligand_info}")
    
    # Setup model paths
    if args.model == 1:
        model_workdir = f"{script_folder}/workdir/big_score_model_sanyueqi_with_time"
        ckpt = "pro_ema_inference_epoch138_model.pt"
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    try:
        # Step 1: Clean PDB files in parallel (smart auto-detect with validation)
        cleaned_dir = os.path.join(work_dir, "cleaned_pdbs")
        logging.info("--- Step 1: PDB Cleaning ---")
        if args.skip_cleaning:
            logging.info("Skipping PDB cleaning (--skip_cleaning flag)...")
            cleaned_pdb_dict = {pdb: pdb for pdb in pdb_files}
        else:
            # Smart PDB cleaning detection with validation
            if os.path.exists(cleaned_dir):
                # Check if we have cleaned versions for all input PDBs
                cleaned_pdb_dict = {}
                missing_cleaned = []
                
                for pdb_file in pdb_files:
                    base_name = os.path.basename(pdb_file)
                    cleaned_file = os.path.join(cleaned_dir, f"cleaned_{base_name}")
                    if os.path.exists(cleaned_file) and os.path.getsize(cleaned_file) > 0:
                        cleaned_pdb_dict[pdb_file] = cleaned_file
                    else:
                        missing_cleaned.append(pdb_file)
                
                if len(missing_cleaned) == 0:
                    logging.info(f"✓ Auto-detected complete cleaned PDBs ({len(cleaned_pdb_dict)}/{len(pdb_files)}), reusing...")
                elif len(missing_cleaned) < len(pdb_files) * 0.1:  # Less than 10% missing
                    logging.warning(f"⚠ Mostly complete cleaned PDBs ({len(cleaned_pdb_dict)}/{len(pdb_files)}), cleaning {len(missing_cleaned)} missing files...")
                    # Clean only the missing files
                    missing_cleaned_dict = clean_pdb_batch(
                        missing_cleaned, cleaned_dir, args.relax_python, script_folder, args.max_clean_workers
                    )
                    cleaned_pdb_dict.update(missing_cleaned_dict)
                else:
                    logging.warning(f"⚠ Incomplete cleaned PDBs ({len(cleaned_pdb_dict)}/{len(pdb_files)}), re-cleaning all...")
                    cleaned_pdb_dict = clean_pdb_batch(
                        pdb_files, cleaned_dir, args.relax_python, script_folder, args.max_clean_workers
                    )
            else:
                logging.info("No existing cleaned PDBs found, performing cleaning...")
                cleaned_pdb_dict = clean_pdb_batch(
                    pdb_files, cleaned_dir, args.relax_python, script_folder, args.max_clean_workers
                )
        
        # Step 2: Create batch CSV
        logging.info("--- Step 2: Batch CSV Creation ---")
        batch_csv = os.path.join(work_dir, "batch_protein_ligand.csv")
        batch_df = create_batch_csv(cleaned_pdb_dict, ligand_info, batch_csv)
        
        # Step 3: Batch ESM processing (smart auto-detect with validation)
        logging.info("--- Step 3: ESM Processing ---")
        esm_fasta = os.path.join(work_dir, "batch_esm_input.fasta")
        esm_output_dir = os.path.join(work_dir, "esm2_output")
        
        if args.skip_esm:
            logging.info("Skipping ESM processing (--skip_esm flag)...")
        else:
            # Smart ESM detection with validation - check the actual ESM output directory
            esm_complete = False
            if os.path.exists(esm_output_dir):
                # Check if ESM embeddings match current protein set
                # Note: Each protein may have multiple chains, so we check if we have at least as many .pt files as proteins
                expected_proteins = len(batch_df)
                actual_files = len([f for f in os.listdir(esm_output_dir) if f.endswith('.pt')])
                if actual_files >= expected_proteins:
                    logging.info(f"✓ Auto-detected complete ESM embeddings ({actual_files} files for {expected_proteins} proteins), skipping ESM processing...")
                    esm_complete = True
                else:
                    logging.warning(f"⚠ Incomplete ESM embeddings detected ({actual_files}/{expected_proteins} proteins), will recompute...")
                    logging.warning(f"  ESM directory: {esm_output_dir}")
            
            if not esm_complete:
                # .pt files incomplete, check if FASTA is ready for extraction
                fasta_ready = False
                if os.path.exists(esm_fasta):
                    # Quick validation: check if FASTA has expected number of sequences
                    with open(esm_fasta, 'r') as f:
                        fasta_count = sum(1 for line in f if line.startswith('>'))
                    if fasta_count >= len(batch_df):
                        logging.info(f"✓ Found valid ESM FASTA ({fasta_count} sequences), running extraction to generate .pt files...")
                        fasta_ready = True
                    else:
                        logging.warning(f"⚠ Invalid ESM FASTA ({fasta_count}/{len(batch_df)} sequences), will regenerate...")
                
                if fasta_ready:
                    # FASTA is good, just run extraction to create .pt files
                    run_batch_esm_extraction(esm_fasta, esm_output_dir, args.python, script_folder, args.device)
                else:
                    # Need to create FASTA first, then extract
                    logging.info("Running full ESM processing: embedding (FASTA creation) → extraction (.pt generation)...")
                    run_batch_esm_embedding(batch_csv, esm_fasta, args.python, script_folder)
                    run_batch_esm_extraction(esm_fasta, esm_output_dir, args.python, script_folder, args.device)
        
        if args.only_esm:
            logging.info("ESM processing complete. Exiting as requested.")
            return
        
        # Step 4: Batch screening (with dataset cache detection)
        logging.info("--- Step 4: Batch Screening ---")
        logging.info(f"Processing all {len(batch_df)} proteins in batches of {args.batch_size}")
        results_dir = os.path.join(args.results, args.header)
        
        # DynamicBind will automatically detect and use existing cache via --use_existing_cache flag
        expected_cache_path = f"{args.cache_path}_torsion"  # DynamicBind appends _torsion
        if os.path.exists(expected_cache_path):
            logging.info(f"✓ Cache directory exists: {expected_cache_path} (DynamicBind will auto-detect usable cache)")
        else:
            logging.info(f"Cache directory not found: {expected_cache_path} (DynamicBind will create new cache)")
        
        run_batch_screening(
            batch_csv, esm_output_dir, results_dir, args, 
            args.python, script_folder, model_workdir, ckpt
        )
        
        # Step 5: Batch relaxation
        logging.info("--- Step 5: Batch Relaxation ---")
        run_batch_relaxation(results_dir, args, args.relax_python, script_folder)
        
        # No combined_results since we're not chunking
        combined_results = None
        
        logging.info(f"\n=== Batch Processing Complete ===")
        logging.info(f"Processed {len(batch_df)} protein mutants")
        logging.info(f"Results saved to: {results_dir}")
        if results_dir:
            logging.info(f"Affinity predictions: {results_dir}/affinity_prediction.csv")
        logging.info(f"Working directory: {work_dir}")
        
        # Create final consolidated CSV with requested format
        final_csv_path = os.path.join(args.results, f"{args.header}_final_results.csv")
        
        # Determine which results to pass to the function
        if combined_results is not None:
            current_results_df = combined_results
        elif results_dir and os.path.exists(f"{results_dir}/affinity_prediction.csv"):
            current_results_df = pd.read_csv(f"{results_dir}/affinity_prediction.csv")
        else:
            current_results_df = None
            
        create_final_summary_csv(combined_results, current_results_df, batch_df, final_csv_path)
        
        # Print summary statistics
        if combined_results is not None:
            results_df = combined_results
        elif results_dir and os.path.exists(f"{results_dir}/affinity_prediction.csv"):
            results_df = pd.read_csv(f"{results_dir}/affinity_prediction.csv")
        else:
            results_df = None
            
        if results_df is not None:
            logging.info(f"\n=== Summary Statistics ===")
            logging.info(f"Total predictions: {len(results_df)}")
            
            # Check if affinity column exists before calculating stats
            if 'affinity' in results_df.columns and len(results_df) > 0:
                # Convert affinity column to numeric to handle potential object dtypes
                try:
                    results_df['affinity'] = pd.to_numeric(results_df['affinity'], errors='coerce')
                    
                    # Check if we have any valid (non-NaN) affinity values
                    valid_affinities = results_df['affinity'].dropna()
                    
                    if len(valid_affinities) > 0:
                        logging.info(f"Valid predictions: {len(valid_affinities)}/{len(results_df)}")
                        logging.info(f"Mean affinity: {valid_affinities.mean():.3f}")
                        logging.info(f"Std affinity: {valid_affinities.std():.3f}")
                        logging.info(f"Min affinity: {valid_affinities.min():.3f}")
                        logging.info(f"Max affinity: {valid_affinities.max():.3f}")
                        
                        # Save top 10 results - only if we have valid affinity values
                        try:
                            top_results = results_df.dropna(subset=['affinity']).nlargest(10, 'affinity')
                            if len(top_results) > 0:
                                logging.info(f"\n=== Top {len(top_results)} Results ===")
                                for _, row in top_results.iterrows():
                                    logging.info(f"{row['name']}: {row['affinity']:.3f}")
                            else:
                                logging.warning("No valid results for top ranking")
                        except Exception as e:
                            logging.error(f"Error getting top results: {e}")
                    else:
                        logging.warning("No valid (non-NaN) affinity predictions found")
                        logging.warning("All complexes likely failed during processing")
                        logging.info(f"Failed predictions: {len(results_df)} (all complexes)")
                        
                except Exception as e:
                    logging.error(f"Error converting affinity column to numeric: {e}")
                    logging.warning(f"Affinity column dtype: {results_df['affinity'].dtype}")
                    logging.warning(f"Sample affinity values: {results_df['affinity'].head()}")
                    
            elif 'affinity' not in results_df.columns:
                logging.warning("Warning: 'affinity' column not found in results")
                logging.warning(f"Available columns: {list(results_df.columns)}")
            else:
                logging.warning("Results DataFrame is empty - no predictions generated")
                
        else:
            logging.warning("No results available for summary statistics")
        
        logging.info("Batch processing completed successfully")
        
    except Exception as e:
        logging.error(f"Batch processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 