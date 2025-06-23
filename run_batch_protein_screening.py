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

def do(cmd, get=False, show=True):
    """Execute shell command"""
    if get:
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0].decode()
        if show:
            print(out, end="")
        return out
    else:
        return subprocess.Popen(cmd, shell=True).wait()

def clean_pdb_batch(pdb_files, cleaned_dir, relax_python, script_folder, max_workers=10):
    """Clean multiple PDB files in parallel"""
    print(f"Cleaning {len(pdb_files)} PDB files...")
    os.makedirs(cleaned_dir, exist_ok=True)
    
    def clean_single_pdb(pdb_file):
        cleaned_file = os.path.join(cleaned_dir, f"cleaned_{os.path.basename(pdb_file)}")
        cmd = f"{relax_python} {script_folder}/clean_pdb.py {pdb_file} {cleaned_file}"
        result = do(cmd, show=False)
        if result == 0:
            return pdb_file, cleaned_file
        else:
            print(f"Warning: Failed to clean {pdb_file}")
            return pdb_file, None
    
    with Pool(max_workers) as pool:
        results = pool.map(clean_single_pdb, pdb_files)
    
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
        data.append({
            'protein_path': cleaned_pdb,
            'ligand': ligand_info,
            'name': protein_name,
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
    cmd = f"CUDA_VISIBLE_DEVICES={device} {python} {script_folder}/esm/scripts/extract.py esm2_t33_650M_UR50D {input_fasta} {output_dir} --repr_layers 33 --include per_tok --truncation_seq_length 10000 --model_dir {script_folder}/esm_models"
    result = do(cmd)
    if result != 0:
        raise RuntimeError("ESM extraction failed")
    print(f"ESM extraction complete: {output_dir}")

def run_batch_screening(batch_csv, esm_dir, output_dir, args, python, script_folder, model_workdir, ckpt):
    """Run high-throughput screening for all protein-ligand pairs"""
    print(f"Running batch screening for all {len(pd.read_csv(batch_csv))} protein-ligand pairs...")
    
    protein_dynamic = "--protein_dynamic" if not args.rigid_protein else ""
    
    cmd = f"CUDA_VISIBLE_DEVICES={args.device} {python} {script_folder}/screening.py --seed {args.seed} --ckpt {ckpt} {protein_dynamic}"
    cmd += f" --save_visualisation --model_dir {model_workdir} --protein_ligand_csv {batch_csv}"
    cmd += f" --esm_embeddings_path {esm_dir} --out_dir {output_dir}"
    cmd += f" --inference_steps {args.inference_steps} --samples_per_complex {args.samples_per_complex}"
    cmd += f" --savings_per_complex {args.savings_per_complex} --batch_size {args.batch_size}"
    cmd += f" --actual_steps {args.inference_steps} --no_final_step_noise"
    
    result = do(cmd)
    if result != 0:
        raise RuntimeError("Batch screening failed")
        
    print("Batch screening complete")

def run_batch_relaxation(output_dir, args, relax_python, script_folder):
    """Run relaxation for all results"""
    if not args.no_relax:
        print("Running batch relaxation...")
        cmd = f"CUDA_VISIBLE_DEVICES={args.device} {relax_python} {script_folder}/relax_final.py --results_path {output_dir} --samples_per_complex {args.samples_per_complex} --num_workers {args.num_workers}"
        result = do(cmd)
        if result != 0:
            print("Warning: Batch relaxation failed")
        else:
            print("Batch relaxation complete")

def process_chunks(batch_df, chunk_size, esm_output_dir, args, python, script_folder, model_workdir, ckpt, work_dir):
    """Process the batch in chunks to manage memory"""
    chunks = [batch_df[i:i+chunk_size] for i in range(0, len(batch_df), chunk_size)]
    print(f"Processing {len(chunks)} chunks of size {chunk_size}")
    
    all_results = []
    
    for i, chunk_df in enumerate(chunks):
        print(f"\n=== Processing chunk {i+1}/{len(chunks)} ({len(chunk_df)} proteins) ===")
        
        # Create chunk CSV
        chunk_csv = os.path.join(work_dir, f"chunk_{i}.csv")
        chunk_df.to_csv(chunk_csv, index=False)
        
        # Create chunk output directory
        chunk_output_dir = os.path.join(args.results, f"{args.header}_chunk_{i}")
        os.makedirs(chunk_output_dir, exist_ok=True)  # Ensure directory exists
        
        try:
            # Run screening for this chunk
            run_batch_screening(
                chunk_csv, esm_output_dir, chunk_output_dir, args,
                python, script_folder, model_workdir, ckpt
            )
            
            # Collect results
            results_file = f"{chunk_output_dir}/affinity_prediction.csv"
            if os.path.exists(results_file):
                try:
                    chunk_results = pd.read_csv(results_file)
                    all_results.append(chunk_results)
                    print(f"Chunk {i+1} completed: {len(chunk_results)} predictions")
                except Exception as read_error:
                    print(f"Warning: Failed to read results file for chunk {i+1}: {read_error}")
            else:
                print(f"Warning: No results found for chunk {i+1} at {results_file}")
                
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        final_results_dir = os.path.join(args.results, f"{args.header}_combined")
        os.makedirs(final_results_dir, exist_ok=True)
        combined_results.to_csv(f"{final_results_dir}/affinity_prediction.csv", index=False)
        print(f"Combined results saved: {final_results_dir}/affinity_prediction.csv")
        return final_results_dir, combined_results
    else:
        return None, None

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
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for screening')
    parser.add_argument('--chunk_size', type=int, default=500, help='Process in chunks of this size to manage memory')
    parser.add_argument('--num_workers', type=int, default=20, help='Workers for relaxation')
    parser.add_argument('--max_clean_workers', type=int, default=10, help='Workers for parallel PDB cleaning')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model', type=int, default=1, help='Model version')
    parser.add_argument('--no_relax', action='store_true', help='Skip relaxation step')
    parser.add_argument('--rigid_protein', action='store_true', help='Use rigid protein')
    parser.add_argument('--ligand_column', type=str, default='ligand', help='Column name for ligand in CSV')
    
    # Python environments
    parser.add_argument('--python', type=str, default='/nfs/hpc/share/nguyhun2/.conda/envs/dynamicbind/bin/python', help='DynamicBind python')
    parser.add_argument('--relax_python', type=str, default='/nfs/hpc/share/nguyhun2/.conda/envs/relax/bin/python', help='Relax python')
    
    # Processing options
    parser.add_argument('--skip_cleaning', action='store_true', help='Skip PDB cleaning (if already cleaned)')
    parser.add_argument('--skip_esm', action='store_true', help='Skip ESM steps (if already done)')
    parser.add_argument('--only_esm', action='store_true', help='Only run ESM steps')
    parser.add_argument('--use_chunks', action='store_true', help='Process in chunks (recommended for >1000 proteins)')
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler(f'batch_run_{timestamp}.log')
    logger = logging.getLogger("")
    logger.addHandler(handler)
    
    logging.info(f'Batch DynamicBind run started: {timestamp}')
    logging.info(f'Command: {" ".join(sys.argv)}')
    
    # Setup paths
    script_folder = os.path.dirname(os.path.realpath(__file__))
    work_dir = f"batch_work_{args.header}_{timestamp}"
    os.makedirs(work_dir, exist_ok=True)
    
    # Find all PDB files in mutants folder
    pdb_pattern = os.path.join(args.mutants_folder, "*.pdb")
    pdb_files = glob.glob(pdb_pattern)
    
    if not pdb_files:
        raise ValueError(f"No PDB files found in {args.mutants_folder}")
    
    print(f"Found {len(pdb_files)} PDB files to process")
    
    # Read ligand information
    ligand_df = pd.read_csv(args.ligand_csv)
    if args.ligand_column not in ligand_df.columns:
        raise ValueError(f"Column '{args.ligand_column}' not found in {args.ligand_csv}")
    
    # For now, use the first ligand (assuming same ligand for all)
    ligand_info = ligand_df[args.ligand_column].iloc[0]
    print(f"Using ligand: {ligand_info}")
    
    # Setup model paths
    if args.model == 1:
        model_workdir = f"{script_folder}/workdir/big_score_model_sanyueqi_with_time"
        ckpt = "pro_ema_inference_epoch138_model.pt"
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    try:
        # Step 1: Clean PDB files in parallel (if not skipped)
        if args.skip_cleaning:
            print("Skipping PDB cleaning...")
            cleaned_pdb_dict = {pdb: pdb for pdb in pdb_files}
        else:
            cleaned_dir = os.path.join(work_dir, "cleaned_pdbs")
            cleaned_pdb_dict = clean_pdb_batch(
                pdb_files, cleaned_dir, args.relax_python, script_folder, args.max_clean_workers
            )
        
        # Step 2: Create batch CSV
        batch_csv = os.path.join(work_dir, f"batch_protein_ligand_{timestamp}.csv")
        batch_df = create_batch_csv(cleaned_pdb_dict, ligand_info, batch_csv)
        
        # Step 3: Batch ESM processing (if not skipped)
        esm_fasta = os.path.join(work_dir, f"batch_esm_input_{timestamp}.fasta")
        esm_output_dir = os.path.join(work_dir, "esm2_output")
        
        if not args.skip_esm:
            run_batch_esm_embedding(batch_csv, esm_fasta, args.python, script_folder)
            run_batch_esm_extraction(esm_fasta, esm_output_dir, args.python, script_folder, args.device)
        
        if args.only_esm:
            print("ESM processing complete. Exiting as requested.")
            return
        
        # Step 4: Batch screening (with chunking if requested)
        if args.use_chunks or len(batch_df) > 1000:
            print(f"Using chunked processing for {len(batch_df)} proteins")
            results_dir, combined_results = process_chunks(
                batch_df, args.chunk_size, esm_output_dir, args, 
                args.python, script_folder, model_workdir, ckpt, work_dir
            )
        else:
            results_dir = os.path.join(args.results, f"{args.header}_{timestamp}")
            run_batch_screening(
                batch_csv, esm_output_dir, results_dir, args, 
                args.python, script_folder, model_workdir, ckpt
            )
            combined_results = None
        
        # Step 5: Batch relaxation (only for non-chunked processing)
        if not args.use_chunks and len(batch_df) <= 1000:
            run_batch_relaxation(results_dir, args, args.relax_python, script_folder)
        
        print(f"\n=== Batch Processing Complete ===")
        print(f"Processed {len(batch_df)} protein mutants")
        print(f"Results saved to: {results_dir}")
        if results_dir:
            print(f"Affinity predictions: {results_dir}/affinity_prediction.csv")
        print(f"Working directory: {work_dir}")
        
        # Print summary statistics
        if combined_results is not None:
            results_df = combined_results
        elif results_dir and os.path.exists(f"{results_dir}/affinity_prediction.csv"):
            results_df = pd.read_csv(f"{results_dir}/affinity_prediction.csv")
        else:
            results_df = None
            
        if results_df is not None:
            print(f"\n=== Summary Statistics ===")
            print(f"Total predictions: {len(results_df)}")
            
            # Check if affinity column exists before calculating stats
            if 'affinity' in results_df.columns:
                print(f"Mean affinity: {results_df['affinity'].mean():.3f}")
                print(f"Std affinity: {results_df['affinity'].std():.3f}")
                print(f"Min affinity: {results_df['affinity'].min():.3f}")
                print(f"Max affinity: {results_df['affinity'].max():.3f}")
                
                # Save top 10 results
                top_results = results_df.nlargest(10, 'affinity')
                print(f"\n=== Top 10 Results ===")
                for _, row in top_results.iterrows():
                    print(f"{row['name']}: {row['affinity']:.3f}")
            else:
                print("Warning: 'affinity' column not found in results")
                print(f"Available columns: {list(results_df.columns)}")
        
        logging.info("Batch processing completed successfully")
        
    except Exception as e:
        logging.error(f"Batch processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 