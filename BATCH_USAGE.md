# Batch DynamicBind Processing for Multiple Protein Mutants

This guide explains how to use the optimized batch processing script to dock thousands of protein mutants against the same ligand efficiently.

## Key Optimizations

The batch script `run_batch_protein_screening.py` provides several optimizations over the original single-protein script:

1. **Batch ESM Embedding Processing**: All proteins are processed together in a single FASTA file
2. **Batch ESM Extraction**: Language model embeddings are generated for all proteins at once
3. **Parallel PDB Cleaning**: Multiple PDB files are cleaned simultaneously
4. **High-Throughput Screening Mode**: Uses `screening.py` instead of `inference.py` for faster processing
5. **Chunked Processing**: For very large datasets (>1000 proteins), processes in manageable chunks
6. **Reduced Default Parameters**: Optimized for speed with fewer samples and inference steps

## Usage

### Basic Usage
```bash
python run_batch_protein_screening.py /path/to/mutants_folder ligand.csv --header my_batch_run
```

### For Large Datasets (>1000 proteins)
```bash
python run_batch_protein_screening.py /path/to/mutants_folder ligand.csv \
    --header large_batch \
    --use_chunks \
    --chunk_size 500 \
    --batch_size 30 \
    --samples_per_complex 2 \
    --inference_steps 8
```

### With Custom Parameters
```bash
python run_batch_protein_screening.py /path/to/mutants_folder ligand.csv \
    --header custom_run \
    --device 0 \
    --samples_per_complex 5 \
    --inference_steps 15 \
    --batch_size 25 \
    --num_workers 40 \
    --no_relax
```

## Required Input Format

### Mutants Folder
Your mutants folder should contain PDB files:
```
mutants_folder/
├── mutant_001.pdb
├── mutant_002.pdb
├── mutant_003.pdb
└── ...
```

### Ligand CSV
A CSV file with at least a `ligand` column containing SMILES strings:
```csv
ligand,name
COc1ccc(C#N)cc1,compound_1
CCN(CC)CC,compound_2
```

## Processing Workflow

1. **PDB Cleaning** (parallel): Clean all PDB files simultaneously
2. **CSV Creation**: Generate a single CSV with all protein-ligand pairs
3. **ESM Embedding Preparation**: Extract protein sequences to FASTA
4. **ESM Extraction**: Generate language model embeddings for all proteins
5. **Batch Screening**: Run high-throughput docking for all pairs
6. **Results Compilation**: Combine all results into final output files

## Output Files

The script generates several output files:

- `affinity_prediction.csv`: Final affinity predictions for all protein-ligand pairs
- `complete_affinity_prediction.csv`: Detailed results with confidence scores
- `batch_run_TIMESTAMP.log`: Detailed log of the entire process
- Working directory with intermediate files

## Performance Tips

### For 8402 Proteins:
1. Use chunked processing: `--use_chunks --chunk_size 500`
2. Reduce samples for speed: `--samples_per_complex 2`
3. Reduce inference steps: `--inference_steps 8`
4. Increase batch size: `--batch_size 50`
5. Skip relaxation initially: `--no_relax`

### Expected Speedup:
- **ESM Processing**: ~100x faster (batch vs individual)
- **Overall Runtime**: ~10-20x faster than running 8402 individual jobs
- **GPU Memory**: More efficient utilization with batching

## Example Commands

### Quick Test Run (100 proteins)
```bash
python run_batch_protein_screening.py test_mutants/ ligand.csv \
    --header test_run \
    --samples_per_complex 1 \
    --inference_steps 5
```

### Production Run (8402 proteins)
```bash
python run_batch_protein_screening.py S3.0_mutants/ AA_AMP_Amino_T7Q.csv \
    --header S3.0_screening \
    --use_chunks \
    --chunk_size 400 \
    --batch_size 40 \
    --samples_per_complex 3 \
    --inference_steps 10 \
    --device 0 \
    --num_workers 30
```

### Resume from ESM Processing
```bash
python run_batch_protein_screening.py S3.0_mutants/ AA_AMP_Amino_T7Q.csv \
    --header S3.0_resumed \
    --skip_cleaning \
    --skip_esm \
    --use_chunks
```

## Monitoring Progress

The script provides detailed progress information:
- Number of PDB files found and successfully cleaned
- ESM processing progress
- Chunk processing status (if using chunks)
- Final summary statistics including top-performing mutants

## Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce `--batch_size` or `--chunk_size`
2. **ESM Extraction Fails**: Check GPU memory and reduce batch size
3. **PDB Cleaning Fails**: Some PDB files may have formatting issues
4. **No Results**: Check log files for specific error messages

### Performance Tuning:
- Start with small test runs to optimize parameters
- Monitor GPU memory usage during ESM extraction
- Adjust chunk size based on available system memory
- Use multiple workers for CPU-intensive steps 