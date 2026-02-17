import subprocess
import argparse
import sys
import os

PRESETS = {
    'he2rna_baseline': [
        '--model', 'he2rna',
        '--backbone', 'resnet50',
        '--batch-size', '64',
    ],
    'vit_baseline': [
        '--model', 'vit_st',
        '--backbone', 'vit_b_16',
        '--batch-size', '32',
    ],
    'attention_mil': [
        '--model', 'attention_mil',
        '--whole-slide',
        '--precomputed',
        '--batch-size', '1',
    ],
    'transmil': [
        '--model', 'transmil',
        '--whole-slide',
        '--precomputed',
        '--batch-size', '1',
    ],
    'stf_pathway_nystrom': [
        '--model', 'interaction',
        '--backbone', 'ctranspath',
        '--precomputed',
        '--whole-slide',
        '--use-nystrom',
        '--pathway-init',
        '--sparsity-lambda', '0.05',
        '--lr', '1e-4',
        '--batch-size', '8',
        '--epochs', '2000',
        '--log-transform',
        '--use-amp',
        '--plot-pathways',
        '--loss', 'mse_pcc',
        '--resume',
    ],
    'stf_pathway': [
        '--model', 'interaction',
        '--backbone', 'ctranspath',
        '--precomputed',
        '--whole-slide',
        '--pathway-init',
        '--sparsity-lambda', '0.05',
        '--lr', '1e-4',
        '--batch-size', '8',
        '--epochs', '2000',
        '--log-transform',
        '--use-amp',
        '--plot-pathways',
        '--loss', 'mse_pcc',
        '--resume',
    ]
}

def main():
    parser = argparse.ArgumentParser(description="Run Spatial TranscriptFormer training presets")
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()), required=True, help='Preset configuration')
    parser.add_argument('--data-dir', type=str, default='A:\\hest_data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit samples')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output dir')
    
    # Custom args allow appending more arguments to train.py
    args, unknown = parser.parse_known_args()
    
    cmd = [
        sys.executable, '-m', 'spatial_transcript_former.train',
        '--data-dir', args.data_dir,
        '--epochs', str(args.epochs),
    ]
    
    if args.max_samples:
        cmd += ['--max-samples', str(args.max_samples)]
        
    if args.output_dir:
        cmd += ['--output-dir', args.output_dir]
    else:
        # Default output dir based on preset
        cmd += ['--output-dir', f'./runs/{args.preset}']

    # Add preset arguments
    cmd += PRESETS[args.preset]
    
    # Add any unknown arguments passed to this script
    cmd += unknown
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
