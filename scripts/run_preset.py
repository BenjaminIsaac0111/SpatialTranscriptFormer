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
    'stf_interaction': [
        '--model', 'interaction',
        '--use-nystrom',
        '--num-pathways', '50',
        '--precomputed',
    ],
    'stf_context': [
        '--model', 'interaction',
        '--use-nystrom',
        '--n-neighbors', '6',
        '--precomputed',
    ],
    'stf_high_lr': [
        '--model', 'interaction',
        '--use-nystrom',
        '--num-pathways', '50',
        '--precomputed',
        '--backbone', 'ctranspath',
        '--lr', '5e-4',
        '--weight-decay', '1e-3',
        '--epochs', '1000',
    ],
    'stf_whole_slide_survpath': [
        '--model', 'interaction',
        '--use-nystrom',
        '--num-pathways', '50',
        '--precomputed',
        '--backbone', 'ctranspath',
        '--lr', '5e-4',
        '--weight-decay', '1e-3',
        '--epochs', '1000',
        '--whole-slide',
        '--augment',
        '--use-amp',
        '--batch-size', '8',
        '--grad-accum-steps', '1',
        '--seed', '42',
        '--loss', 'mse',
        '--log-transform',
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
