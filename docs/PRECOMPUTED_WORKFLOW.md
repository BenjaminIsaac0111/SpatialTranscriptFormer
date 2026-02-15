# Pre-Computed Feature Workflow

This workflow enables training the Spatial TranscriptFormer using pre-computed features from a frozen backbone (e.g., ResNet50, CTransPath). This approach aligns with the methodology of Jaume et al. (CVPR 2024) and significantly accelerates training (approx. 25x faster).

## 1. Extract Features

Run the extraction script to process H&E patches and save feature tensors to `he_features/`.

```powershell
python src/spatial_transcript_former/data/extract_features.py --data-dir A:\hest_data --backbone resnet50 --batch-size 32
```

**Arguments:**

- `--data-dir`: Root directory containing `patches/` and `st/`.
- `--output-dir`: Output directory (default: `he_features` inside inputs).
- `--backbone`: Model backbone (default: `resnet50`).
- `--limit`: (Optional) Limit number of slides to process (for testing).

## 2. Train with Pre-Computed Features

Train the model using the `--precomputed` flag. The script will automatically filter for samples that have existing feature files.

```powershell
python src/spatial_transcript_former/train.py --data-dir A:\hest_data --model interaction --precomputed --epochs 50 --batch-size 32 --n-neighbors 6
```

**Key Arguments:**

- `--precomputed`: Enables the pre-computed feature loader.
- `--backbone`: **Must match** the backbone used for extraction (to determine input logic, though weights won't be loaded if not needed).
- `--n-neighbors`: Number of neighbors to retrieve from the pre-computed coordinate tree.

## 3. Enable Long-Range Interactions (Global Context)

To model long-range interactions (similar to Jaume et al.), use the `--use-global-context` flag. This will mix randomly sampled patches from the entire slide into the context window for each training sample.

```powershell
python src/spatial_transcript_former/train.py --data-dir A:\hest_data --model interaction --precomputed --use-global-context --global-context-size 256
```

- `--use-global-context`: Enables mixing of global patches.
- `--global-context-size`: Number of random global patches to include (default: 128).

## 4. Pure Global Mode (No Neighbors)

To use **only global context** (and the center patch), simply set neighbors to 0:

```powershell
python src/spatial_transcript_former/train.py --data-dir A:\hest_data --model interaction --precomputed --use-global-context --global-context-size 256 --n-neighbors 0
```

This will construct a sequence of `[Center Patch] + [256 Random Global Patches]`.

## Performance Note

- Training speed is approximately **60 iterations/second** (vs ~2 iter/s with live backbone).
- Memory usage is lower as images are not loaded.
