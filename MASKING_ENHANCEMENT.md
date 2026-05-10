# Style Encoder Enhancement with Masking Heads

## Overview

This branch implements masking-based style enhancement for the HiGAN+ style encoder, inspired by the DiffBrush paper (https://arxiv.org/pdf/2508.03256). The enhancement adds two specialized masking heads to improve style learning in both vertical and horizontal directions.

## Key Features

### 1. Vertical Masking Head (Column-wise Masking)
- **Purpose**: Enhances style learning for vertical alignment patterns
- **Method**: Applies column-wise masking to feature maps
- **Benefits**: Better captures vertical writing characteristics like character alignment and stroke consistency

### 2. Horizontal Masking Head (Row-wise Masking)
- **Purpose**: Enhances style learning for horizontal patterns
- **Method**: Applies row-wise masking to feature maps
- **Benefits**: Better captures word spacing, character spacing, and horizontal flow

### 3. Style Fusion
- Combines base style encoding with masked style representations
- Uses a fusion network to integrate complementary style information
- Maintains compatibility with existing VAE mode

## Architecture Changes

### Modified Files

1. **networks/module.py**
   - Added `VerticalMaskingHead` class
   - Added `HorizontalMaskingHead` class
   - Enhanced `StyleEncoder` with masking capabilities
   - Added `use_masking` parameter (default: True)
   - Added `mask_ratio` parameter (default: 0.3)

2. **networks/loss.py**
   - Added `masked_style_consistency_loss` function
   - Ensures consistency between base and masked styles
   - Encourages diversity between vertical and horizontal representations

3. **networks/model.py**
   - Integrated masking loss into training loop
   - Added `mask_consistency_loss` to loss metrics
   - Updated logging to track masking loss

## Usage

### Training with Masking Enhancement

The masking enhancement is enabled by default. To train:

```bash
python train.py --config your_config.yaml
```

### Disabling Masking (for comparison)

To disable masking and use the original StyleEncoder:

```python
# In your config or model initialization
style_encoder = StyleEncoder(
    style_dim=32,
    in_dim=256,
    use_masking=False  # Disable masking
)
```

### Adjusting Mask Ratio

The mask ratio controls how much of the feature map is masked:

```python
style_encoder = StyleEncoder(
    style_dim=32,
    in_dim=256,
    use_masking=True,
    mask_ratio=0.3  # 30% of features are masked (default)
)
```

## Technical Details

### Vertical Masking Head

```python
class VerticalMaskingHead(nn.Module):
    """
    Applies column-wise masking to enhance vertical style patterns.
    - Randomly masks columns in the feature map
    - Aggregates masked features
    - Encodes to style vector
    """
```

### Horizontal Masking Head

```python
class HorizontalMaskingHead(nn.Module):
    """
    Applies row-wise masking to enhance horizontal style patterns.
    - Randomly masks rows (channels) in the feature map
    - Aggregates masked features
    - Encodes to style vector
    """
```

### Loss Function

The consistency loss ensures that masked styles remain consistent with the base style while encouraging diversity:

```python
consistency_loss = (vertical_loss + horizontal_loss) * 0.5 + diversity_loss * 0.1
```

Where:
- `vertical_loss`: L1 distance between base and vertical masked style
- `horizontal_loss`: L1 distance between base and horizontal masked style
- `diversity_loss`: Negative L1 distance between vertical and horizontal styles (encourages diversity)

## Expected Benefits

1. **Improved Style Consistency**: Better capture of writing style characteristics
2. **Enhanced Vertical Alignment**: More accurate vertical stroke patterns
3. **Better Spacing**: Improved character and word spacing
4. **Robust Style Learning**: More robust to variations in input images

## Training Metrics

The training loop now tracks an additional metric:
- `mask_consistency_loss`: Measures the consistency between base and masked styles

Monitor this metric to ensure the masking heads are learning effectively.

## Comparison with Baseline

To compare with the baseline (without masking):
1. Train a model with `use_masking=True` (this branch)
2. Train a model with `use_masking=False` (baseline)
3. Compare FID, KID, and visual quality metrics

## References

- DiffBrush Paper: https://arxiv.org/pdf/2508.03256
- DiffBrush Code: https://github.com/dailenson/DiffBrush
- HiGAN+ Original: https://github.com/ganji15/HiGANplus

## Notes

- Masking is only applied during training (`self.training` mode)
- During inference, the full style encoder is used without masking
- The masking ratio can be tuned based on dataset characteristics
- The consistency loss weight (0.1) can be adjusted in model.py line 976
