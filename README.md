# LRS Spatial Reconstruction

Light Readout System (LRS) spatial reconstruction for DUNE 2x2 detector. Processes HDF5 simulation outputs to reconstruct particle positions from light detector signals.

## Quick Start

```bash
# 1. Activate environment
source ndlar_flow.venv/bin/activate

# 2. Preprocess data (coordinate transform + filtering)
python process_light_outputs.py --n 100 --nint1 --mod123 --dE-weight

# 3. Run reconstruction + analysis
python run_analytic_reco.py \
  processed_outputs_*/truth_reco_processed.csv \
  --save --export-1d-all --plot-2d-heatmaps
```

## What This Does

1. **Preprocessing** (`process_light_outputs.py`):
   - Loads raw event data from HDF5/CSV
   - Extracts TPC geometry bounds
   - Transforms coordinates from mirrored detector frame → unified common frame
   - Filters events (optional: single interaction, specific modules)
   - Outputs: `truth_reco_processed.csv`

2. **Reconstruction** (`run_analytic_reco.py`):
   - Loads processed events + detector geometry
   - Computes predicted positions via PDE-weighted centroid of detector signals
   - Calculates residuals (predicted - true positions)
   - Generates statistical summaries and visualizations

## Key Features

- **Coordinate Unification**: Handles DUNE 2x2's mirrored module geometry
- **PDE Correction**: Accounts for photon detection efficiency variations
- **Statistical Analysis**: 1D residual profiles, 2D heatmaps with quantile overlays
- **Reproducibility**: Timestamped outputs with JSON manifests

## Documentation

- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - Comprehensive guide for AI coding agents (START HERE)
- **[QUICKREF.md](QUICKREF.md)** - Command reference and common issues
- **[REFACTORING_NOTES.md](REFACTORING_NOTES.md)** - Current refactoring effort details
- **[TRANSFORM_BUG_EXPLAINED.md](TRANSFORM_BUG_EXPLAINED.md)** - Visual explanation of coordinate transforms

## Core Architecture

```
HDF5 files → process_light_outputs.py → CSV (transformed coords)
                                           ↓
                                  run_analytic_reco.py
                                           ↓
                      predictions + residuals + plots
```

**Critical Component**: `utils.py` provides canonical coordinate transforms that MUST be used consistently for both event data and detector geometry.

## Current Status (October 2025)

✅ **Fully Operational**:
- `utils.py` provides centralized transform functions
- `process_light_outputs.py` uses utils for coordinate transforms
- `run_analytic_reco.py` correctly transforms detector geometry to common frame
- 3D event displays with accurate detector positions
- Full visualization suite: 1D residuals, 2D heatmaps, 3D event displays
- Organized output directory structure with subdirectories for each plot type

## Requirements

- Python 3.10+
- numpy, pandas, h5py, matplotlib, pyyaml (pre-installed in `ndlar_flow.venv/`)

## Output Structure

```
processed_outputs_n100_nint1_mod123_dEweight_TIMESTAMP/
├── truth_reco_processed.csv    # Transformed event coordinates
├── processing_inputs.json       # Reproducibility metadata
└── geom/
    └── tpc_bounds_mm.npy       # Cached TPC geometry

analytic_reco_processed_outputs_n100_nint1_mod123_dEweight_TIMESTAMP/
└── outputs_pde_TIMESTAMP/
    ├── manifest.json                      # Run metadata
    ├── predictions/
    │   └── predictions.csv.gz             # predx, predy, predz
    ├── residuals_1d/
    │   └── resids_f_*.csv.gz              # 1D residual statistics
    ├── plots_1d/
    │   └── plot_1d_*_{mu,sig}.png         # 1D residual plots
    ├── heatmaps/
    │   ├── heatmaps_f_truex/              # Residuals vs truex
    │   ├── heatmaps_f_truey/              # Residuals vs truey
    │   ├── heatmaps_f_truez/              # Residuals vs truez
    │   ├── heatmaps_f_total_signal/       # Residuals vs total_signal
    │   └── pred_vs_true/                  # Predicted vs true positions
    └── event_displays_3d/
        └── event_display_3d_*.png         # 3D event visualizations
```

## Common Commands

```bash
# Check environment
source ndlar_flow.venv/bin/activate
python -c "from utils import *; print('OK')"

# Full pipeline with all outputs
python process_light_outputs.py --n 100 --nint1 --mod123 --dE-weight
python run_analytic_reco.py processed_outputs_*/truth_reco_processed.csv \
  --save --export-1d-all --plot-2d-heatmaps --plot-1d-all

# Unweighted reconstruction (skip PDE correction)
python run_analytic_reco.py <csv> --uw --save
```

## Troubleshooting

**Predictions way off from true positions?**
→ Verify detector geometry was transformed with `transform_geom()`. See [TRANSFORM_BUG_EXPLAINED.md](TRANSFORM_BUG_EXPLAINED.md)

**3D event displays show wrong detector positions?**
→ Ensure `transform_geom()` uses consistent `out_x/out_y/out_z` parameters matching plotting expectations

**Missing geometry files?**
→ Check sibling repo: `../lrs_sanity_check/geom_files/light_module_desc-4.0.0.{csv,yaml}`

**KeyError for column names?**
→ Run preprocessing first to generate transformed coordinates

See [QUICKREF.md](QUICKREF.md) for more common issues.

## Example Outputs

### Predicted vs True Position
Heatmap showing reconstruction accuracy with y=x reference line (white dashed):

![Predicted vs True Y](examples/heatmap_predy_vs_truey.png)

### 2D Residual Heatmap
Residual distribution vs independent variable with quantile overlays (red lines at 16%, 50%, 84%):

![Residuals vs TrueY](examples/heatmap_dr_vs_truey.png)

### 1D Residual Profile
Standard deviation of residuals with statistical uncertainties as a function of true position:

![1D Residuals Sigma](examples/plot_1d_truez_sig.png)

### 3D Event Display
Event visualization showing detectors (sized by signal), true position (blue star), and predicted position (red star):

![3D Event Display](examples/event_display_3d_dr.png)

## Contributing

When modifying coordinate transforms:
1. **Only edit `utils.py`** - it's the single source of truth
2. Use `AlignParams` to ensure consistency between events and geometry
3. Test with before/after comparison of pred vs true plots
4. Update [.github/copilot-instructions.md](.github/copilot-instructions.md) if adding new patterns

## Contact

See `.github/copilot-instructions.md` for detailed technical documentation.
