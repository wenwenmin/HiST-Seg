HiST-Seg baseline code: path configuration guide
================================================

This directory contains the eight baseline implementations and the shared
spot-to-pixel evaluation scripts used by the HiST-Seg experiments. The code
has been prepared for public release: private absolute paths were removed,
Chinese source comments were removed or replaced with English descriptions,
and experiment entry points contain a short path-configuration note.

1. Path conventions
-------------------

The public copy uses the following repository-relative placeholders:

  data/         Native baseline inputs and aligned evaluation data.
  outputs/      Predictions, checkpoints, metrics, and converted pixel masks.
  workspaces/   Temporary work directories used by iStar and TESLA.
  weights/      Pretrained weights required by an individual baseline.

These directories are examples only; the data and pretrained weights are not
included. Replace the defaults in each experiment script or configuration file
with paths on your system. When a script provides command-line path arguments,
prefer those arguments over editing the source file.

Any remaining value named REPLACE_WITH_YOUR_PATH or REPLACE_WITH_YOUR_HOME is
an explicit placeholder and must be replaced before execution. A convenient
search command is:

  rg -n "REPLACE_WITH_YOUR_PATH|REPLACE_WITH_YOUR_HOME" .

Some upstream baseline scripts use relative paths such as ../../data,
../../dataset, or ../../predict_files. These are preserved because they belong
to the original baseline layout. Either reproduce that layout or replace the
corresponding variables near the beginning or main section of the script.

2. Two-stage protocol for spot-level baselines
-----------------------------------------------

BLEEP, MCGAE, MEATRD, and SpaCRD are spot-level methods. Their experiments use
two distinct coordinate spaces and must be run in the following order.

Stage A - native-data spot prediction:

  Train and infer each method using its method-native, original-scale H&E and
  ST inputs, or NPZ/BIN features prepared from those native inputs. Do not use
  resized_data or cropped_data as model inputs. The output must contain one
  prediction value per original ST spot, in the same observation order as the
  source AnnData object.

  Typical native inputs used by the supplied scripts include:

    data/HB/her2st/<section>/he-raw.jpg
    data/HB/her2st/<section>/<section>.h5ad
    data/HB/Visium_HBC/filtered_feature_bc_matrix.h5
    data/HB/Ductal_CIS/filtered_feature_bc_matrix.h5
    data/HB/Xenium_HBC/spot level/transfered.h5ad
    data/CRC_new/<section>/tissue_hires_image.png
    data/CRC_new/<section>/transformed.h5ad

  BLEEP reads precomputed NPZ features derived from the native data. MEATRD
  and SpaCRD use prepare_*.py to build their BIN/NPZ inputs. MCGAE reads the
  native H&E and ST files directly. Replace these example paths with the
  corresponding native-data paths on your system.

Stage B - pixel-level evaluation in the common aligned space:

  The shared evaluators load the aligned evaluation package, read the aligned
  spot coordinates, rasterize the native-data spot predictions with a
  tissue-clipped nearest-spot Voronoi rule, and then compute pixel metrics.
  The aligned files are evaluation geometry only; they are not spot-model
  training or inference inputs.

  STHBC evaluation:

    data/STHBC/resized_data/<section>/he_resized.png
    data/STHBC/resized_data/<section>/label_resized.png
    data/STHBC/resized_data/<section>/mask_resized.png
    data/STHBC/resized_data/<section>/adata_resized.h5ad
    coordinate key: adata.obsm['spatial_resized']

  CRC evaluation:

    data/CRC/cropped_data/he_cropped/<section>.png
    data/CRC/cropped_data/label_cropped/<section>.png
    data/CRC/cropped_data/tissue_mask_cropped/<section>.png
    data/CRC/cropped_data/h5ad_cropped/<section>_cropped.h5ad
    coordinate key: adata.obsm['spatial_cropped']

  The released CRC crops did not require the additional longest-side resize,
  so cropped_data is the common aligned evaluation space for CRC. If a future
  CRC image is resized, its evaluator and coordinate key must be changed to
  the corresponding resized files and spatial_resized coordinates.

  Cross-dataset evaluation:

    data/ViHBC/resized_data/{he,label,mask,adata}_resized.*
    data/DuCIS/resized_data/{he,label,mask,adata}_resized.*
    data/XeHBC/resized_data/{he,label,mask,adata}_resized.*
    coordinate key: adata.obsm['spatial_resized']

The native and aligned AnnData files must contain the same spots in the same
order. In particular, the alignment/cropping/resizing step must transform spot
coordinates without filtering, sorting, or reindexing observations. The
evaluation scripts verify the prediction count, but plain-text predictions do
not contain barcodes and therefore cannot recover from a changed spot order.

3. Model-specific path settings
-------------------------------

AHF-U-Net:
  Update patch, H5AD, checkpoint, prediction, and whole-slide data roots in
  run_*.py, evaluate_*.py, and image_baseline_profiler_core.py.

BLEEP, MCGAE, MEATRD, and SpaCRD:
  Update the native dataset, prepared-feature, label, and prediction roots in
  the run_* and prepare_* scripts. These scripts must point to original-scale
  inputs, not to resized_data or cropped_data. Several upstream relative paths
  are retained and should be adapted to your local layout.

CGS-Net:
  Edit the local YAML file in sthbc_loso/, crc_loso/, or cross_dataset/.
  Set project_root, data_root, weights_dir, output_root, and any external
  slide_sources before running the corresponding run_*.py script.

iStar:
  Update base_data_dir and workspace_dir in the run_* scripts. Point the
  checkpoint variables or download scripts to the required HIPT checkpoints.

TESLA:
  Update the data roots and tesla_workspace_dir/TESLA_WORKSPACE variables in
  run_*.py, evaluate_*.py, and profile_vihbc.py.

4. Shared spot-to-pixel evaluation
----------------------------------

The following scripts convert spot-level predictions into tissue-clipped
pixel-level Voronoi maps and compute the reported segmentation metrics:

  evaluate_spot_to_pixel_sthbc.py
  evaluate_spot_to_pixel_crc.py
  evaluate_spot_to_pixel_cross_dataset.py

Run these scripts only after native-data spot inference has finished. Set the
prediction root, aligned evaluation-data root, and output root through their
command-line arguments or the documented variables in the main block.
Prediction filenames and spot ordering must match the output produced by the
relevant spot-level baseline script.

5. Execution notes
------------------

Run an experiment from its baseline directory unless the script documents a
different working directory. Install each upstream method's dependencies and
obtain all required pretrained weights separately. The cleanup performed for
this public copy changes documentation, comments, and path defaults only; it
does not change the model architecture, optimization logic, random seeds, or
metrics.
