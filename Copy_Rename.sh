#!/bin/bash

# Check if destination directory was provided as argument, otherwise use current directory
DEST_DIR="${1:-.}"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Define source directories
DIR_CON="/Users/cruz/RESEARCH/COSMO/CMBFSCNN/CMBFSCNN_1/DATA_results_con/output_PS"
DIR_S5D10="/Users/cruz/RESEARCH/COSMO/CMBFSCNN/CMBFSCNN_1/DATA_results_s5_d10/output_PS"
DIR_SIN_CON="/Users/cruz/RESEARCH/COSMO/CMBFSCNN/CMBFSCNN_1/DATA_results_sin_con/output_PS"
DIR_SIN="/Users/cruz/RESEARCH/COSMO/CMBFSCNN/CMBFSCNN_1/DATA_results_LiteBIRD_sin/output_PS"

# Copy and rename files for standard case (s1d1)
cp "$DIR_CON/output_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s1d1_recovered_auto.npy"
cp "$DIR_CON/target_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s1d1_target_auto.npy"
cp "$DIR_CON/output_cmb_EB_cross_PS_nlb_5.npy" "$DEST_DIR/s1d1_recovered_cross.npy"
cp "$DIR_CON/true_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s1d1_target_cross.npy"

# Copy and rename files for s5d10 case
cp "$DIR_S5D10/output_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s5d10_recovered_auto.npy"
cp "$DIR_S5D10/target_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s5d10_target_auto.npy"
cp "$DIR_S5D10/output_cmb_EB_cross_PS_nlb_5.npy" "$DEST_DIR/s5d10_recovered_cross.npy"
cp "$DIR_S5D10/true_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s5d10_target_cross.npy"

# Copy and rename files for fixed foregrounds in training only
cp "$DIR_SIN_CON/output_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s1d1_fixed_train_recovered_auto.npy"
cp "$DIR_SIN_CON/target_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s1d1_fixed_train_target_auto.npy"
cp "$DIR_SIN_CON/output_cmb_EB_cross_PS_nlb_5.npy" "$DEST_DIR/s1d1_fixed_train_recovered_cross.npy"
cp "$DIR_SIN_CON/true_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s1d1_fixed_train_target_cross.npy"

# Copy and rename files for fixed foregrounds in both training and testing
cp "$DIR_SIN/output_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s1d1_fixed_all_recovered_auto.npy"
cp "$DIR_SIN/target_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s1d1_fixed_all_target_auto.npy"
cp "$DIR_SIN/output_cmb_EB_cross_PS_nlb_5.npy" "$DEST_DIR/s1d1_fixed_all_recovered_cross.npy"
cp "$DIR_SIN/true_cmb_EB_PS_nlb_5.npy" "$DEST_DIR/s1d1_fixed_all_target_cross.npy"
