# Configuration for a sample training run of the M²RA-GNN model.
# This file defines hyperparameters and data paths for a reproducible experiment.

run_name: "preliminary_test_run_35_precursors"

data:
  # Path to the processed RAG data objects
  processed_data_path: "../data/processed/initial_dataset.pt"
  # Splitting ratios for the dataset
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

model:
  # M²RA-GNN architecture parameters
  type: "M2RAGNN"
  hidden_channels: 128
  num_gnn_layers: 4
  num_attention_heads: 4
  use_3d_coords: True

training:
  # Training hyperparameters
  optimizer: "AdamW"
  learning_rate: 0.001
  weight_decay: 0.01
  batch_size: 8
  epochs: 200
  # Loss function weights for multi-task learning
  loss_weights:
    yield: 0.5
    activity_ic50: 0.5

# Active learning loop settings (for later stages)
active_learning:
  enabled: False
  acquisition_function: "EHVI" # Expected Hypervolume Improvement
  num_suggestions_per_loop: 5
