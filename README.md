# Reimplementing ManiDext (Baseline + Stage 1 MDM)
Reimplemented the Baseline MDM and Stage 1 Contact and Correspondence Map MDM from the [ManiDext paper](https://jiajunzhang16.github.io/manidext/) and did a short training. Model checkpoints link available in `models/download.md` file. 

(Please note: just a proof of concept that the training works so the result quality doesn't match that of the official paper due to shorter training).

Thanks to the original authors for the idea and the paper.

### TODO: 
- [ ] Stage 2 MDM Implementation and Training.

## Dataset Download Instructions
### ARCTIC
Please download [ARCTIC Dataset](https://arctic.is.tue.mpg.de/) per the instructions from [here](https://github.com/zc-alexfan/arctic/tree/master/docs/data). You can just use until the "Download "smaller" files (required)" commands, as no images are required.

Please remember the path to the ARCTIC unpacked data folder. (e.g. `arctic/unpack/arctic_data/`)
It should have these following folders: `$ARCTIC_PATH/data/meta`, `$ARCTIC_PATH/data/raw_seqs`, `$ARCTIC_PATH/data/splits_json`.

## Setup
If conda is not installed, please follow official Conda installation process [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Conda `environment.yml` is provided and some additional requirements for `bps_torch` and `chumpy` is provided below.

```bash
# Run environment.yml
conda env create -f environment.yml
conda activate manidext_env

# Install bps_torch from https://github.com/otaheri/bps_torch
pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch

# Install chumpy (required for loading MANO model files)
pip install git+https://github.com/mattloper/chumpy
```

## Code Structure/Explanation:
This is how the code is structured, with explanations for each file:
``` bash
<root>
|------ data
        |------ artic_dataloader.py # Contains the dataloader for training. Also the dataloader has functions to canonicalize the object and hand sequences and compute Ground Truth Contact Correspondence maps on first run, and caches them under this `data` folder.
        |------ canon_seq.py # Contains functions to canonicalize and uncanonicalize the object and hand sequences. Also contains the batched versions of the uncanonicalized functions.
|------ models # please download from instructions from `models/download.md`
        |------ baseline # contains pretrained baseline model for iter 21610.
        |------ cse # contains pretrained hand continuous surface embedding files.
        |------ mano # please put MANO model files here.
        |------ samples # contains some generated some samples from the shorter pre-trained models to visualize.
                |------ baseline
                        |------sample_scissor.pkl
                |------ stage1
                        |------sample_espresso.pkl
        |------ stage1 # contains pretrained stage1 model for iter 46379.
        |------ bps.pt # Cache for the Random BPS sphere needed to encode the dataset.
|------ training
        |------ diffusion
                |------ baseline_mdm.py # Contains the modified MDM implementation, Input and Output Processes and Object Motion and Geometry Embeddings for Baseline Training.
                |------ cfg.py # Template Classifier-Free Guidance implementation from Tim Pierce (https://github.com/TeaPearce/Conditional_Diffusion_MNIST)
                |------ modified_mdm.py # Contains the modified MDM implementation, Input and Output Processes and Object Motion and Geometry Embeddings for Stage 1 MDM training.
        |------ baseline_trainer.py # Python file to train and generate samples from the baseline MDM
        |------ mano_cse.py # File to optimize the hand Continuous Surface Embeddings (CSE)
        |------ stage1_trainer.py # Python file to train and generate samples from the Stage 1 (Contact and Correspondence Map Generation) MDM
|------ utils # contains some utils files for MANO, Viewer, Open3D and Articulated Object Visualization.
|------ visualization
        |------ bps_vis.py # Visualize the motion and the GT contact and embedding maps using the BPS representation of the object mesh. This is also used for sanity checking the canonicalization and uncanonicalization processes. Visualizes it from the Dataloader Cache.
        |------ ce_map_vis.py # Visualize the Ground Truth Contact and Correspondence Embedding Maps (computes at run-time directly from ARCTIC Dataset).
        |------ gt_vis.py # Visualize the Ground Truth Data directly from ARCTIC Dataset.
        |------ mano_cse_vis.py # Visualize the MANO Continuous Surface Embeddings.
        |------ vis_baseline_res.py # Visualize the hand-pose results from generated samples from the pre-trained baseline model.
        |------ vis_stage1_res.py # Visualize the contact map results from generated samples from the pre-trained stage1 model.
```

## Visualize Results
Please ensure ARCTIC dataset is downloaded. `ARCTIC_PATH` is the path to arctic dataset.

### 1. Visualize Ground Truth Data from ARCTIC
```bash
python visualization/gt_vis.py [-h] [--arctic_path ARCTIC_PATH] [--subject SUBJECT] [--object OBJECT] [--action ACTION] [--seq_num SEQ_NUM]
# Arguments:
#   --arctic_path ARCTIC_PATH
#                         Path to unpacked arctic dataset.
#   --subject SUBJECT     Subject id e.g. s01, s02, ... to visualize.
#   --object OBJECT       Object name e.g. box, capsulemachine, ketchup, ... to visualize.
#   --action ACTION       Action type (grab or use) to visualize.
#   --seq_num SEQ_NUM     Sequence number to visualize (e.g. 01, 02, 03, 04).
```

### 2. Visualize MANO Continuous Surface Embeddings
```bash
python visualization/mano_cse_vis.py [-h] [--emb_dim EMB_DIM] [--rhand] [--vis_emb] [--vis_dist] [--vert_idx VERT_IDX]
# Arguments:
#   --emb_dim EMB_DIM    Dimension of embeddings
#   --rhand              Use right hand model else left hand
#   --vis_emb            Visualize embeddings
#   --vis_dist           Visualize distances from a vertex
#   --vert_idx VERT_IDX  Vertex index for distance visualization
```

### 3. Visualize Contact and Correspondence Embedding Maps
```bash
python visualization/ce_map_vis.py [-h] [--arctic_path ARCTIC_PATH] [--subject SUBJECT] [--object OBJECT] [--action ACTION] [--seq_num SEQ_NUM] [--rhand] [--vis_emb]
# Arguments:
#   --arctic_path ARCTIC_PATH
#                         Path to unpacked arctic dataset.
#   --subject SUBJECT     Subject id e.g. s01, s02, ... to visualize.
#   --object OBJECT       Object name e.g. box, capsulemachine, ketchup, ... to visualize.
#   --action ACTION       Action type (grab or use) to visualize.
#   --seq_num SEQ_NUM     Sequence number to visualize (e.g. 01, 02, 03, 04).
#   --rhand               Use right hand model else left hand
#   --vis_emb             Visualize embeddings
```

### 4. Visualize BPS Motion and Check Uncanonicalization from Dataloader
```bash
python visualization/bps_vis.py [-h] [--arctic_path ARCTIC_PATH]
# Arguments:
#   --arctic_path ARCTIC_PATH
#                         Path to unpacked arctic dataset.
```

### 5. Visualize Hand-Pose Generated Samples from Pre-trained Baseline MDM Model
(Note: also shows the Ground Truth next to it.) 
```bash
python visualization/vis_baseline_res.py [-h] [--arctic_path ARCTIC_PATH] [--sample_path SAMPLE_PATH]
# Arguments:
#   --arctic_path ARCTIC_PATH
#                         Path to unpacked arctic dataset.
#   --sample_path SAMPLE_PATH
#                         Path to the generated sample
```

### 6. Visualize Contact Map Generated Samples from Pre-trained Stage 1 MDM Model
(Note: also shows the Ground Truth next to it.) 
```bash
python visualization/vis_stage1_res.py [-h] [--arctic_path ARCTIC_PATH] [--sample_path SAMPLE_PATH]
# Arguments:
#   --arctic_path ARCTIC_PATH
#                         Path to unpacked arctic dataset.
#   --sample_path SAMPLE_PATH
#                         Path to the generated sample
```

## Train or sample from Baseline MDM:
```bash
python training/baseline_trainer.py [-h] [--arctic_path ARCTIC_PATH] [--batch_size BATCH_SIZE] [--seqlen SEQLEN] [--num_iters NUM_ITERS] [--num_timesteps NUM_TIMESTEPS]
                           [--bps_dim BPS_DIM] [--mano_cse_dim MANO_CSE_DIM] [--load_iter LOAD_ITER] [--sample_mode]
# Arguments:
#   --arctic_path ARCTIC_PATH
#                         Path to unpacked arctic dataset.
#   --batch_size BATCH_SIZE
#                         Batch size.
#   --seqlen SEQLEN       Sequence length for clipping data
#   --num_iters NUM_ITERS
#                         Number of iters to train
#   --num_timesteps NUM_TIMESTEPS
#                         Number of timesteps for DDPM
#   --bps_dim BPS_DIM     Basis Point Set dimension
#   --mano_cse_dim MANO_CSE_DIM
#                         MANO Continuous Surface Embedding dimension
#   --load_iter LOAD_ITER
#                         Load model from iteration
#   --sample_mode         Sample from trained model
```

## Train or sample from Stage1 MDM:
```bash
python training/stage1_trainer.py [-h] [--arctic_path ARCTIC_PATH] [--batch_size BATCH_SIZE] [--seqlen SEQLEN] [--num_iters NUM_ITERS] [--num_timesteps NUM_TIMESTEPS]
                         [--bps_dim BPS_DIM] [--mano_cse_dim MANO_CSE_DIM] [--load_iter LOAD_ITER] [--sample_mode]
# Arguments:
#   --arctic_path ARCTIC_PATH
#                         Path to unpacked arctic dataset.
#   --batch_size BATCH_SIZE
#                         Batch size.
#   --seqlen SEQLEN       Sequence length for clipping data
#   --num_iters NUM_ITERS
#                         Number of iters to train
#   --num_timesteps NUM_TIMESTEPS
#                         Number of timesteps for DDPM
#   --bps_dim BPS_DIM     Basis Point Set dimension
#   --mano_cse_dim MANO_CSE_DIM
#                         MANO Continuous Surface Embedding dimension
#   --load_iter LOAD_ITER
#                         Load model from iteration
#   --sample_mode         Sample from trained model
```