## Data
test_split = 0.15
random_seed = 2220

## Training
batch_size = 512
learning_rate = 9e-6
weight_decay = 1e-6
epochs = 100
num_stages = 2
num_layers = 9
num_f_maps = 256
dim = 1024
wav2vec_dim = 149
compute_dim = 256
num_classes = 9
causal_conv = False
num_resblocks = 2
patience_lim = 8    # early stopper callback patience
escb_beta = 0.25    # sensitivity margin