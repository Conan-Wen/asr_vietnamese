import os

from data.data_params import batch_size


default_epochs = 100
batch_size
validation_split_rate = 0.1

# Current Dirctory
cur_dir = os.path.dirname(os.path.abspath(__file__))

checkpoint_filepath = os.path.join(cur_dir, "ckpt", "checkpoint_epoch-{epoch:02d}.hdf5")
