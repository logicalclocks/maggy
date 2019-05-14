
from maggy.core import config

if config.tf_version >= 2:
    from maggy.tb_experimental import write_hparams_proto

tensorboard_dir = None

def _register(trial_dir):
    global tensorboard_dir

    tensorboard_dir = trial_dir

def logdir():
    global tensorboard_dir
    return tensorboard_dir
