
from maggy.core import config
from maggy.tb_experimental import write_hparams_config

tensorboard_dir = None

def _register(trial_dir):
    global tensorboard_dir

    tensorboard_dir = trial_dir

def logdir():
    global tensorboard_dir
    return tensorboard_dir
