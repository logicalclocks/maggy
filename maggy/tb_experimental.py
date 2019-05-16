
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

def create_hparams_config(sp):
    hparams = []

    for k, v in sp.names().items():
        if v == 'DOUBLE':
            hparams.append(hp.HParam(k, hp.RealInterval(float(sp.get(k)[0]), float(sp.get(k)[1]))))
        elif v == 'INTEGER':
            hparams.append(hp.HParam(k, hp.IntInterval(sp.get(k)[0], sp.get(k)[1])))
        elif v == 'DISCRETE':
            hparams.append(hp.HParam(k, hp.Discrete(sp.get(k))))
        elif v == 'CATEGORICAL':
            hparams.append(hp.HParam(k, hp.Discrete(sp.get(k))))

    return hparams

def write_hparams_proto(logdir, searchspace):

    HPARAMS = create_hparams_config(searchspace)
    METRICS = [
        hp.Metric(
            "epoch_accuracy",
            group="validation",
            display_name="accuracy (val.)",
        ),
        hp.Metric(
            "epoch_loss",
            group="validation",
            display_name="loss (val.)",
        ),
        hp.Metric(
            "batch_accuracy",
            group="train",
            display_name="accuracy (train)",
        ),
        hp.Metric(
            "batch_loss",
            group="train",
            display_name="loss (train)",
        ),
    ]

    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)
