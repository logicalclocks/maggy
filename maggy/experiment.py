#
#   Copyright 2020 Logical Clocks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

"""
Experiment module used for running asynchronous optimization tasks.

The programming model is that you wrap the code containing the model
training inside a wrapper function.
Inside that wrapper function provide all imports and parts that make up your
experiment, see examples below. Whenever a function to run an experiment is
invoked it is also registered in the Experiments service along with the
provided information.
"""
import os
import atexit
import time

from hops import util as hopsutil
from hops.experiment_impl.util import experiment_utils

from maggy import util, tensorboard
from maggy.core import trialexecutor, experimentdriver

app_id = None
running = False
run_id = 1
experiment_json = None


def lagom(
    train_fn,
    name="no-name",
    experiment_type="optimization",
    searchspace=None,
    optimizer=None,
    direction="max",
    num_trials=1,
    ablation_study=None,
    ablator=None,
    optimization_key="metric",
    hb_interval=1,
    es_policy="median",
    es_interval=1,
    es_min=10,
    description="",
):
    """Launches a maggy experiment, which depending on `experiment_type` can
    either be a hyperparameter optimization or an ablation study experiment.
    Given a search space, objective and a model training procedure `train_fn`
    (black-box function), an experiment is the whole process of finding the
    best hyperparameter combination in the search space, optimizing the
    black-box function. Currently maggy supports random search and a median
    stopping rule.

    **lagom** is a Swedish word meaning "just the right amount".

    :param train_fn: User defined experiment containing the model training.
    :type train_fn: function
    :param name: A user defined experiment identifier.
    :type name: str
    :param experiment_type: Type of Maggy experiment, either 'optimization'
        (default) or 'ablation'.
    :type experiment_type: str
    :param searchspace: A maggy Searchspace object from which samples are
        drawn.
    :type searchspace: Searchspace
    :param optimizer: The optimizer is the part generating new trials.
    :type optimizer: str, AbstractOptimizer
    :param direction: If set to ‘max’ the highest value returned will
        correspond to the best solution, if set to ‘min’ the opposite is true.
    :type direction: str
    :param num_trials: the number of trials to evaluate given the search space,
        each containing a different hyperparameter combination
    :type num_trials: int
    :param ablation_study: Ablation study object. Can be None for optimization
        experiment type.
    :type ablation_study: AblationStudy
    :param ablator: Ablator to use for experiment type 'ablation'.
    :type ablator: str, AbstractAblator
    :param optimization_key: Name of the metric to be optimized
    :type optimization_key: str, optional
    :param hb_interval: The heartbeat interval in seconds from trial executor
        to experiment driver, defaults to 1
    :type hb_interval: int, optional
    :param es_policy: The earlystopping policy, defaults to 'median'
    :type es_policy: str, optional
    :param es_interval: Frequency interval in number of steps to check currently
        running trials for early stopping, defaults to 1.
    :type es_interval: int, optional
    :param es_min: Minimum number of trials finalized before checking for
        early stopping, defaults to 10
    :type es_min: int, optional
    :param description: A longer description of the experiment.
    :type description: str, optional
    :raises RuntimeError: An experiment is currently running.
    :return: A dictionary indicating the best trial and best hyperparameter
        combination with it's performance metric
    :rtype: dict
    """
    global running
    if running:
        raise RuntimeError("An experiment is currently running.")

    job_start = time.time()
    sc = hopsutil._find_spark().sparkContext
    exp_driver = None

    try:
        global app_id
        global experiment_json
        global run_id
        app_id = str(sc.applicationId)

        app_id, run_id = util._validate_ml_id(app_id, run_id)

        # start run
        running = True
        experiment_utils._set_ml_id(app_id, run_id)

        # create experiment dir
        experiment_utils._create_experiment_dir(app_id, run_id)

        tensorboard._register(experiment_utils._get_logdir(app_id, run_id))

        num_executors = util.num_executors(sc)

        # start experiment driver
        if experiment_type == "optimization":

            assert num_trials > 0, "number of trials should be greater " + "than zero"
            tensorboard._write_hparams_config(
                experiment_utils._get_logdir(app_id, run_id), searchspace
            )

            if num_executors > num_trials:
                num_executors = num_trials

            exp_driver = experimentdriver.ExperimentDriver(
                "optimization",
                searchspace=searchspace,
                optimizer=optimizer,
                direction=direction,
                num_trials=num_trials,
                name=name,
                num_executors=num_executors,
                hb_interval=hb_interval,
                es_policy=es_policy,
                es_interval=es_interval,
                es_min=es_min,
                description=description,
                log_dir=experiment_utils._get_logdir(app_id, run_id),
            )

            exp_function = exp_driver.optimizer.name()

        elif experiment_type == "ablation":
            exp_driver = experimentdriver.ExperimentDriver(
                "ablation",
                ablation_study=ablation_study,
                ablator=ablator,
                name=name,
                num_executors=num_executors,
                hb_interval=hb_interval,
                description=description,
                log_dir=experiment_utils._get_logdir(app_id, run_id),
            )
            # using exp_driver.num_executor since
            # it has been set using ablator.get_number_of_trials()
            # in experiment.py
            if num_executors > exp_driver.num_executors:
                num_executors = exp_driver.num_executors

            exp_function = exp_driver.ablator.name()
        else:
            running = False
            raise RuntimeError(
                "Unknown experiment_type:"
                "should be either 'optimization' or 'ablation', "
                "But it is '{0}'".format(str(experiment_type))
            )

        nodeRDD = sc.parallelize(range(num_executors), num_executors)

        # Do provenance after initializing exp_driver, because exp_driver does
        # the type checks for optimizer and searchspace
        sc.setJobGroup(os.environ["ML_ID"], "{0} | {1}".format(name, exp_function))

        experiment_json = experiment_utils._populate_experiment(
            name,
            exp_function,
            "MAGGY",
            exp_driver.searchspace.json(),
            description,
            app_id,
            direction,
            optimization_key,
        )

        exp_ml_id = app_id + "_" + str(run_id)
        experiment_json = experiment_utils._attach_experiment_xattr(exp_ml_id, experiment_json, "INIT")

        util._log(
            "Started Maggy Experiment: {0}, {1}, run {2}".format(name, app_id, run_id)
        )

        exp_driver.init(job_start)

        server_addr = exp_driver.server_addr

        # Force execution on executor, since GPU is located on executor
        nodeRDD.foreachPartition(
            trialexecutor._prepare_func(
                app_id,
                run_id,
                experiment_type,
                train_fn,
                server_addr,
                hb_interval,
                exp_driver._secret,
                optimization_key,
                experiment_utils._get_logdir(app_id, run_id),
            )
        )
        job_end = time.time()

        result = exp_driver.finalize(job_end)
        best_logdir = (
            experiment_utils._get_logdir(app_id, run_id) + "/" + result["best_id"]
        )

        util._finalize_experiment(
            experiment_json,
            float(result["best_val"]),
            app_id,
            run_id,
            "FINISHED",
            exp_driver.duration,
            experiment_utils._get_logdir(app_id, run_id),
            best_logdir,
            optimization_key,
        )

        util._log("Finished Experiment")

        return result

    except:  # noqa: E722
        _exception_handler(
            experiment_utils._seconds_to_milliseconds(time.time() - job_start)
        )
        if exp_driver:
            if exp_driver.exception:
                raise exp_driver.exception
        raise
    finally:
        # grace period to send last logs to sparkmagic
        # sparkmagic hb poll intervall is 5 seconds, therefore wait 6 seconds
        time.sleep(6)
        # cleanup spark jobs
        if running and exp_driver is not None:
            exp_driver.stop()
        run_id += 1
        running = False
        sc.setJobGroup("", "")

    return result


def _exception_handler(duration):
    """
    Handles exceptions during execution of an experiment

    :param duration: duration of the experiment until exception in milliseconds
    :type duration: int
    """
    try:
        global running
        global experiment_json
        if running and experiment_json is not None:
            experiment_json["state"] = "FAILED"
            experiment_json["duration"] = duration
            exp_ml_id = app_id + "_" + str(run_id)
            experiment_utils._attach_experiment_xattr(exp_ml_id, experiment_json, "FULL_UPDATE")
    except Exception as err:
        util._log(err)


def _exit_handler():
    """
    Handles jobs killed by the user.
    """
    try:
        global running
        global experiment_json
        if running and experiment_json is not None:
            experiment_json["status"] = "KILLED"
            exp_ml_id = app_id + "_" + str(run_id)
            experiment_utils._attach_experiment_xattr(exp_ml_id, experiment_json, "FULL_UPDATE")
    except Exception as err:
        util._log(err)


atexit.register(_exit_handler)
