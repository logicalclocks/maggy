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
import uuid
import time
import traceback
from datetime import datetime
from copy import deepcopy

from hops import hdfs
from maggy.optimizer.abstractoptimizer import AbstractOptimizer
from maggy.pruner import Hyperband
from maggy.searchspace import Searchspace
from maggy.trial import Trial


class RandomSearch(AbstractOptimizer):
    def __init__(self, pruner=None, pruner_kwargs=None):
        super().__init__()
        self.config_buffer = []

        # configure pruner
        if pruner:
            self.init_pruner(pruner, pruner_kwargs)

        # configure logger
        self.log_file = "hdfs:///Projects/{}/Experiments_Data/optimizer_{}_{}_{}.log".format(
            hdfs.project_name(),
            self.name(),
            self.pruner.name() if self.pruner else "",
            str(uuid.uuid4()),
        )
        if not hdfs.exists(self.log_file):
            hdfs.dump("", self.log_file)
        self.fd = hdfs.open_file(self.log_file, flags="w")
        self._log("Initialized Logger")
        self._log("Initilized Optimizer {}: \n {}".format(self.name(), self.__dict__))

        # helper variable to calculate time needed for calculating next suggestion
        self.sampling_time_start = 0.0

    def initialize(self):

        if (
            Searchspace.DOUBLE not in self.searchspace.names().values()
            and Searchspace.INTEGER not in self.searchspace.names().values()
        ):
            raise NotImplementedError(
                "Searchspace needs at least one continuous parameter for random search."
            )

        self.config_buffer = self.searchspace.get_random_parameter_values(
            self.num_trials
        )

        # configure number of trials ( for maggy progress bar )
        if self.pruner:
            self.num_trials = (
                self.pruner.num_trials()
            )  # todo does not work yet, progress bar uses num_trials from kwarg of optimizer

    def get_suggestion(self, trial=None):
        try:
            self._log("### start get_suggestion ###")
            self.sampling_time_start = time.time()

            # sampling routine for randomsearch + pruner
            if self.pruner:
                next_trial_info = self.pruner.pruning_routine()
                if next_trial_info == "IDLE":
                    self._log(
                        "Worker is IDLE and has to wait until a new trial can be scheduled"
                    )
                    return "IDLE"
                elif next_trial_info is None:
                    # experiment is finished
                    self._log("Experiment has finished")
                    return None
                elif next_trial_info["trial_id"]:
                    # copy hparams of given promoted trial and start new trial with it
                    parent_trial_id = next_trial_info["trial_id"]
                    parent_trial_hparams = deepcopy(
                        self.get_hparams_dict(trial_ids=parent_trial_id)[
                            parent_trial_id
                        ]
                    )
                    # update trial info dict and create new trial object
                    next_trial = self.create_trial(
                        hparams=parent_trial_hparams,
                        sample_type="promoted",
                        run_budget=next_trial_info["budget"],
                    )
                    self._log(
                        "use hparams from promoted trial {}".format(parent_trial_id)
                    )
                else:
                    # start sampling procedure with given budget
                    parent_trial_id = None
                    run_budget = next_trial_info["budget"]
                    hparams = self.searchspace.get_random_parameter_values(1)[0]
                    next_trial = self.create_trial(
                        hparams=hparams, sample_type="random", run_budget=run_budget
                    )

                # report new trial id to pruner
                self.pruner.report_trial(
                    original_trial_id=parent_trial_id, new_trial_id=next_trial.trial_id
                )

                self._log(
                    "start trial {}: {}. info_dict: {} \n".format(
                        next_trial.trial_id, next_trial.params, next_trial.info_dict
                    )
                )
                return next_trial

            # sampling routine for pure random search
            elif self.config_buffer:
                run_budget = 0
                next_trial_params = self.config_buffer.pop()
                next_trial = self.create_trial(
                    hparams=next_trial_params,
                    sample_type="random",
                    run_budget=run_budget,
                )

                self._log(
                    "start trial {}: {}, {} \n".format(
                        next_trial.trial_id, next_trial.params, next_trial.info_dict
                    )
                )

                return next_trial
            else:
                return None

        except BaseException:
            self._log(traceback.format_exc())
            self._close_log()

            if self.pruner:
                self.pruner._close_log()

    def finalize_experiment(self, trials):
        self._log("Experiment finished")
        self._close_log()

        if self.pruner:
            self.pruner._close_log()

        return

    # todo methods below are duplicated from `bayes/base.py`. If decided to keep `randomsearch.py` like this,
    #    move methods below to `abstractoptimizer.py`

    def get_hparams_dict(self, trial_ids="all"):
        """returns dict of hparams of finished trials with `trial_id` as key and hparams dict as value

        :param trial_ids: trial_id or list of trial_ids that should be returned.
                          If set to default ("all"), return all trials
        :type trial_ids: list[str]|str
        :return: dict of trial_ids and hparams. Example: {`trial_id1`: `hparam_dict1`, ... , `trial_idn`: `hparam_dictn`}
        :rtype: dict
        """
        # return trials with specified trial_ids or return all trials
        include_trial = (
            lambda x: x == trial_ids or x in trial_ids or trial_ids == "all"
        )  # noqa: E731

        hparam_dict = {
            trial.trial_id: trial.params
            for trial in self.final_store
            if include_trial(trial.trial_id)
        }

        return hparam_dict

    def get_metrics_dict(self, trial_ids="all"):
        """returns dict of final metrics with `trial_id` as key and final metric as value

        In case that the optimization `direction` is `max`, negate the metrics so it becomes a `min` problem

        :param trial_ids: trial_id or list of trial_ids that should be returned.
                          If set to default ("all"), return all trials
        :type trial_ids: list[str]|str
        :return: dict of trial_ids and final_metrics. Example: {`trial_id1`: `metric1`, ... , `trial_idn`: `metricn`}
        :rtype: dict
        """
        if self.direction == "max":
            metric_multiplier = -1
        else:
            metric_multiplier = 1

        # return trials with specified trial_ids or return all trials
        include_trial = (
            lambda x: x == trial_ids or x in trial_ids or trial_ids == "all"
        )  # noqa: E731

        metrics = {
            trial.trial_id: trial.final_metric * metric_multiplier
            for trial in self.final_store
            if include_trial(trial.trial_id)
        }

        return metrics

    def init_pruner(self, pruner, pruner_kwargs):
        """intializes pruner

        :param pruner: name of pruner. so far only "hyperband" implemented
        :type pruner: str
        :param pruner_kwargs: dict of pruner kwargs
        :type pruner_kwargs: dict
        :return: initiated pruner instance
        """
        allowed_pruners = ["hyperband"]
        if pruner not in allowed_pruners:
            raise ValueError(
                "expected pruner to be in {}, got {}".format(allowed_pruners, pruner)
            )

        if pruner == "hyperband":
            self.pruner = Hyperband(
                trial_metric_getter=self.get_metrics_dict, **pruner_kwargs
            )

    def create_trial(self, hparams, sample_type, run_budget=0, model_budget=None):
        """helper function to create trial with budget and trial_dict

        `run_budget == 0` means that it is a single fidelity optimization and budget does not need to be passed to Trial
        and hence training function

        :param hparams: hparam dict
        :type hparams: dict
        :param sample_type: specifies how the hapram config was sampled.
                            can take values:
                                - "random": config was sampled randomly
                                - "random_forced": config was sampled randomly because model sampling returned a config that
                                             already exists
                                - "promoted": config was promoted in multi fidelity bandit based setting
                                - "model": config was sampled from surrogate by optimizing acquisiton function
        :type sample_type: str
        :param run_budget: budget for trial or 0 if there is no budget, i.e. single fidelity optimization
        :type run_budget: int
        :param model_budget: If sample_type == `model`, specifies from which model the sample was generated
        :type model_budget: int
        :return: Trial object with specified params
        :rtype: Trial
        """
        # validations
        allowed_sample_type_values = ["random", "random_forced", "model", "promoted"]
        if sample_type not in allowed_sample_type_values:
            raise ValueError(
                "expected sample_type to be in {}, got {}".format(
                    allowed_sample_type_values, sample_type
                )
            )
        if sample_type == "model" and model_budget is None:
            raise ValueError(
                "expected `model_budget` because sample_type==`model`, got None"
            )

        # calculate time needed for sampling
        sampling_time = time.time() - self.sampling_time_start
        self.sampling_time_start = 0.0

        # init trial info dict
        trial_info_dict = {
            "run_budget": run_budget,
            "sample_type": sample_type,
            "sampling_time": sampling_time,
        }
        if model_budget is not None:
            trial_info_dict["model_budget"] = model_budget

        # todo legacy → in the long run have budget as explicit attr of trial object
        if run_budget > 0:
            hparams["budget"] = run_budget

        return Trial(hparams, trial_type="optimization", info_dict=trial_info_dict)

    def _log(self, msg):
        if not self.fd.closed:
            msg = datetime.now().isoformat() + ": " + str(msg)
            self.fd.write((msg + "\n").encode())

    def _close_log(self):
        if not self.fd.closed:
            self.fd.flush()
            self.fd.close()
