"""
The experiment driver implements the functionality for scheduling trials on maggy.
"""
import queue
import threading
import json
import os
import secrets
from datetime import datetime
from maggy import util
from maggy.optimizer import AbstractOptimizer, RandomSearch
from maggy.core import config, rpc
from maggy.trial import Trial
from maggy.earlystop import AbstractEarlyStop, MedianStoppingRule
from maggy.searchspace import Searchspace

from hops import constants as hopsconstants
from hops import hdfs as hopshdfs
from hops import util as hopsutil


class ExperimentDriver(object):

    SECRET_BYTES = 8

    def __init__(self, searchspace, optimizer, direction, num_trials, name, num_executors, hb_interval, es_policy, es_interval, es_min, description, log_dir, trial_dir):

        self._final_store = []

        # perform type checks
        if isinstance(searchspace, Searchspace):
            self.searchspace = searchspace
        else:
            raise Exception(
                "No valid searchspace. Please use maggy Searchspace class.")

        if isinstance(optimizer, str):
            if optimizer.lower() == 'randomsearch':
                self.optimizer = RandomSearch(num_trials, self.searchspace, self._final_store)
            else:
                raise Exception(
                    "Unknown Optimizer. Can't initialize experiment driver.")
        elif isinstance(optimizer, AbstractOptimizer):
            print("Custom Optimizer initialized.")
            self.optimizer = optimizer
        else:
            raise Exception(
                "Unknown Optimizer. Can't initialize experiment driver.")

        if isinstance(direction, str):
            if direction.lower() not in ['min', 'max']:
                raise Exception(
                    "Unknown direction. Can't initialize experiment driver.")
        else:
            raise Exception(
                "Unknown direction. Can't initialize experiment driver.")

        if isinstance(es_policy, str):
            if es_policy.lower() == 'median':
                self.earlystop_check = MedianStoppingRule.earlystop_check
            else:
                raise Exception(
                    "Unknown Early Stopping Policy. Can't initialize experiment driver.")
        elif isinstance(es_policy, AbstractEarlyStop):
            print("Custom Early Esrly Stopping policy initialized.")
            self.earlystop_check = es_policy.earlystop_check

        self.direction = direction.lower()
        self._trial_store = {}
        self.num_executors = num_executors
        self._message_q = queue.Queue()
        self.name = name
        self.num_trials = num_trials
        self.experiment_done = False
        self.worker_done = False
        self.hb_interval = hb_interval
        self.es_interval = es_interval
        self.es_min = es_min
        self.description = description
        self.direction = direction.lower()
        self.server = rpc.Server(num_executors)
        self._secret = self._generate_secret(ExperimentDriver.SECRET_BYTES)
        self.result = None
        self.job_start = datetime.now()
        self.executor_logs = ''
        self.maggy_log = ''
        self.log_lock = threading.RLock()
        self.log_file = log_dir+ '/maggy.log'
        self.trial_dir = trial_dir

        #Open File desc for HDFS to log
        if not hopshdfs.exists(self.log_file):
            hopshdfs.dump('', self.log_file)
        self.fd = hopshdfs.open_file(self.log_file, flags='w')

    def init(self):

        self.server_addr = self.server.start(self)

        self.optimizer.initialize()

        self._start_worker()

    def finalize(self, job_start, job_end):

        _ = self.optimizer.finalize_experiment(self._final_store)

        self.job_end = datetime.now()

        self.duration = hopsutil._time_diff(self.job_start, self.job_end)

        if self.direction == 'max':
            results = '\n------ ' + str(self.optimizer.__class__.__name__) + ' results ------ direction(' + self.direction + ') \n' \
                'BEST combination ' + json.dumps(self.result['max_hp']) + ' -- metric ' + str(self.result['max_val']) + '\n' \
                'WORST combination ' + json.dumps(self.result['min_hp']) + ' -- metric ' + str(self.result['min_val']) + '\n' \
                'AVERAGE metric -- ' + str(self.result['avg']) + '\n' \
                'EARLY STOPPED Trials -- ' + str(self.result['early_stopped']) + '\n' \
                'Total job time ' + self.duration + '\n'
            # TODO: write to hdfs
            print(results)
        elif self.direction == 'min':
            results = '\n------ ' + str(self.optimizer.__class__.__name__) + ' results ------ direction(' + self.direction + ') \n' \
                'BEST combination ' + json.dumps(self.result['min_hp']) + ' -- metric ' + str(self.result['min_val']) + '\n' \
                'WORST combination ' + json.dumps(self.result['max_hp']) + ' -- metric ' + str(self.result['max_val']) + '\n' \
                'AVERAGE metric -- ' + str(self.result['avg']) + '\n' \
                'EARLY STOPPED Trials -- ' + str(self.result['early_stopped']) + '\n' \
                'Total job time ' + self.duration + '\n'
            # TODO: write to hdfs
            print(results)

        return self.result

    def get_trial(self, trial_id):
        return self._trial_store[trial_id]

    def add_trial(self, trial):
        self._trial_store[trial.trial_id] = trial

    def add_message(self, msg):
        self._message_q.put(msg)

    def _start_worker(self):

        def _target_function(self):

            time_earlystop_check = datetime.now()

            while not self.worker_done:
                trial = None
                # get a message
                try:
                    msg = self._message_q.get_nowait()
                except:
                    msg = {'type': None}

                if (datetime.now() - time_earlystop_check).total_seconds() >= self.es_interval:
                    time_earlystop_check = datetime.now()

                    # pass currently running trials to early stop component
                    if len(self._final_store) > self.es_min:
                        self._log("Check for early stopping.")
                        to_stop = self.earlystop_check(
                            self._trial_store, self._final_store, self.direction)
                        if len(to_stop) > 0:
                            self._log("Trials to stop: {}".format(to_stop))
                        for trial_id in to_stop:
                            self.get_trial(trial_id).set_early_stop()

                # depending on message do the work
                # 1. METRIC
                if msg['type'] == 'METRIC':
                    self.get_trial(msg['trial_id']).append_metric(msg['data'])

                    # append executor logs if in the message
                    logs = msg.get('logs', None)
                    if logs is not None:
                        with self.log_lock:
                            self.executor_logs = self.executor_logs + logs

                # 2. BLACK
                elif msg['type'] == 'BLACK':
                    trial = self.get_trial(msg['trial_id'])
                    with trial.lock:
                        trial.status = Trial.SCHEDULED
                        self.server.reservations.assign_trial(
                            msg['partition_id'], msg['trial_id'])

                # 3. FINAL
                elif msg['type'] == 'FINAL':
                    # set status
                    # get trial only once
                    trial = self.get_trial(msg['trial_id'])

                    # finalize the trial object
                    with trial.lock:
                        trial.status = Trial.FINALIZED
                        trial.final_metric = msg['data']
                        trial.duration = hopsutil._time_diff(
                            trial.start, datetime.now())

                    # move trial to the finalized ones
                    self._final_store.append(trial)
                    self._trial_store.pop(trial.trial_id)

                    # update result dictionary
                    self._update_result(trial)
                    # keep for later in case tqdm doesn't work
                    self.maggy_log = self._update_maggy_log()
                    self._log(self.maggy_log)

                    hopshdfs.dump(trial.to_json(), self.trial_dir + '/' + trial.trial_id + '/trial')

                    # assign new trial
                    trial = self.optimizer.get_suggestion(trial)
                    if trial is None:
                        self.server.reservations.assign_trial(
                            msg['partition_id'], None)
                        self.experiment_done = True
                    else:
                        with trial.lock:
                            trial.start = datetime.now()
                            trial.status = Trial.SCHEDULED
                            self.server.reservations.assign_trial(
                                msg['partition_id'], trial.trial_id)
                            self.add_trial(trial)

                # 4. REG
                elif msg['type'] == 'REG':
                    trial = self.optimizer.get_suggestion()
                    if trial is None:
                        self.experiment_done = True
                    else:
                        with trial.lock:
                            trial.start = datetime.now()
                            trial.status = Trial.SCHEDULED
                            self.server.reservations.assign_trial(
                                msg['partition_id'], trial.trial_id)
                            self.add_trial(trial)

        t = threading.Thread(target=_target_function, args=(self,))
        t.daemon = True
        t.start()

    def stop(self):
        """Stop the Driver's worker thread and server."""
        self.worker_done = True
        self.server.stop()
        self.fd.flush()
        self.fd.close()

    def json(self, sc):
        """Get all relevant experiment information in JSON format.
        """
        user = None
        if hopsconstants.ENV_VARIABLES.HOPSWORKS_USER_ENV_VAR in os.environ:
            user = os.environ[hopsconstants.ENV_VARIABLES.HOPSWORKS_USER_ENV_VAR]

        experiment_json = {'project': hopshdfs.project_name(),
            'user': user,
            'name': self.name,
            'module': 'maggy',
            'function': self.optimizer.__class__.__name__,
            'app_id': str(sc.applicationId),
            'start': self.job_start.isoformat(),
            'memory_per_executor': str(sc._conf.get("spark.executor.memory")),
            'gpus_per_executor': str(sc._conf.get("spark.executor.gpus")),
            'executors': self.num_executors,
            'logdir': self.trial_dir,
            'hyperparameter_space': json.dumps(self.searchspace.to_dict()),
            # 'versioned_resources': versioned_resources,
            'description': self.description}

        if self.experiment_done:
            experiment_json['status'] = "FINISHED"
            experiment_json['finished'] = self.job_end.isoformat()
            experiment_json['duration'] = self.duration
            if self.direction == 'max':
                experiment_json['hyperparameter'] = json.dumps(self.result['max_hp'])
                experiment_json['metric'] = self.result['max_val']
            elif self.direction == 'min':
                experiment_json['hyperparameter'] = json.dumps(self.result['min_hp'])
                experiment_json['metric'] = self.result['min_val']

        else:
            experiment_json['status'] = "RUNNING"

        return json.dumps(experiment_json)

    def _generate_secret(self, nbytes):
        """Generates a secret to be used by all clients during the experiment
        to authenticate their messages with the experiment driver.
        """
        return secrets.token_hex(nbytes=nbytes)

    def _update_result(self, trial):
        """Given a finalized trial updates the current result's best and
        worst trial.
        """

        metric = trial.final_metric
        param_string = trial.params
        trial_id = trial.trial_id

        # First finalized trial
        if self.result is None:
            self.result = {'max_id': trial_id, 'max_val': metric,
                'max_hp': param_string, 'min_id': trial_id,
                'min_val': metric, 'min_hp': param_string,
                'avg': metric, 'metric_list': [metric], 'num_trials': 1,
                'early_stopped': 0}

            if trial.early_stop:
                self.result['early_stopped'] += 1

            return

        if metric > self.result['max_val']:
            self.result['max_val'] = metric
            self.result['max_id'] = trial_id
            self.result['max_hp'] = param_string
        if metric < self.result['min_val']:
            self.result['min_val'] = metric
            self.result['min_id'] = trial_id
            self.result['min_hp'] = param_string

        # update average
        self.result['metric_list'].append(metric)
        self.result['num_trials'] += 1
        self.result['avg'] = sum(self.result['metric_list'])/float(
            len(self.result['metric_list']))

        if trial.early_stop:
                self.result['early_stopped'] += 1

    def _update_maggy_log(self):
        """Creates the status of a maggy experiment with a progress bar.
        """
        finished = self.result['num_trials']

        log = 'Maggy ' + str(finished) + '/' + str(self.num_trials) + \
            ' (' + str(self.result['early_stopped']) + ') ' + \
            util._progress_bar(finished, self.num_trials) + ' - BEST ' + \
            json.dumps(self.result['max_hp']) + ' - metric ' + \
            str(self.result['max_val'])

        return log

    def _get_logs(self):
        """Return current experiment status and executor logs to send them to
        spark magic.
        """
        with  self.log_lock:
            temp = self.executor_logs
            # clear the executor logs since they are being sent
            self.executor_logs = ''
            return self.result, temp

    def _log(self, log_msg):
        """Logs a string to the maggy driver log file.
        """
        msg = datetime.now().isoformat() + ': ' + str(log_msg)
        self.fd.write((msg + '\n').encode())
