import json
import threading
import hashlib


class Trial(object):
    """A Trial object contains all relevant information about the evaluation
    of an hyperparameter combination.

    It is used as shared memory between
    the worker thread and rpc server thread. The server thread performs only
    lookups on the `early_stop` and `params` attributes.
    """

    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    FINALIZED = "FINALIZED"

    def __init__(self, params):
        """Create a new trial object from a hyperparameter combination
        ``params``.

        :param params: A dictionary of Hyperparameters as key value pairs.
        :type params: dict
        """
        self.trial_id = Trial._generate_id(params)
        self.params = params
        self.status = Trial.PENDING
        self.early_stop = False
        self.final_metric = None
        self.metric_history = []
        self.start = None
        self.duration = None
        self.lock = threading.RLock()

    def get_early_stop(self):
        """Return the early stopping flag of the trial."""
        with self.lock:
            return self.early_stop

    def set_early_stop(self):
        """Set the early stopping flag of the trial to true."""
        with self.lock:
            self.early_stop = True

    def append_metric(self, metric):
        """Append a metric from the heartbeats to the history."""
        with self.lock:
            self.metric_history.append(metric)

    @classmethod
    def _generate_id(cls, params):
        """
        Class method to generate a hash from a hyperparameter dictionary.

        All keys in the dictionary have to be strings. The hash is a to 16
        characters truncated md5 hash and stable across processes.

        :param params: Hyperparameters
        :type params: dictionary
        :raises ValueError: All hyperparameter names have to be strings.
        :raises ValueError: Hyperparameters need to be a dictionary.
        :return: Sixteen character truncated md5 hash
        :rtype: str
        """

        # ensure params is a dictionary
        if isinstance(params, dict):
            # check that all keys are strings
            if False in set(isinstance(k, str) for k in params.keys()):
                raise ValueError(
                    'All hyperparameter names have to be strings.')

            return hashlib.md5(
                json.dumps(params, sort_keys=True).encode('utf-8')
                ).hexdigest()[:16]

        raise ValueError("Hyperparameters need to be a dictionary.")

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        obj_dict = {
            "__class__": self.__class__.__name__
        }

        temp_dict = self.__dict__.copy()
        temp_dict.pop('lock')
        temp_dict.pop('start')

        obj_dict.update(temp_dict)

        return obj_dict

    @classmethod
    def from_json(cls, json_str):
        """Creates a Trial instance from a previously json serialized Trial
        object instance.

        :param json_str: String containing the object.
        :type json_str: str
        :raises ValueError: json_str is not a Trial object.
        :return: Instantiated object instance of Trial.
        :rtype: Trial
        """

        temp_dict = json.loads(json_str)
        if temp_dict.get('__class__', None) != 'Trial':
            raise ValueError("json_str is not a Trial object.")
        if temp_dict.get('params', None) is not None:
            instance = cls(temp_dict.get('params'))
            instance.trial_id = temp_dict['trial_id']
            instance.status = temp_dict['status']
            instance.early_stop = temp_dict['early_stop']
            instance.final_metric = temp_dict['final_metric']
            instance.metric_history = temp_dict['metric_history']
            instance.duration = temp_dict['duration']

        return instance
