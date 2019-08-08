from abc import ABC, abstractmethod


class AbstractAblator(ABC):

    def __init__(self, ablation_study, final_store):
        self.ablation_study = ablation_study
        self.final_store = final_store
        self.trial_buffer = []

    @abstractmethod
    def get_number_of_trials(self):
        pass

    @abstractmethod
    def get_dataset_generator(self, ablated_feature, dataset_type='tfrecord'):
        pass

    @abstractmethod
    def get_model_generator(self, ablated_layer):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_trial(self, ablation_trial=None):
        """
        Return a trial to be assigned to an executor.
        The trial should contain a dataset generator and a model generator.
        Depending on the ablator policy, the trials could come from a list (buffer) of pre-made trials,
        Or generated on the fly.
        """
        pass

    @abstractmethod
    def finalize_experiment(self, trials):
        pass

