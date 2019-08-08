from maggy.earlystop import AbstractEarlyStop


class NoStoppingRule(AbstractEarlyStop):
    """The no stopping rule never stops any trials early.
    """
    @staticmethod
    def earlystop_check(to_check, finalized_trials, direction):
        stop = []
        return stop
