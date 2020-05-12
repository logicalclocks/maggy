from maggy.optimizer.bayes import base, simple, tpe

BaseAsyncBO = base.BaseAsyncBO
GP = simple.GP
TPE = tpe.TPE

__all__ = [
    "TPE",
    "BaseAsyncBO",
    "GP",
]
