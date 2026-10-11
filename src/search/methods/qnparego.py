"""qNParEGO (Ax / BoTorch): same start, constraints and surrogate as qNEHVI, ParEGO random scalarisation."""
from botorch.acquisition.multi_objective.parego import qLogNParEGO

from src.search.methods.qnehvi import QNehvi


class QNParego(QNehvi):
    ACQF = qLogNParEGO
