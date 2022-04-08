from .._base.cost import CostFunction, CostFunction_Chi2, CostFunction_NegLogLikelihood, \
    CostFunction_GaussApproximation


__all__ = [
    "HistCostFunction",
    "HistCostFunction_Chi2",
    "HistCostFunction_NegLogLikelihood",
    "HistCostFunction_GaussApproximation"
]


class HistCostFunction(CostFunction):
    pass


class HistCostFunction_Chi2(CostFunction_Chi2):
    pass


class HistCostFunction_NegLogLikelihood(CostFunction_NegLogLikelihood):
    pass


class HistCostFunction_GaussApproximation(CostFunction_GaussApproximation):
    pass
