from .._base.cost import CostFunction, CostFunction_Chi2, CostFunction_NegLogLikelihood


__all__ = [
    "HistCostFunction",
    "HistCostFunction_Chi2",
    "HistCostFunction_NegLogLikelihood"
]


class HistCostFunction(CostFunction):
    pass


class HistCostFunction_Chi2(CostFunction_Chi2):
    pass


class HistCostFunction_NegLogLikelihood(CostFunction_NegLogLikelihood):
    pass

