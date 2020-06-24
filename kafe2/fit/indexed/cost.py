from .._base.cost import CostFunction, CostFunction_Chi2, CostFunction_NegLogLikelihood


__all__ = [
    "IndexedCostFunction",
    "IndexedCostFunction_Chi2",
    "IndexedCostFunction_NegLogLikelihood"
]


class IndexedCostFunction(CostFunction):
    pass


class IndexedCostFunction_Chi2(CostFunction_Chi2):
    pass


class IndexedCostFunction_NegLogLikelihood(CostFunction_NegLogLikelihood):
    pass

