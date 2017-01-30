import logging

from ..tools import print_dict_recursive

#from kafe.fit.fitters.simple_fitter import SimpleFitter, SimpleFitterIMinuit
from kafe.fit.fitters.nexus_fitter import NexusFitter#, NexusFitterIMinuit

logger = logging.getLogger(__name__)
logging.basicConfig()


def _benchmark_fitter_scipy_optimize(n_fits=1000, n_pts=10):
    import numpy as np
    from kafe.core.nexus import Nexus

    ps = Nexus()

    _x0 = np.arange(n_pts)
    ps.new(x_support=_x0)
    ps.new(y_measured=_x0)
    ps.new(y_errors=np.ones(n_pts) * 1.0)
    ps.new(a=1)
    ps.new(b=0)

    def y_theory(x_support, a, b):
        return a * x_support + b

    def chi2(y_measured, y_theory, y_errors):
        _p = y_measured - y_theory
        return np.sum(_p/y_errors) ** 2

    ps.new_function(y_theory)
    ps.new_function(chi2)

    sf = NexusFitter(ps, ('a', 'b'), 'chi2')

    # _par_chi2 = ps.get('chi2')
    # _par_chi2_2 = ps._get_par_obj(_par_chi2)

    for i in xrange(n_fits):
        _yd = np.random.normal(0, 1, n_pts)
        ps.set(y_measured=np.arange(n_pts)+_yd)
        sf.do_fit()

def _benchmark_direct_curve_fit(n_fits=1000, n_pts=10):
    import numpy as np
    from scipy.optimize import minimize

    _x0 = np.arange(n_pts)
    x_support = _x0
    y_measured = _x0
    y_errors = np.ones(n_pts) * 1.0
    a = 1.0
    b = 0.0

    def y_theory(x, a=1, b=1):
        return a * x + b

    # _par_chi2 = ps.get('chi2')
    # _par_chi2_2 = ps._get_par_obj(_par_chi2)

    for i in xrange(n_fits):
        _yd = np.random.normal(0, 1, n_pts)
        y_measured = _x0 + _yd

        def chi2((a, b)):
            #global y_errors, x_support
            _p = y_measured - y_theory(x_support, a, b)
            return np.sum(_p / y_errors) ** 2

        # curve_fit(f, xdata, ydata,
        #           p0=None, sigma=None, absolute_sigma=False,
        #           check_finite=True, bounds=(-np.inf, np.inf), method=None)

        minimize(chi2, (1, 0),
                 args=(), method=None, jac=None, hess=None,
                 hessp=None, bounds=None, constraints=(), tol=None,
                 callback=None, options=None)

def _benchmark_fitter_iminuit(n_fits=1000, n_pts=10):
    import numpy as np
    from kafe.core.nexus import Nexus

    ps = Nexus()

    _x0 = np.arange(n_pts)
    ps.new(x_support=_x0)
    ps.new(y_measured=_x0)
    ps.new(y_errors=np.ones(n_pts) * 1.0)
    ps.new(a=1)
    ps.new(b=0)

    def y_theory(x_support, a, b):
        return a * x_support + b

    def chi2(y_measured, y_theory, y_errors):
        _p = y_measured - y_theory
        return np.sum(_p/y_errors) ** 2

    ps.new_function(y_theory)
    ps.new_function(chi2)

    sf = NexusFitterIMinuit(ps, ('a', 'b'), 'chi2')

    # _par_chi2 = ps.get('chi2')
    # _par_chi2_2 = ps._get_par_obj(_par_chi2)

    for i in xrange(n_fits):
        _yd = np.random.normal(0, 1, n_pts)
        ps.set(y_measured=np.arange(n_pts)+_yd)
        sf.do_fit()

def _aliased_multifit_try1(n_pts=10):
    import numpy as np
    from kafe.core.nexus import Nexus

    ps = Nexus()

    _x0 = np.arange(n_pts)
    _y0_1 = np.arange(n_pts)
    _y0_2 = np.arange(n_pts) * 2.0
    ps.new(x_support=_x0)
    ps.new(y1_measured=_y0_1)
    ps.new(y2_measured=_y0_2)
    ps.new(y1_errors=np.ones(n_pts) * 1.0)
    ps.new(y2_errors=np.ones(n_pts) * 2.0)

    ps.new(a1=1)
    ps.new(a2=1)
    ps.new(b1=0)
    ps.new(b2=0)

    @ps.new_function
    def y1_theory(x_support, a1, b1):
        return a1 * x_support + b1

    @ps.new_function
    def y2_theory(x_support, a2, b2):
        return a2 * x_support + b2

    @ps.new_function
    def chi2_1(y1_measured, y1_theory, y1_errors):
        _p = y1_measured - y1_theory
        return np.sum(_p/y1_errors) ** 2

    @ps.new_function
    def chi2_2(y2_measured, y2_theory, y2_errors):
        _p = y2_measured - y2_theory
        return np.sum(_p/y2_errors) ** 2

    @ps.new_function
    def chi2_total(chi2_1, chi2_2):
        return chi2_1 + chi2_2

    #ps.new_function(y1_theory)
    #ps.new_function(chi2_1)
    #ps.new_function(y2_theory)
    #ps.new_function(chi2_2)
    #ps.new_function(chi2_total)

    #ps.new_alias(b_alias='b')

    sf1 = NexusFitterIMinuit(ps, ('a1', 'b1'), 'chi2_1')
    sf = NexusFitterIMinuit(ps, ('a1', 'a2', 'b1', 'b2'), 'chi2_total')

    # _par_chi2 = ps.get('chi2')
    # _par_chi2_2 = ps._get_par_obj(_par_chi2)

    _yd1 = np.random.normal(0, 1, n_pts)
    _yd2 = np.random.normal(0, 1, n_pts)
    ps.set(y1_measured=_y0_1+_yd1)
    ps.set(y2_measured=_y0_2+_yd2)
    sf.do_fit()

    print ""
    print "post-fit"
    print_dict_recursive(sf.fit_parameter_values)

    print_dict_recursive(ps.parameter_values_dict)

    print sf._NexusFitterIMinuit__iminuit.covariance
    print sf._NexusFitterIMinuit__iminuit.errors
    print sf._NexusFitterIMinuit__iminuit.values

if __name__ == "__main__":
    # import numpy as np
    # from tools import print_dict_recursive
    # from core.parameter_space import ParameterSpace
    #
    # ps = ParameterSpace()
    # N_POINTS = 10
    # ps.new(x_support=np.arange(N_POINTS))
    # ps.new(y_measured=np.arange(N_POINTS)+np.random.normal(0, 1, N_POINTS))
    # ps.new(y_errors=np.ones(N_POINTS) * 1.0)
    # ps.new(a=1)
    # ps.new(b=0)
    #
    # def y_theory(x_support, a, b):
    #     return a * x_support + b
    #
    # def chi2(y_measured, y_theory, y_errors):
    #     _p = y_measured - y_theory
    #     return np.sum(_p/y_errors) ** 2
    #
    # ps.new_function(y_theory)
    # ps.new_function(chi2)
    #
    # #sf = NexusFitter(ps, ('a', 'b'), 'chi2')
    # sf = NexusFitterIMinuit(ps, ('a', 'b'), 'chi2')
    #
    # print ""
    # print "pre-fit"
    # print_dict_recursive(sf.fit_parameter_values)
    #
    # sf.do_fit()
    #
    # print ""
    # print "post-fit"
    # print_dict_recursive(sf.fit_parameter_values)

    # N_FITS = 10

    # import timeit
    # from tools import print_dict_recursive
    #
    # _t = dict(scipy={}, curve={}, iminuit={})
    # #for n in (10, 100, 1000, 10000, 100000):
    # for n in (10, 100, 1000, 10000, 100000):
    #     _t['curve'][n] = timeit.timeit('_benchmark_direct_curve_fit(%d)' % (n,),
    #                                    setup="from __main__ import _benchmark_direct_curve_fit",
    #                                    number=1)
    #     _t['scipy'][n] = timeit.timeit('_benchmark_fitter_scipy_optimize(%d)' % (n,),
    #                                    setup="from __main__ import _benchmark_fitter_scipy_optimize",
    #                                    number=1)
    #     _t['iminuit'][n] = timeit.timeit('_benchmark_fitter_iminuit(%d)' % (n,),
    #                                      setup="from __main__ import _benchmark_fitter_iminuit",
    #                                      number=1)
    #
    # print_dict_recursive(_t)

    #_aliased_multifit_try1(10)

    pass