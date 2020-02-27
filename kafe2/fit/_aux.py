import numpy as np

from copy import deepcopy

# represent errors in a binwise fashion as rectangles
def step_fill_between(axes,
                      x, y,
                      yerr=None, xerr=None,
                      draw_central_value=False, **kwargs):
    """
    Draw two step functions and fill the area in-between.

    :param axes: target ``Axes`` object
    :type axes: ``matplotlib`` ``Axes`` object
    :param x: *x* coordinates of the steps
    :type x: list of float
    :param y: *y* coordinates of the steps
    :type y: list of float
    :param yerr: distances between the center and upper step function (or lower and upper, if tuple)
    :type yerr: list of float or list of 2-tuple of float
    :param xerr: distances between the step center and preceding/next step (or both, if tuple)
    :type xerr: list of float or list of 2-tuple of float
    :param draw_central_value: if ``True``, also draw the central step function
    :type draw_central_value: bool
    :param kwargs: keyword arguments accepted by ``matplotlib`` methods ``plot`` and ``fill_between``
    :type kwargs: dict
    :return: plot handle(s)
    """
    # -- give errors common structure: tuple(err_dn, err_up)

    _yerr = yerr
    _xerr = xerr

    if _yerr is not None:
        if isinstance(_yerr, tuple) and len(_yerr)==1:
            _yerr = (yerr[0], yerr[0])
        elif isinstance(_yerr, tuple) and len(_yerr)==2:
            _yerr = (yerr[0], yerr[1])
        else:
            try:
                if _yerr.ndim == 1:
                    _yerr = (yerr, yerr)
                elif _yerr.ndim == 2:
                    _yerr = (yerr[0], yerr[1])
                else:
                    raise ValueError("Cannot interpret error array with shape %s" % (_yerr.shape,))
            except AttributeError:
                _xerr = (xerr, xerr)
    else:
        draw_central_value = True  # force drawing central value, if no y errors

    if _xerr is not None:
        if isinstance(_xerr, tuple) and len(_xerr)==1:
            _xerr = (xerr[0], xerr[0])
        elif isinstance(_xerr, tuple) and len(_xerr)==2:
            _xerr = (xerr[0], xerr[1])
        else:
            try:
                if _xerr.ndim == 1:
                    _xerr = (xerr, xerr)
                elif _xerr.ndim == 2:
                    _xerr = (xerr[0], xerr[1])
                else:
                    raise ValueError("Cannot interpret error array with shape %s" % (_xerr.shape,))
            except AttributeError:
                _xerr = (xerr, xerr)

    # -- do actual plotting

    if _xerr is not None:
        # calculate box corners for fill_between
        _x_lo = x - _xerr[0]
        _x_hi = x + _xerr[1]

        # interleave x and y bin boundaries
        _x_plot = np.zeros(len(x)*2)
        _x_plot[0::2] = _x_lo
        _x_plot[1::2] = _x_hi
    else:
        raise ValueError('xerr cannot be None')

    if draw_central_value:
        _y_plot_central = np.zeros(len(y) * 2)
        _y_plot_central[0::2] = y
        _y_plot_central[1::2] = y
        _modkwargs = deepcopy(kwargs)
        _modkwargs.pop('edgecolor', None)
        _modkwargs.pop('alpha', None)
        p_cv = axes.plot(_x_plot, _y_plot_central, **_modkwargs)
    else:
        p_cv = None

    if _yerr is not None:
        _y_lo = y - _yerr[0]
        _y_hi = y + _yerr[1]

        _y_plot_lo = np.zeros(len(y)*2)
        _y_plot_lo[0::2] = _y_lo
        _y_plot_lo[1::2] = _y_lo
        _y_plot_hi = np.zeros(len(y)*2)
        _y_plot_hi[0::2] = _y_hi
        _y_plot_hi[1::2] = _y_hi

        p_fb = axes.fill_between(_x_plot, _y_plot_lo, _y_plot_hi, **kwargs)
    else:
        p_fb = None

    # FIXME: (for _yerr!=None, return both artists somehow)
    if p_cv is not None and p_fb is not None:
        #return (p_fb, p_cv)  # doesn't work!
        return p_fb
    if p_cv is None:
        return p_fb
    if p_fb is None:
        #return p_cv  # doesn't work!
        return p_cv[0]
