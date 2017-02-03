import abc
import numpy as np

from . import ParametricModelBaseMixin

from .indexed import IndexedContainer, IndexedContainerException


class HistContainerException(IndexedContainerException):
    pass


class HistContainer(IndexedContainer):
    def __init__(self, n_bins, bin_range, bin_edges=None, fill_data=None, dtype=int):
        super(HistContainer, self).__init__(data=np.zeros(n_bins+2), dtype=dtype)  # underflow and overflow bins
        self._processed_entries = []
        self._unprocessed_entries = []
        # TODO: think of a way to implement weights

        if len(bin_range) != 2:
            raise HistContainerException(
                "Invalid bin range specification: %r! Must be tuple of 2 floats..." % (bin_range,))
        self._bin_range = tuple(bin_range)

        if bin_edges is not None:
            # user specified bin edges -> check that these match the given 'n_bins' and 'bin_range'
            _sz = self.size
            if len(bin_edges) == _sz + 1:
                # assume bin edges given including bin range
                if not (bin_edges[0] == self.low and bin_edges[-1] == self.high):
                    raise HistContainerException(
                        "Invalid bin edge specification! First and last elements %r must match histogram bin range %r."
                        % ((bin_edges[0], bin_edges[-1]), self.bin_range))
                _bin_edges_including_low_and_high = bin_edges
            elif len(bin_edges) == _sz - 1:
                # <check if ordered and strictly within bin range>
                _bin_edges_including_low_and_high = [self._bin_range[0]] + bin_edges + [self._bin_range[1]]
            else:
                raise HistContainerException(
                    "Invalid bin edge specification! Number of elements (%d) must match histogram bin number (%d) +/- 1."
                    % (len(bin_edges), _sz))
            self.rebin(new_bin_edges=_bin_edges_including_low_and_high)
        else:
            # construct own bin edges
            self._bin_edges = np.linspace(self.low, self.high, n_bins+1)

        if fill_data is not None:
            self.fill(fill_data)


    # -- private methods

    def _fill_unprocessed(self):
        if not self._unprocessed_entries:
            return
        _entries_sorted = np.sort(self._unprocessed_entries)

        _current_entry_index = 0
        _current_entry_value = _entries_sorted[0]
        _current_bin_index = 0  # start with underflow bin
        _current_bin_upper_edge = self.low
        _current_bin_lower_edge = -np.inf

        # print "Bin %d (%5.3f, %5.3f)" % (_current_bin_index, _current_bin_lower_edge, _current_bin_upper_edge)
        while _current_bin_upper_edge <= self.high:
            if _current_entry_value >= _current_bin_upper_edge:
                # if last bin, quit
                if _current_bin_upper_edge == self.high:
                    # print "\tthis is the last bin: quitting loop"
                    break
                # move to next bin
                _current_bin_index += 1
                _current_bin_upper_edge = self._bin_edges[_current_bin_index]
                _current_bin_lower_edge = self._bin_edges[_current_bin_index - 1]
                # print "\tmove to next bin"
                # print "Bin %d (%5.3f, %5.3f)" % (_current_bin_index, _current_bin_lower_edge, _current_bin_upper_edge)
                continue
            # increment histogram bin and move to next entry
            # print "\tputting entry '%s' in bin %d (%5.3f, %5.3f)" % (
            # _current_entry_value, _current_bin_index, _current_bin_lower_edge, _current_bin_upper_edge)
            # add to processed entries
            self._processed_entries.append(_current_entry_value)
            self._idx_data[_current_bin_index] += 1
            _current_entry_index += 1
            _current_entry_value = _entries_sorted[_current_entry_index]

        # remaining entries are overflows
        _overflow_entries = _entries_sorted[_current_entry_index:]
        self._idx_data[-1] += len(_overflow_entries)
        # print "Bin %d (of): add %d entries" % (_current_bin_index+1, len(_overflow_entries))
        # for i in _overflow_entries:
        #     print '\t', i
        self._processed_entries += list(_overflow_entries)

    # -- public properties

    @property
    def size(self):
        return len(self._idx_data) - 2  # don't consider underflow and overflow bins

    @property
    def data(self):
        if self._unprocessed_entries:  # process outstanding entries
            self._fill_unprocessed()
        return self._idx_data[1:-1]  # don't consider underflow and overflow bins

    @data.setter
    def data(self, data):
        raise HistContainerException("Changing histogram data directly is not allowed!")

    @property
    def low(self):
        return self._bin_range[0]

    @property
    def high(self):
        return self._bin_range[1]

    @property
    def bin_range(self):
        return self._bin_range

    @property
    def overflow(self):
        return self._idx_data[-1]

    @property
    def underflow(self):
        return self._idx_data[0]

    # -- public methods

    def fill(self, entries):
       try:
           self._unprocessed_entries += list(entries)
       except TypeError:
           self._unprocessed_entries.append(entries)

    def rebin(self, new_bin_edges):
        _new_bin_edges = np.asarray(new_bin_edges, dtype=float)
        # check if list is sorted
        if not (np.diff(_new_bin_edges) >= 0).all():
            raise HistContainerException(
                "Invalid bin edge specification! Edge sequence must be sorted in ascending order!")
        self._bin_edges = _new_bin_edges
        self._idx_data = np.zeros(len(self._bin_edges) -1 + 2)

        # mark all entries as unprocessed
        self._unprocessed_entries += self._processed_entries
        self._processed_entries = []


class HistParametricModelException(HistContainerException):
    pass


class HistParametricModel(ParametricModelBaseMixin, HistContainer):
    def __init__(self, n_bins, bin_range, model_func, model_parameters, bin_edges=None, model_func_antiderivative=None):
        # print "IndexedParametricModel.__init__(model_func=%r, model_parameters=%r)" % (model_func, model_parameters)
        self._model_func_antider_handle = model_func_antiderivative
        super(HistParametricModel, self).__init__(model_func, model_parameters, n_bins, bin_range,
                                                  bin_edges=bin_edges, fill_data=None, dtype=float)

    # -- private methods

    def _eval_model_func_integral_over_bins(self):
        _as = self._bin_edges[:-1]
        _bs = self._bin_edges[1:]
        assert len(_as) == len(_bs) == self.size
        if self._model_func_antider_handle is not None:
            _fval_antider_as = self._model_func_antider_handle(_as, *self._model_parameters)
            _fval_antider_bs = self._model_func_antider_handle(_bs, *self._model_parameters)
            assert len(_fval_antider_as) == len(_fval_antider_bs) == self.size
            _int_val = np.asarray(_fval_antider_bs) - np.asarray(_fval_antider_as)
        else:
            import scipy.integrate as integrate
            _integrand_func = lambda x: self._model_function_handle(x, *self._model_parameters)
            # TODO: find more efficient alternative
            _int_val = np.zeros(self.size)
            for _i, (_a, _b) in enumerate(zip(_as, _bs)):
                _int_val[_i], _ = integrate.quad(_integrand_func, _a, _b)
        return _int_val

    def _recalculate(self):
        # don't use parent class setter for 'data' -> set directly
        self._idx_data[1:-1] = self._eval_model_func_integral_over_bins()
        self._pm_calculation_stale = False


    # -- public properties

    @property
    def data(self):
        if self._pm_calculation_stale:
            self._recalculate()
        return super(HistParametricModel, self).data

    @data.setter
    def data(self, new_data):
        raise HistParametricModelException("Parametric model data cannot be set!")

    # -- public methods

    def fill(self, entries):
        raise HistParametricModelException("Parametric model of histogram cannot be filled!")