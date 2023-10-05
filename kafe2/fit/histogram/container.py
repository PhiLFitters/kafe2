import warnings

import numpy as np

from ..indexed import IndexedContainer

__all__ = ["HistContainer"]


class HistContainer(IndexedContainer):
    """
    This object is a specialized data container for organizing data into *histograms*.

    A histogram is a compact representation of a potentially large number of
    entries which are distributed along a continuum of values. Histograms
    divide the continuum into intervals ("bins") and count the number of entries
    per interval.

    .. .. inheritance-diagram:: HistContainer
    ..    :parts: 1

    """

    def __init__(self, n_bins=None, bin_range=None, bin_edges=None, fill_data=None, dtype=int):
        """
        Construct a histogram:

        :param n_bins: how many bins raw data should be split into.
        :type n_bins: int
        :param bin_range: the lower and upper bound for the bins specified by n_bins.
        :type bin_range: iterable[float] of length 2
        :param bin_edges: explicit bin edges for raw data. If ``None``, each bin will have the same
            width.
        :type bin_edges: iterable[float]
        :param fill_data: entries to fill into the histogram
        :type fill_data: iterable[float]
        :param dtype: data type of histogram entries
        :type dtype: type
        """
        if bin_edges is None:
            if n_bins is None or bin_range is None:
                raise ValueError("Either n_bins and bin_range or bin_edges must be specified.")
        else:
            if n_bins is None:
                n_bins = len(bin_edges) - 1
            if n_bins != len(bin_edges) - 1 and n_bins != len(bin_edges) + 1:
                raise ValueError(f"n_bins is {n_bins} but bin_edges implies either " f"{len(bin_edges) - 1} or {len(bin_edges) - 1} bins")
            if bin_range is None:
                bin_range = (bin_edges[0], bin_edges[-1])
            if n_bins == len(bin_edges) - 1:
                if bin_range[0] != bin_edges[0] or bin_range[1] != bin_edges[-1]:
                    raise ValueError(f"bin_range is {bin_range} but bin_edges implies " f"{(bin_edges[0], bin_edges[-1])}")
            if n_bins == len(bin_edges) + 1:
                if bin_range[0] > bin_edges[0] or bin_range[1] < bin_edges[-1]:
                    raise ValueError(f"bin_range is {bin_range} but does not encompass inner " f"bin edges {(bin_edges[0], bin_edges[-1])}")

        if len(bin_range) != 2:
            raise ValueError(f"bin_range must be iterable of 2 floats but received {bin_range}")
        low, high = tuple(bin_range)

        super(HistContainer, self).__init__(data=np.zeros(n_bins + 2), dtype=dtype)  # underflow and overflow bins
        self._manual_heights = False
        self._processed_entries = []
        self._unprocessed_entries = []
        # TODO: think of a way to implement weights

        if bin_edges is not None:
            if len(bin_edges) == self.size + 1:
                self.rebin(bin_edges)
            elif len(bin_edges) == self.size - 1:
                # Add low and high to edges if inner edges specified:
                self.rebin([low] + list(bin_edges) + [high])
            else:
                assert False
        else:
            # construct own bin edges
            self._bin_edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)

        if fill_data is not None:
            self.fill(fill_data)

    # -- private methods

    def _fill_unprocessed(self):
        """fill any entries marked as unprocessed into the histogram"""
        if self._manual_heights:
            raise RuntimeError(
                "The bin heights have been set manually. Filling entries is not " "available anymore. Please construct a new HistContainer!"
            )
        if not self._unprocessed_entries:
            return
        _entries_sorted = np.sort(self._unprocessed_entries)
        if np.all(self._unprocessed_entries == np.floor(self._unprocessed_entries)):
            warnings.warn("Histogram data is int. Make sure to consider this for your bin edges!")

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
            # print "\tputting entry #%d '%s' in bin %d (%5.3f, %5.3f)" % (
            #     _current_entry_index, _current_entry_value, _current_bin_index, _current_bin_lower_edge, _current_bin_upper_edge)
            # add to processed entries
            self._processed_entries.append(_current_entry_value)
            self._data[_current_bin_index] += 1
            _current_entry_index += 1
            if _current_entry_index >= len(_entries_sorted):  # important BUGFIX!!!!!!!!!!
                break
            _current_entry_value = _entries_sorted[_current_entry_index]

        # any remaining entries are overflows
        if _current_entry_index < len(_entries_sorted):  # important BUGFIX!!!!!!!!!!
            _overflow_entries = _entries_sorted[_current_entry_index:]
            self._data[-1] += len(_overflow_entries)
            # print "Bin %d (of): add %d entries" % (_current_bin_index+1, len(_overflow_entries))
            # for i in _overflow_entries:
            #     print '\t', i
            self._processed_entries += list(_overflow_entries)

        self._unprocessed_entries = []

    def _get_error_reference(self):
        return self._data[1:-1]

    # -- public properties

    @property
    def size(self):
        """the number of bins (excluding underflow and overflow bins)"""
        return len(self._data) - 2  # don't consider underflow and overflow bins

    @property
    def n_entries(self):
        """the number of entries"""
        return np.sum(self._data) + len(self._unprocessed_entries)

    @property
    def data(self):
        """the number of entries in each bin"""
        if self._unprocessed_entries:  # process outstanding entries
            self._fill_unprocessed()
        # NOTE: returned array starts at 0
        return self._data[1:-1].copy()  # don't consider underflow and overflow bins

    @data.setter
    def data(self, data):
        raise TypeError("Changing histogram data directly is not allowed! Use fill() or set_bins().")

    @property
    def raw_data(self):
        """the number of entries in each bin"""
        # TODO: commit unprocessed entries (?)
        return self._processed_entries + self._unprocessed_entries

    @property
    def low(self):
        """the lower edge of the histogram"""
        return self._bin_edges[0]

    @property
    def high(self):
        """the upper edge of the histogram"""
        return self._bin_edges[-1]

    @property
    def bin_range(self):
        """a tuple containing the lower and upper edges of the histogram"""
        return self.low, self.high

    @property
    def overflow(self):
        """the number of entries in the overflow bin"""
        return self._data[-1]

    @property
    def underflow(self):
        """the number of entries in the underflow bin"""
        return self._data[0]

    @property
    def n_bins(self):
        """the number of bins"""
        return len(self._bin_edges) - 1

    @property
    def bin_edges(self):
        """a list of the bin edges (including the outermost ones)"""
        return self._bin_edges.copy()

    @property
    def bin_widths(self):
        """a list of the bin widths"""
        _be = self._bin_edges
        return _be[1:] - _be[:-1]

    @property
    def bin_centers(self):
        """a list of the (geometrical) bin centers"""
        _be = self._bin_edges
        return 0.5 * (_be[1:] + _be[:-1])

    # -- public methods

    def fill(self, entries):
        """
        Fill new entries into the histogram.

        :param entries: list of entries
        :type entries: list of floats
        """
        if self._manual_heights:
            raise RuntimeError(
                "The bin heights have been set manually. Filling additional data is not " "possible anymore. Please construct a new HistContainer!"
            )
        if np.asarray(entries).ndim > 1:
            raise ValueError("Fill data must be scalar or one-dimensional " f"but received fill data with {np.asarray(entries).ndim} dimensions.")
        try:
            self._unprocessed_entries += list(entries)
        except TypeError:
            self._unprocessed_entries.append(entries)

    def rebin(self, new_bin_edges):
        """
        Change the histogram binning.

        :param new_bin_edges: list of new bin edges in ascending order
        :type new_bin_edges: list of float
        """
        if self._manual_heights:
            raise RuntimeError("The bin heights have been set manually. Rebinning is not possible " "anymore. Please construct a new HistContainer!")
        _new_bin_edges = np.asarray(new_bin_edges, dtype=float)
        # check if list is sorted
        if not (np.diff(_new_bin_edges) >= 0).all():
            raise ValueError("Invalid bin edge specification! Edge sequence must be sorted in ascending order!")
        self._bin_edges = _new_bin_edges
        self._data = np.zeros(len(self._bin_edges) - 1 + 2)

        # mark all entries as unprocessed
        self._unprocessed_entries += self._processed_entries
        self._processed_entries = []

    def set_bins(self, bin_heights, underflow=0, overflow=0):
        """
        Set the bin heights according to a pre-calculated histogram
        :param bin_heights: Heights of the bins
        :type bin_heights: list of int
        :param underflow: Number of entries in the underflow bin
        :type underflow: int
        :param overflow: Number of entries in the overflow bin
        :type overflow: int
        """
        self._manual_heights = True
        _new_data = np.array(bin_heights)
        if _new_data.ndim != 1:
            raise ValueError("Invalid dimensions for bin heights. " f"Got {_new_data.ndim}-d array, expected 1-d array")
        _new_data = np.append(np.insert(_new_data, 0, underflow), overflow)
        if len(_new_data) != len(self._data):
            raise ValueError("Length of bin entries does not match binning. " "Got {}, expected {}".format(len(_new_data) - 2, len(self._data) - 2))
        self._data = _new_data
        self._processed_entries = []
        self._unprocessed_entries = []
