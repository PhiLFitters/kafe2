import numpy as np

from ..indexed import IndexedContainer
from ..indexed.container import IndexedContainerException


__all__ = ["HistContainer"]


class HistContainerException(IndexedContainerException):
    pass


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
    def __init__(self, n_bins, bin_range, bin_edges=None, fill_data=None, dtype=int):
        """
        Construct a histogram:

        :param n_bins: number of bins
        :type n_bins: int
        :param bin_range: the lower and upper edges of the entire histogram
        :type bin_range: tuple of floats
        :param bin_edges: the bin edges (if ``None``, each bin will have the same width)
        :type bin_edges: list of floats
        :param fill_data: entries to fill into the histogram
        :type fill_data: list of floats
        :param dtype: data type of histogram entries
        :type dtype: type
        """
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
        """fill any entries marked as unprocessed into the histogram"""
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
            # print "\tputting entry #%d '%s' in bin %d (%5.3f, %5.3f)" % (
            #     _current_entry_index, _current_entry_value, _current_bin_index, _current_bin_lower_edge, _current_bin_upper_edge)
            # add to processed entries
            self._processed_entries.append(_current_entry_value)
            self._idx_data[_current_bin_index] += 1
            _current_entry_index += 1
            if _current_entry_index >= len(_entries_sorted):    # important BUGFIX!!!!!!!!!!
                break
            _current_entry_value = _entries_sorted[_current_entry_index]

        # any remaining entries are overflows
        if _current_entry_index < len(_entries_sorted):    # important BUGFIX!!!!!!!!!!

            _overflow_entries = _entries_sorted[_current_entry_index:]
            self._idx_data[-1] += len(_overflow_entries)
            # print "Bin %d (of): add %d entries" % (_current_bin_index+1, len(_overflow_entries))
            # for i in _overflow_entries:
            #     print '\t', i
            self._processed_entries += list(_overflow_entries)

        self._unprocessed_entries = []

    # -- public properties

    @property
    def size(self):
        """the number of bins (excluding underflow and overflow bins)"""
        return len(self._idx_data) - 2  # don't consider underflow and overflow bins

    @property
    def n_entries(self):
        """the number of entries"""
        return len(self._processed_entries) + len(self._unprocessed_entries)

    @property
    def data(self):
        """the number of entries in each bin"""
        if self._unprocessed_entries:  # process outstanding entries
            self._fill_unprocessed()
        # NOTE: returned array starts at 0
        return self._idx_data[1:-1].copy()  # don't consider underflow and overflow bins

    @data.setter
    def data(self, data):
        raise HistContainerException("Changing histogram data directly is not allowed! Use fill().")

    @property
    def raw_data(self):
        """the number of entries in each bin"""
        # TODO: commit unprocessed entries (?)
        return self._processed_entries + self._unprocessed_entries

    @property
    def low(self):
        """the lower edge of the histogram"""
        return self._bin_range[0]

    @property
    def high(self):
        """the upper edge of the histogram"""
        return self._bin_range[1]

    @property
    def bin_range(self):
        """a tuple containing the lower and upper edges of the histogram"""
        return self._bin_range

    @property
    def overflow(self):
        """the number of entries in the overflow bin"""
        return self._idx_data[-1]

    @property
    def underflow(self):
        """the number of entries in the underflow bin"""
        return self._idx_data[0]

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
        return (_be[1:] - _be[:-1])

    @property
    def bin_centers(self):
        """a list of the (geometrical) bin centers"""
        _be = self._bin_edges
        return 0.5*(_be[1:] + _be[:-1])

    # -- public methods

    def fill(self, entries):
       """
       Fill new entries into the histogram.

       :param entries: list of entries
       :type entries: list of floats
       """
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