
from enum import Enum

import numpy as np
import scipy

zeta_space = np.linspace(0, 1, 101)

def get_prominence():
    return 0.005

class ZetaDecomp(Enum):
    TOPOUT = 1
    MCD = 2

class PeakTrace:
    """
    A trace of a peak. The classes are:

    UNCLASSIFIED, DISAPPEARING, APPEARING, STABLE, NOISE, SHADOW
    """
    UNCLASSIFIED = 0
    DISAPPEARING = 1
    APPEARING = 2
    STABLE = 3
    NOISE = -1
    SHADOW = 101

    colors = {NOISE: "gray", UNCLASSIFIED: "black", DISAPPEARING: "green", APPEARING: "red", STABLE: "blue", SHADOW: "magenta"}

    names = {NOISE: "noise", UNCLASSIFIED: "unclassified", DISAPPEARING: "disappearing", APPEARING: "appearing", STABLE: "stable", SHADOW: "negative"}


    def __init__(self, x, y, data, group=0):
        """ Both args are plain numpy arrays """
        y = np.asarray(y)

        self.x = np.asarray(x)
        self.y_idx = y
        self.y = zeta_space[y]
        data = np.array(data)
        self.prom = data[:,0]
        self.left_base = data[:,1].astype(int)
        self.right_base = data[:,2].astype(int)
        self.values = data[:,3]

        self.group = group

    def __len__(self):
        return len(self.x)

    def __getitem__(self, splice):
        return self.x[splice], self.y[splice], self.prom[splice]

    def summarize(self):
        return f"Length {len(self)}; {self.x[0]}->{self.x[-1]}; {self.y[0]}->{self.y[-1]}; {self.prom[0]}->{self.prom[-1]}"

def trace_vertical_lines(data, n=2, threshold=None):
    """
    Trace vertical-ish lines in a 2D array with a specific start-order:
    last row, first row, second-last, second, third-last, third, ...

    data: 2D array of tuples. Row 0 is top; last row is data[-1].
    n: window half-width (int). At each step search columns [x-n, x+n].
    threshold: minimum height of a peak (float).

    Return the list of PeakTraces
    """
    if threshold is None:
        threshold = get_prominence()

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError("data must be 3D")
    rows, cols, _ = data.shape

    used = np.zeros((rows, cols), dtype=bool)
    lines = []

    # Build start-row order: last, first, second-last, second, ...
    order = []
    low, high = 0, rows - 1
    take_high = True
    while low <= high:
        order.append(high if take_high else low)
        if take_high:
            high -= 1
        else:
            low += 1
        take_high = not take_high

    remaining = True
    while remaining:
        remaining = False
        for start_row in order:
            row_vals = data[start_row, :, 0]
            candidates = np.where((row_vals > threshold) & (~used[start_row]))[0]
            if candidates.size == 0:
                continue
            remaining = True
            # process stronger peaks first
            candidates = candidates[np.argsort(row_vals[candidates])[::-1]]
            for start_x in candidates:
                if used[start_row, start_x]:
                    continue
                x = int(start_x)
                y = int(start_row)
                start_val = float(data[y, x, 0])
                if start_val <= threshold:
                    continue
                line = []
                used[y, x] = True
                line.append((x, y, data[y, x]))

                # decide direction: -1 for up, +1 for down
                direction = -1 if start_row >= (rows / 2) else 1

                yy = y
                while True:
                    yy_next = yy + direction
                    if yy_next < 0 or yy_next >= rows:
                        break
                    left = max(0, x - n)
                    right = min(cols - 1, x + n)
                    window = data[yy_next, left:right + 1, 0]
                    if window.size == 0:
                        break
                    local_idx = np.argmax(window)
                    new_x = left + int(local_idx)
                    new_val = float(data[yy_next, new_x, 0])
                    if new_val <= threshold:
                        break
                    x = new_x
                    yy = yy_next
                    used[yy, x] = True
                    line.append((x, yy, data[yy_next, new_x]))

                x_vals, y_vals, z_vals = zip(*line)

                lines.append(PeakTrace(x_vals, y_vals, z_vals))
    return lines

def get_prominence_spectrum(spectrum, baseline=0, minima=False):
    """
    Return an array of the same shape as "spectrum" with a peak's
    prominence and bases where there is a peak, otherwise 0.
    """
    x = np.asarray(spectrum)
    if x.ndim != 1:
        raise ValueError("spectrum must be 1D")

    if minima:
        x = -x
        x[x < -baseline] = -baseline
    else:
        x[x < baseline] = baseline

    # Find peaks
    peaks, props = scipy.signal.find_peaks(x)

    if peaks.size == 0:
        return np.zeros((*x.shape, 4), dtype=float)

    # Compute prominences for the peaks found
    prominences, left_bases, right_bases = scipy.signal.peak_prominences(x, peaks)
    values = x[peaks]

    # Create output array and place prominence values at peak indices
    prom_array = np.zeros((*x.shape, 4), dtype=float)
    prom_array[peaks, :] = np.vstack((prominences, left_bases, right_bases, values)).T

    return prom_array

def match_peaks(A, B, max_shift_limit=3):
    """
    Match items from A to B so abs(a-b) <= shift where shift is increased
    from 0..max_shift_limit, so better matches take precedence over reducing leftovers.
    Each item can appear in only one pair.

    Return a list of (a_value, b_value)
    """
    # Work on sorted lists but keep original values
    a_sorted = sorted(A)
    b_sorted = sorted(B)

    # Track used indices
    used_a = [False] * len(a_sorted)
    used_b = [False] * len(b_sorted)

    pairs = []

    # Build index maps to original values if needed (we only return values)
    for shift in range(0, max_shift_limit + 1):
        # Two-pointer scan to find candidate pairs with abs diff <= shift
        i = 0
        j = 0
        while i < len(a_sorted) and j < len(b_sorted):
            if used_a[i]:
                i += 1
                continue
            if used_b[j]:
                j += 1
                continue

            a_val = a_sorted[i]
            b_val = b_sorted[j]
            diff = b_val - a_val

            if abs(diff) <= shift:
                # Found a pair within current shift.
                pairs.append((a_val, b_val))
                used_a[i] = True
                used_b[j] = True
                i += 1
                j += 1
            else:
                # Advance the smaller value to try to reduce difference
                if a_val < b_val:
                    i += 1
                else:
                    j += 1

    return sorted(pairs)

def zeta_scan_prominence(start, mixture, baseline=0, minima=False):
    return np.stack([get_prominence_spectrum(mixture - zeta * start, baseline=baseline, minima=minima) for zeta in zeta_space], axis=0)

def get_peak_traces(start, mixture, baseline=0, prominence=None):
    """
    Given a prominence scan, extract the traces of peaks, that is how they
    shift or disappear over the course of the scan.
    """
    if prominence is None:
        prominence = get_prominence()

    prominence_scan = zeta_scan_prominence(start, mixture, minima=False)
    minima_promscan = zeta_scan_prominence(start, mixture, baseline=baseline, minima=True)

    # These are the traces of the peaks observed at zeta=0
    peak_traces = trace_vertical_lines(prominence_scan, threshold=prominence)
    minima_traces = trace_vertical_lines(minima_promscan, threshold=prominence)

    background_maxima = {}
    end_minima = {}

    # Classify them
    for trace in peak_traces:
        # Does not disappear, i.e. product peak or overlap of product and background
        if len(trace) == len(zeta_space):
            trace.group = PeakTrace.STABLE

        elif zeta_space[0] not in trace.y:
            if zeta_space[-1] not in trace.y:
                # Is nonsensical peak
                trace.group = PeakTrace.NOISE
            else:
                # Is product peak
                trace.group = PeakTrace.APPEARING

        # Too short or too shallow
        elif len(trace) < 3 or max(trace.prom) < 3 * get_prominence():
            trace.group = PeakTrace.NOISE

        else:
            # This seems to be a useful background maximum
            trace.group = PeakTrace.DISAPPEARING
            background_maxima[trace.x[0]] = trace

    for trace in minima_traces:
        if len(trace) >= len(zeta_space) - 5:
            continue

        if zeta_space[-1] not in trace.y:
            # Does not start at zeta=1
            continue

        # We don't filter by size because the minima also need to break through the baseline
        trace.group = PeakTrace.SHADOW
        peak_traces.append(trace)
        end_minima[trace.x[0]] = trace

    # Associate minima to maxima
    extremum_pairs = match_peaks(background_maxima.keys(), end_minima.keys())
    extremum_pairs = {background_maxima[bkg]: end_minima[min] for bkg, min in extremum_pairs}

    return peak_traces, end_minima, extremum_pairs

def decompose_zeta_peakcount(start, mixture, baseline=0, include_appearing=False, verbose=False):
    """ Count the number of maxima with a given prominence or minima that come out of them. """
    peak_traces, minimum_traces, extremum_pairs = get_peak_traces(start, mixture, baseline=baseline)

    n_over_zeta = np.zeros(len(zeta_space), dtype=int)

    for trace in peak_traces:
        if trace.group in [PeakTrace.SHADOW, PeakTrace.DISAPPEARING] or include_appearing and trace.group == PeakTrace.APPEARING:
            zeta0 = min(trace.y[0], trace.y[-1])
            zeta1 = max(trace.y[0], trace.y[-1])

            # Optional: If there is a corresponding minimum but no gap due to shifts, open it artificially
            if True:
                if trace.group == PeakTrace.DISAPPEARING:
                    negative = extremum_pairs.get(trace, None)
                    if negative is not None and len(negative) + len(trace) >= len(zeta_space):
                        # Open a gap until twice the prominence at the end of the disappearing trace
                        # Since it is always disappearing, we know the order
                        tmp = zeta1
                        zeta1 = trace.y[np.argmin(np.abs((trace.prom - 2*trace.prom[-1])))]
                        #print("Opening gap at", tmp, "to", zeta1)

            for i_zeta, zeta in enumerate(zeta_space):
                if zeta0 <= zeta <= zeta1:
                    n_over_zeta[i_zeta] += 1

    # Check whether there is one global minimum. If not, take the one closer to the topout value.
    topout_zeta, _, topout_meta = decompose_zeta_topout(start, mixture, verbose=verbose)

    def find_minima(a):
        a = np.asarray(a)

        if a.size == 0:
            return []

        m = int(a.min())
        is_min = (a == m).astype(int)
        diff   = np.diff(is_min)
        starts = np.nonzero(diff == 1)[0]
        ends   = np.nonzero(diff == -1)[0]

        # Push starts to the first point within the valley
        starts = starts + 1

        if len(starts) > len(ends):
            ends = np.append(ends, len(a) - 1)
        elif len(ends) > len(starts):
            starts = np.insert(starts, 0, 0)

        return [(zeta_space[int(s)], zeta_space[int(e)]) for s,e in zip(starts, ends)]

    minima_valleys = find_minima(n_over_zeta)

    if len(minima_valleys) < 1:
        return None

    best_valley = minima_valleys[0]

    vert_lines = [topout_zeta]

    if len(minima_valleys) > 1:
        best_distance = np.inf

        for valley in minima_valleys:
            dist = min(abs(valley[0] - topout_zeta), abs(valley[1] - topout_zeta))
            if dist < best_distance:
                best_distance = dist
                best_valley = valley

        vert_lines.extend(best_valley)

    zeta = np.mean(best_valley)
    
    unmixed = mixture - start * zeta
    
    if verbose:
        metadata = {
            "kind": ZetaDecomp.MCD,
            "background": start,
            "mixture": mixture,
            "result": unmixed,
            "zeta": zeta,
            "n-over-zeta": n_over_zeta,
            "topout-index": topout_meta and topout_meta["topout-index"],
            "topout-zeta": topout_meta and topout_meta["zeta"],
        }
    else:
        metadata = {
            "n-over-zeta": n_over_zeta
        }

    return zeta, unmixed, metadata

def decompose_zeta_topout(background, mixture, verbose=False):
    """
    Scale the background until it doesn't fit vertically into the mixture
    spectrum anymore. This would only work if at least one peak was
    perfectly free of overlap.
    """
    peaks, _ = scipy.signal.find_peaks(background, prominence=get_prominence())

    quot = mixture[peaks] / background[peaks]
    #quot = mixture / background
    zeta = np.nanmin(quot)

    final_spectrum = mixture - background * zeta

    if verbose:
        topind = peaks[np.nanargmin(quot)]
        #topind = np.nanargmin(quot)

        metadata = {
            "kind": ZetaDecomp.TOPOUT,
            "background": background,
            "mixture": mixture,
            "result": final_spectrum,
            "zeta": zeta,
            "topout-index": topind,
        }
    else:
        metadata = None

    return zeta, final_spectrum, metadata

def decompose_zeta_structured(start, mixture, include_appearing=False, baseline=0, verbose=False):
    """
    Try to perform an MCD; if no suitable peaks are found, decompose by non-negativity instead.
    """
    peak_traces, minimum_traces, extremum_pairs = get_peak_traces(start, mixture)

    disappearing_bkg = [trace for trace in peak_traces if trace.group == PeakTrace.DISAPPEARING]

    if len(disappearing_bkg) == 0:
        # There are no disappearing background peaks
        return decompose_zeta_topout(start, mixture, verbose=verbose)

    ret = decompose_zeta_peakcount(start, mixture, include_appearing=include_appearing, baseline=baseline, verbose=verbose)

    if ret is None:
        return decompose_zeta_topout(start, mixture, verbose=verbose)

    return ret

def mcd(background, mixture, verbose=False):
    return decompose_zeta_structured(background, mixture, verbose=verbose)
