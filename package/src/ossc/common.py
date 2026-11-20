
import numpy as np

def get_solvent_spectrum(exp):
    spectrum = exp.loc[exp["stage"] == "Solvent", "ir"].iloc[-1]

    if "ir-stddev" in exp.columns:
        uncertainty = exp.loc[exp["stage"] == "Solvent", "ir-stddev"].iloc[-1]
    else:
        uncertainty = None

    return spectrum, uncertainty

def get_solvent(exp):
    return exp.attrs["chemicals"][0]

def get_noise_mask(std_dev, threshold=0.01, window=11):
    """
    Create a boolean mask where std_dev > threshold, convolved with a
    sliding window.
    """
    if std_dev is None:
        return False

    cond = std_dev > threshold

    kernel = np.ones(window, dtype=int)

    # convolve cond (as int) with kernel using 'same' mode
    expanded = np.convolve(cond.astype(int), kernel, mode='same') > 0

    return expanded

def trim_spectra(exp, xmin, xmax):
    """ Trim all spectra to the given x-range, inclusive. """
    x = exp.attrs["ir-x"]

    mask = (x >= xmin) & (x <= xmax)
    exp.attrs["ir-x"] = x[mask]
    exp["ir"] = exp["ir"].map(lambda arr: arr[mask])

    if "ir-stddev" in exp.columns:
        exp["ir-stddev"] = exp["ir-stddev"].map(lambda arr: arr[mask])


def prepare_experiment(exp, xmin, xmax):
    trim_spectra(exp, xmin, xmax)

    solv_spec, solv_stddev = get_solvent_spectrum(exp)

    ir_mask = get_noise_mask(solv_stddev)

    exp["ir-raw"] = exp["ir"].map(lambda arr: arr)
    exp["ir"] = exp["ir"].map(lambda arr: np.where(ir_mask, np.nan, arr))

    exp.attrs["ir-mask"] = ir_mask

def prepare_external_calibration(exp, chemical, ir_range, nan_mask=None, max_conc=None, nan_value=np.nan):
    """
    Truncate an external calibration experiment to the relevant concentration range and apply the nan mask.
    """
    if max_conc is not None:
        exp = exp[exp[f"{chemical} volume fraction"] <= max_conc]

    exp = exp.copy()

    trim_spectra(exp, *ir_range)

    if nan_mask is not None:
        exp["ir"] = exp["ir"].map(
            lambda a: np.where(nan_mask, nan_value, a)
        )

    return exp

def get_new_chemical(exp, knowns):
    candidate = None
    highest_vol = 0

    for chem in exp.attrs["chemicals"]:
        if not chem in knowns:
            vol = exp[f"{chem} volume"].iloc[-1]
            if vol > highest_vol:
                highest_vol = vol
                candidate = chem

    return candidate

def get_component_scores(data, *components):
    """
    Return the scalar products of all components with the data.
    """
    new_ys = []

    for comp in components:
        new_ys.append(np.array([np.nansum(q * comp, axis=0) for q in data]))

    return np.array(new_ys)

def orthogonalize_sets(targets, others):
    """
    Given the arrays in the list "targets", orthogonalize out all the variance that could be explained by the arrays in list "others".

    Further, return a list of floats giving the scalar product of the normalized, unorthogonalized targets to their orthogonalized variants prior to their renormalization.
    """
    if len(others) == 0:
        results = []
        for orig in targets:
            normalized = orig / np.linalg.norm(orig)

            results.append(normalized)

        return results, np.ones((len(targets), ))

    nonort_vecs = []
    for quant in others:
        vec = np.nan_to_num(quant)
        vec /= np.linalg.norm(vec)
        nonort_vecs.append(vec)

    A = np.column_stack(nonort_vecs)

    # I could probably use a faster algorithm, but whatever.
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    orthonormal_basis = U[:,:].T

    results = []
    overlaps = []

    for orig in targets:
        raw_spectrum = np.nan_to_num(orig)
        raw_spectrum /= np.linalg.norm(raw_spectrum)

        spectrum = raw_spectrum.copy()

        # Orthogonalize out all others
        for vec in orthonormal_basis:
            spectrum -= vec * np.dot(vec, spectrum)

        overlaps.append(np.dot(raw_spectrum, spectrum))

        spectrum /= np.linalg.norm(spectrum)

        results.append(spectrum)

    return results, overlaps

def get_equ_factors(chkpt, starting_mats):
    """
    Get the factors to convert the chemicals' volume fractions to equivalents. This assumes each chemical is dosed only once, because the final volume of each chemical is taken as the reference.
    """
    exp = chkpt.experiment

    used_vol = chkpt.experiment["Used volume"].to_numpy()

    factors = {}

    for i_chem, (chem, prediction) in enumerate(chkpt.pred_conc.items()):
        if chem in starting_mats:
            vol = 0
            for rct in starting_mats[chem]:
                vol += exp[f"{rct} volume"].to_numpy()[-1]
        else:
            vol = exp[f"{chem} volume"].to_numpy()[-1]

        factors[chem] = np.divide(
            used_vol, vol,
            out=np.zeros_like(used_vol, dtype=float),
            where=vol != 0,
        )

    return factors


