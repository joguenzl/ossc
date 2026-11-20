
from ossc.common import get_solvent_spectrum
from ossc.dataset import SubSet, SubSetCalibrator, EvaluationCheckpoint
from ossc.deconvolution import mcd, zeta_space

from itertools import chain

from sklearn.linear_model import LinearRegression

import numpy as np

def process_subset(dataset, subset, prior_calibrators, metadata=None, verbose=False):
    """ Given a subset, extract all info necessary for its chemicals' calibrators. """
    extra_components = None
    linear_model = True
    solv_spec = get_solvent_spectrum(dataset.experiment)
    wavenumbers = dataset.experiment.attrs["ir-x"]
    solvent = dataset.solvent

    if metadata is None:
        metadata = {
            "mcd-spectra": {},
            "mcd-meta": {},
        }
    else:
        if not "mcd-spectra" in metadata:
            metadata["mcd-spectra"] = {}
        if not "mcd-meta" in metadata:
            metadata["mcd-meta"] = {}

    mcd_spectra = metadata["mcd-spectra"]
    mcd_meta = metadata["mcd-meta"]

    if subset.kind == SubSet.SOLVENT:
        # Just extract the solvent spectrum.
        extracted, _ = get_solvent_spectrum(subset.exp)
        extracted = [extracted]

        # All the dosages because the solvent is always precisely known. Could also be all snapshots, but then we might calibrate on data including not-yet-extracted chemicals.
        subexp = dataset.experiment[:dataset.reaction_onset]

        chemicals = subset.chemicals
        concentrations = [subexp[f"{chem} volume fraction"].to_numpy() for chem in chemicals]

        mcd_spectra[chemicals[0]] = extracted[0] # Is already at 100 %.

    elif subset.kind == SubSet.DOSAGE:
        # Get the educt spectra as a PCA / SVD of the subset.
        subexp = subset.exp

        decomp = subset.cached_model
        oss_svd = decomp.named_steps["ortho"]
        oss_regression = decomp.named_steps["reg"]

        oss_pcs = oss_svd.components_[:len(subset.stages)]
        extra_pcs = oss_svd.components_[len(subset.stages):]
        oss_coefs = oss_regression.coef_
        oss_intercept = oss_regression.intercept_

        # These still contain solvent, which will get orthogonalized out later
        extracted = oss_coefs @ oss_pcs
        extracted = [spec for spec in extracted] # From 2d-ndarray to list of ndarrays

        chemicals = subset.chemicals
        concentrations = [subexp[f"{chem} volume fraction"].to_numpy() for chem in chemicals]

        # Also do an MCD to get the reactant spectra.
        for stage, chem in zip(subset.stages, subset.chemicals):
            stage_exp = subexp[subexp["stage"] == stage]
            snap = stage_exp.iloc[-1]
            ref = dataset.experiment.iloc[stage_exp.index[0] - 1]

            _, unmixed, mcd_metadata = mcd(ref["ir"], snap["ir"], verbose=verbose)
            unmixed /= snap.loc[f"{chem} volume fraction"]
            mcd_spectra[chem] = unmixed
            mcd_meta[chem] = mcd_metadata

    elif subset.kind == SubSet.EXT_DOSAGE:
        # Get the analyte spectrum as a PCA / SVD of the subset.
        subexp = subset.exp
        chemicals = subset.chemicals
        concentrations = [subexp[f"{chem} volume fraction"].to_numpy() for chem in chemicals]
        linear_model = False # Only one chemical per stage in this case

        decomp = subset.cached_model
        oss_svd = decomp.named_steps["ortho"]

        extracted = oss_svd.components_[:len(subset.stages)]
        extracted = [spec for spec in extracted] # From 2d-ndarray to list of ndarrays

        solv_spec, _ = get_solvent_spectrum(subexp)

        for stage, chem in zip(subset.stages, subset.chemicals):
            _, unmixed, mcd_metadata = mcd(solv_spec, subexp["ir"].iloc[-1], verbose=verbose)
            unmixed /= subexp[f"{chem} volume fraction"].iloc[-1]
            mcd_spectra[chem] = unmixed
            mcd_meta[chem] = mcd_metadata

    elif subset.kind == SubSet.MONITORING:
        # No new chemicals
        return None, None

    elif subset.kind == SubSet.REACTION:
        # We use minimal complexity deconvolution to separate the last pre-reaction snapshot into reactants and solvent,
        # then get the product spectrum by doing MCD on the reaction snapshots.
        linear_model = False

        # Get the cumulative reactant spectrum and the scaled solvent background
        solv_spectrum, _ = get_solvent_spectrum(dataset.experiment)
        # There might be a monitoring stage between the dosage and this stage; we want the snapshot right at the end of the dosage
        own_subset_index = dataset.subsets.index(subset)
        prior_subset = dataset.subsets[own_subset_index - 1]

        if prior_subset.kind == SubSet.MONITORING:
            prior_subset = dataset.subsets[own_subset_index - 2]
            if prior_subset.kind != SubSet.DOSAGE:
                raise Exception(f"A {prior_subset.kind} was found before a monitoring stage; that case is not implemented.")
            pre_reaction = prior_subset.exp["ir"].iloc[-1]
        else:
            index = subset.exp.index[0]
            pre_reaction = dataset.experiment["ir"].iloc[index - 1]

        # Evaluate the principal component on the reference snapshot, which serves as its zero when smoothing
        pca = subset.cached_model
        principal_c = pca.components_
        reference_pc_score = pca.transform([np.nan_to_num(pre_reaction)])[0]

        stage_scores = subset.stage_scores
        # This is now proportional to a good zeta estimate
        component_zeta_all = stage_scores - reference_pc_score

        # Separate the reference into reactants and background -- since the solvent does not change anymore,
        # this background stays constant within this subset.
        solv_zeta, cumul_educts, solv_subtract_meta = mcd(solv_spectrum, pre_reaction, verbose=verbose)
        solv_background = solv_spectrum * solv_zeta

        # Plot zeta over time and how the extracted spectra change
        zeta_vals = []
        prod_estimates = []
        elapsed = []
        zeta_traces = []
        component_zeta = []
        availability_mask = []
        snapshot_metadata = []

        for target_ir, target_elapsed, component_zeta_loc in zip(subset.exp["ir"].to_numpy(), subset.exp["Time since start"].to_numpy(), component_zeta_all):
            zeta_val, prod_estimate_loc, zeta_meta = mcd(cumul_educts, target_ir - solv_background, verbose=verbose)
            zeta_curve = zeta_meta["n-over-zeta"]

            # The curve is None if no peaks were detected and topout was used.
            if zeta_val > 0.95 or zeta_curve is None:
                # Very little signal, so skip it
                availability_mask.append(False)
                continue

            availability_mask.append(True)
            elapsed.append(target_elapsed)
            zeta_vals.append(zeta_val)
            zeta_traces.append(zeta_curve)
            component_zeta.append(component_zeta_loc)
            snapshot_metadata.append(zeta_meta)

            # TODO is this too naive?
            prod_estimate_loc /= 1 - zeta_val
            prod_estimates.append(prod_estimate_loc)

        if len(elapsed) == 0:
            raise Exception("Nothing to decompose was found in the zeta plot.")

        zeta_surface = [zeta_space, elapsed, zeta_traces]

        subexp = subset.exp
        chemicals = subset.chemicals

        # We scale the component_zeta using the instantaneous zeta
        zeta_inst = np.array(zeta_vals)
        zeta_comp = np.array(component_zeta)

        # We do a proportional fit because the intercept is fixed to the reference
        scaling_model = LinearRegression(fit_intercept=False)
        scaling_model.fit(np.atleast_2d(1 - zeta_inst).T, zeta_comp)
        zeta_comp /= scaling_model.coef_[0]
        zeta = zeta_comp
        zeta_comp = 1 - zeta_comp

        products_insta = [wavenumbers, elapsed, prod_estimates]

        # Calculate the product spectra from the smoothed curve
        prod_estimates_smooth = []
        for target_ir, zeta_loc in zip([snap for snap, filter in zip(subset.exp["ir"], availability_mask) if filter], zeta_comp):
            estimate = (target_ir - solv_background - zeta_loc * cumul_educts) / (1 - zeta_loc)
            prod_estimates_smooth.append(estimate)

        products_smoothed = [wavenumbers, elapsed, prod_estimates_smooth]

        # We average over the latter half of extracted spectra
        half_point = len(prod_estimates_smooth) // 2
        extracted = np.mean([estimate for estimate in prod_estimates_smooth[half_point:]], axis=0)

        byprod_meta = {}

        for byproduct in subset.byproducts:
            # Get the byproduct's spectrum.
            if byproduct not in metadata["mcd-spectra"]:
                raise Exception(f"Trying to remove byproduct {byproduct}, but there is no pure component spectrum for it.")

            byspec = metadata["mcd-spectra"][byproduct]
            _, extracted, byp_meta = mcd(byspec, extracted, verbose=verbose)
            byprod_meta["byproduct-" + byproduct] = byp_meta

        zeta_metadata = {
            "solv-background": solv_background,
            "solv-zeta": solv_zeta,
            "educt-background": cumul_educts,
            "availability": availability_mask,
            "plot-instantaneous": zeta_inst,
            "plot": zeta_comp,
            "surface": zeta_surface,
            "product-insta": products_insta,
            "product-smooth": products_smoothed,
            "solvent-meta": solv_subtract_meta,
            "snapshot-meta": snapshot_metadata,
            **byprod_meta
        }

        # Rescale to account for dilution
        # Use the true conc, not the zeta value because that one has nonlinearities
        extracted /= 1 - subset.exp[f"{solvent} volume fraction"].iloc[-1]
        mcd_spectra[subset.chemicals[0]] = extracted
        mcd_meta[subset.chemicals[0]] = zeta_metadata

        # Wrap it for the checkpoint __init__
        extracted = [extracted]

        if "zeta" in metadata:
            # This is the second reaction step
            metadata["zeta2"] = zeta_metadata
        else:
            metadata["zeta"] = zeta_metadata

        # For generality: We calculate how much volume the known chemicals consume, then suppose that all remaining volume must be occupied by the product.
        measured_conc = subexp[f"{solvent} volume fraction"].to_numpy().copy()

        for calibrator in prior_calibrators:
            if solvent in calibrator.chemicals:
                # We know its exact concentration
                continue

            other_components = list(chain.from_iterable([calib.components for calib in prior_calibrators if calib is not calibrator]))
            other_components.extend(extracted)
            pred_conc = calibrator.predict(subexp, other_components)

            for conc in pred_conc.values():
                measured_conc += conc

        prod_conc = 1 - measured_conc
        concentrations = [prod_conc]

    else:
        #raise NotImplementedError("Not implemented: " + str(subset.kind))
        print("Not implemented: " + str(subset.kind))
        return None, None

    return SubSetCalibrator(subexp, chemicals, extracted, concentrations, extra_components=extra_components, linear_model=linear_model), metadata

def eval_subset(dataset, index, prior, verbose=False):
    """
    Evaluate a SubSet and render the main output.
    prior: EvaluationCheckpoint containing all prior calibrators
    index: The index of the subset
    """
    if len(dataset.subsets) <= index:
        return prior

    subset = dataset.subsets[index]

    all_calibrators = prior.calibrators.copy() if prior is not None else []

    calibrator, new_metadata = process_subset(dataset, subset, all_calibrators, prior.metadata.copy() if prior is not None else None, verbose=verbose)

    if calibrator is not None:
        all_calibrators += [calibrator]

    # Calculate all predicted concentrations so we can compare them later.
    all_stages = list(chain.from_iterable([subset.stages for subset in dataset.subsets[:index + 1]]))

    experiment = dataset.experiment[dataset.experiment["stage"].isin(all_stages)]

    pred_conc = {}

    for calibrator in all_calibrators:
        other_components = list(chain.from_iterable([calib.components for calib in all_calibrators if calib is not calibrator]))
        pred_conc.update(calibrator.predict(experiment, other_components))

    if prior:
        metadata = prior.metadata.copy()
        if new_metadata is not None:
            metadata.update(**new_metadata)
    else:
        metadata = new_metadata

    checkpoint = EvaluationCheckpoint(dataset, index + 1, all_calibrators, pred_conc, metadata)

    return checkpoint

def eval_dataset(dataset, verbose=False):
    """ A wrapper that iterates over all subsets, and only produces the final checkpoint. """
    calibrators = []

    metadata = {}

    for subset in dataset.subsets:
        calib, new_metadata = process_subset(dataset, subset, calibrators, metadata, verbose=verbose)

        if new_metadata is not None:
            metadata.update(new_metadata)

        if calib is not None:
            calibrators.append(calib)

    # Cache all concentrations
    pred_conc = {}

    for calibrator in calibrators:
        other_components = list(chain.from_iterable([calib.components for calib in calibrators if calib is not calibrator]))
        pred_conc.update(calibrator.predict(dataset.experiment, other_components))

    checkpoint = EvaluationCheckpoint(dataset, len(dataset.subsets), calibrators, pred_conc, metadata=metadata)

    return checkpoint
