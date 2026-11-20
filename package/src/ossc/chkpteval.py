
from ossc.dataset import SubSet
import ossc.simulator as sim

import numpy as np
import scipy

def evaluate_monitoring(checkpoint):
    """
    Determine the consumption of reactants during the dosage to make sure it is small.
    This should be called on the reaction checkpoint after the actual monitoring so that the product is accounted for in the quantification.
    """
    # Does the checkpoint contain such a monitoring phase, i.e. one that comes between a dosage and a reaction set?
    last_subset = None
    reactants = []

    for subset in checkpoint.dataset.subsets[:checkpoint.n_subsets]:
        if subset.kind == SubSet.MONITORING \
            and last_subset is not None \
            and last_subset.kind in [SubSet.DOSAGE, SubSet.EXT_DOSAGE]:
            # This is the one
            break

        last_subset = subset

        if subset.kind in [SubSet.DOSAGE, SubSet.EXT_DOSAGE]:
            reactants.extend(subset.chemicals)
    else:
        raise Exception("No pre-catalyst monitoring stage was found.")

    # Get the concentrations within the monitoring phase
    full_exp = checkpoint.dataset.experiment
    subset_length = len(subset.exp)
    first_index = subset.exp.index[0]
    monitoring = subset.exp
    elapsed = monitoring["Time since start"]
    concentrations = {chem: checkpoint.pred_conc[chem][first_index:first_index + subset_length] for chem in reactants}

    elapsed = last_subset.exp["Time since start"]
    possible_consumption_duration = elapsed.iloc[-1] - elapsed.iloc[0]
    elapsed = monitoring["Time since start"]
    monitoring_duration = elapsed.iloc[-1] - elapsed.iloc[0]

    # Average the concentrations to ignore the difference in molar volumes
    educt_conc = np.mean([conc for conc in concentrations.values()], axis=0)
    # Get dA/dt from a linear regression

    linear_reg = scipy.stats.linregress(elapsed, educt_conc)
    slope = linear_reg.slope
    reg_r = linear_reg.rvalue

    kf = -slope * (1 / np.mean(educt_conc)**2)

    # Since we now have kf, simulate an experiment with that value to gauge the spontaneous consumption
    sim_exp = sim.run_reaction(sim.spectra[sim.SpectraKind.NOOVERLAP], "Isopropanol", ["Aniline", "Thiophene-2-carboxaldehyde", "Thioimine", "Water"], 10*kf, 16, has_cat=True, kf_nocat=kf)
    anil_true = sim_exp[sim_exp["stage"] == "Thiophene-2-carboxaldehyde dosage"]["True Aniline volume fraction"].iloc[-1]
    anil_nominal = sim_exp[sim_exp["stage"] == "Thiophene-2-carboxaldehyde dosage"]["Aniline volume fraction"].iloc[-1]
    ref_consumption = 1 - anil_true / anil_nominal

    return {
        "kf": kf,
        "epsilon": ref_consumption
    }

def evaluate_mixture(chkpt):
    """
    For nonreactive mixtures, we can measure how well the reconstructions match the exactly known concentration profiles.
    """
    chem_labels = []
    correlations = []
    avg_error = []

    time = chkpt.experiment["Time since start"].to_numpy()

    for chem, prediction in chkpt.pred_conc.items():
        if chem == chkpt.dataset.solvent:
            continue

        truth = chkpt.dataset.get_true_conc(chem)[:len(time)]

        max_conc = np.nanmax(truth)

        chem_labels.append(chem)
        avg_error.append(np.sqrt(np.mean((prediction - truth)**2)))
        correlations.append(np.corrcoef(prediction, truth)[0, 1])

    columns = {
        "Chemical": chem_labels,
        "Correlation (Prediction <-> Exact)": correlations,
        "Average error": avg_error,
    }

    return columns
