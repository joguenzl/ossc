
"""
Reference implementation of the Orthogonal SubSet Components (OSSC)
algorithm to analyze in situ spectroscopic data of experiments performed
in an automatic batch reactor.
"""

from ossc.common import get_component_scores, orthogonalize_sets, get_equ_factors, get_solvent, get_solvent_spectrum, get_noise_mask, trim_spectra, prepare_experiment, get_new_chemical, prepare_external_calibration
from ossc.chkpteval import evaluate_monitoring, evaluate_mixture
from ossc.dataset import Dataset, SubSet, EvaluationCheckpoint
from ossc.subseteval import process_subset, eval_subset, eval_dataset
from ossc.io import read_campaign, write_campaign
from ossc.deconvolution import mcd, zeta_space
from ossc.simulator import SpectraKind, NonProp, run_reaction, run_mixture
from ossc.viz import render_3d, render_spectra, render_true_conc, visualize_mcd, render_equconc, render_volconc, render_stacked_spectra, render_zeta_surface, render_conc
