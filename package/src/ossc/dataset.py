
from ossc.common import get_component_scores, get_solvent, get_new_chemical, prepare_external_calibration, orthogonalize_sets

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

class Dataset:
    """ An experiment split into its different stages. """

    def __init__(self, exp, ir_range, products=None, n_internal=1000, catalyst=None, external_calibs=None):
        """
        n_internal: The number of chemicals in the initial dosage to be calibrated on this experiment rather than using external calibrations. This includes the solvent.
        products: List of chemicals. Places "Product1" etc if there aren't enough entries.
        """
        self.experiment = exp
        self.ir_range = ir_range
        if products is None:
            products = []
        self.products = products.copy()
        self.noise_mask = exp.attrs["ir-mask"]
        self.external_calibs = external_calibs

        solvent = get_solvent(exp)
        self.solvent = solvent

        self.digital = exp.attrs.get("simulated", False)

        products = products.copy()

        # Group the full dataset into the different stages
        snapshot_sets = []

        buffer = {
            "kind": None,
            "chemicals": [],
            "stages": [],
            "ground_truth_trends": [],
        }

        current_kind = None

        known_chemicals = []
        ignored_chemicals = [] # E.g. catalysts we cannot see

        i_internal = 0
        # Once a reaction has started, all later dosages use an external calibration
        reaction_seen = False

        def flush_subset(next_kind):
            nonlocal buffer

            if buffer["kind"] == SubSet.EXT_DOSAGE:
                if self.external_calibs is None:
                    raise Exception(f"Trying to use an external calibration, but no library is available for solvent {solvent}.")

                # Use the external dosage's data
                if len(buffer["chemicals"]) > 1:
                    raise Exception("Only one chemical is allowed per external dosage.")

                chem = buffer["chemicals"][0]

                if chem not in self.external_calibs:
                    raise Exception(f"Trying to use an external calibration, but the library for solvent {solvent} does not seem to contain the analyte {chem}.")

                ext_exp = prepare_external_calibration(external_calibs[chem], chem, 
                    self.ir_range, nan_mask=self.noise_mask,
                    max_conc=max(exp[f"{chem} volume fraction"].values)
                )
                snapshot_sets.append(SubSet(ext_exp, **buffer))
            else:
                snapshot_sets.append(SubSet(exp[exp["stage"].isin(buffer["stages"])], **buffer))

            buffer = {
                "kind": next_kind,
                "chemicals": [],
                "stages": [],
                "ground_truth_trends": [],
            }

        for raw_stage_name in exp["stage"].unique():
            stage = exp[exp["stage"] == raw_stage_name]

            stage_name = raw_stage_name.lower()

            # Determine what kind of stage this is
            if "empty" in stage_name:
                continue

            # This may be None; whether it should be depends on the kind of experiment
            new_chem = get_new_chemical(stage, known_chemicals)

            if "solvent" in stage_name:
                if current_kind is not None:
                    raise ValueError("Solvent stage isn't first stage for experiment " + exp.name)

                buffer["kind"] = SubSet.SOLVENT
                current_kind = SubSet.SOLVENT

            elif "reaction" in stage_name:
                current_kind = SubSet.REACTION
                reaction_seen = True

                if new_chem is not None:
                    # If this is a pseudoreaction, this is the chemical that is the "product"
                    if new_chem == catalyst:
                        # Just more catalyst
                        ignored_chemicals.append(new_chem)
                        known_chemicals.append(new_chem)
                        new_chem = None
                    else:
                        raise Exception(f"In experiment {exp.name}, stage {stage_name} contains the previously unseen {new_chem}, but should be a reaction.")
                else:
                    if products is not None and len(products) > 0:
                        new_chem = products.pop(0)
                    else:
                        # Provide a generic product
                        n_prod = len([sset for sset in snapshot_sets if sset.kind == SubSet.REACTION])
                        new_chem = f"Product {n_prod + 1}"

            elif "monitor" in stage_name:
                current_kind = SubSet.MONITORING

            elif "dosage" in stage_name:
                # In validation experiments, we might dose chemicals more than once
                if new_chem is None:
                    current_kind = SubSet.MONITORING
                elif new_chem == catalyst:
                    # Just the catalyst
                    ignored_chemicals.append(new_chem)
                    known_chemicals.append(new_chem)
                    new_chem = None
                    # After the catalyst dosage, we necessarily move to the actual reaction
                    current_kind = SubSet.REACTION
                else:
                    if i_internal < n_internal and not (n_internal is not np.inf and reaction_seen):
                        i_internal += 1
                        current_kind = SubSet.DOSAGE
                    else:
                        current_kind = SubSet.EXT_DOSAGE

            else:
                raise ValueError(f"Unknown stage kind {stage_name} in experiment {exp.name}")

            if current_kind != buffer["kind"] or current_kind == SubSet.EXT_DOSAGE:
                flush_subset(current_kind)

            if new_chem:
                known_chemicals.append(new_chem)
                buffer["chemicals"].append(new_chem)

                if self.digital:
                    buffer["ground_truth_trends"].append(f"True {new_chem} volume fraction")
                else:
                    buffer["ground_truth_trends"].append(f"{new_chem} volume fraction")

            buffer["stages"].append(raw_stage_name)

        flush_subset(current_kind)

        self.subsets = snapshot_sets

        # Bundle up some subset-dependent info
        self.conc_trends = {}
        self.reaction_onset = None

        prereaction_stages = []

        for subset in self.subsets:
            for chem, handle in zip(subset.chemicals, subset.ground_truth_trends):
                self.conc_trends[chem] = handle

            if self.reaction_onset is None and subset.kind in SubSet.REACTION:
                self.reaction_onset = self.experiment.index[self.experiment["stage"] == subset.stages[0]].tolist()[0]

            prereaction_stages.extend(subset.stages)

    def get_true_conc(self, chem):
        """
        Return the ground truth; in case of a true experiment, trim it to before the reaction starts, neglecting spontaneous reaction.
        """
        exp = self.experiment

        try:
            trend = exp[self.conc_trends[chem]].to_numpy().copy()
        except:
            # If this is a product, just put zeroes.
            trend = np.zeros(len(exp))

        if self.reaction_onset is not None and not self.digital and not chem == self.solvent:
            # We only have a ground truth up to the catalyst addition
            trend[self.reaction_onset:] = np.nan

        return trend

    def overview(self):
        ret = ""
        for subset in self.subsets:
            ret += str(subset.kind) + "\n" + str(subset.chemicals) + "\n\n"

        return ret

class SubSet:
    """
    The original experiment orders the data by stages. We reorganize them by how they are going to be processed.
    """
    SOLVENT = "solvent"
    DOSAGE = "dosage"
    EXT_DOSAGE = "external dosage"
    REACTION = "reaction"
    MONITORING = "monitoring"

    def __init__(self, exp, kind, stages, chemicals, ground_truth_trends):
        """
        exp is already trimmed to the specific stages.
        """
        if isinstance(chemicals, str):
            raise ValueError("A single chemical was passed; if intentional, wrap it in a list.")

        self.exp = exp
        self.kind = kind
        self.stages = stages
        self.chemicals = chemicals
        self.ground_truth_trends = ground_truth_trends
        self.byproducts = []

        n_components = len(stages)

        # Cache the PCA where necessary for better performance
        if self.kind in (SubSet.DOSAGE, SubSet.EXT_DOSAGE):
            # This subtracts the last record before this dosage from all spectra. The results are the same as for a PCA, except the spectra use a different shift. Obviously this ignores solvent dilution, which is accounted for by orthogonalizing out the solvent signal later.
            comp_pipe = Pipeline([
                ("shift", FunctionTransformer(lambda X: X - np.nan_to_num(exp["ir"].iloc[0]), validate=False)),
                ("ortho", TruncatedSVD(n_components=n_components)),
                #("ortho", PCA(n_components=n_components)),
                #("select", ColumnTransformer([
                #        ("keep", "passthrough", list(range(len(stages))))
                #    ], remainder="drop")),
                ("reg", LinearRegression()),
            ])

            self.cached_model = comp_pipe

            ir_data = np.stack(exp["ir"].to_numpy())
            np.nan_to_num(ir_data, copy=False)

            conc_data = np.stack([exp[f"{chem} volume fraction"].to_numpy() for chem in chemicals]).T

            self.cached_model.fit(ir_data, conc_data)

        elif self.kind in (SubSet.REACTION):
            # This model isn't meant for prediction, we only need the first principal component of the stage
            pca = PCA(n_components=1)
            feature_data = np.stack(exp["ir"].to_numpy())
            np.nan_to_num(feature_data, copy=False)

            pca.fit(feature_data)

            # These scores are a qualitative trend for the reaction progress,
            # we need to fit it later to make it quantitative.
            self.stage_scores = pca.transform(feature_data)[:, 0]
            self.cached_model = pca
        else:
            self.cached_model = None

    def add_byproduct(self, chem):
        """ If a reaction produces a known byproduct, this adds an extra MCD to remove it. """
        if self.kind is not SubSet.REACTION:
            raise Exception("Trying to add a byproduct to a non-reaction SubSet.")
            
        self.byproducts.append(chem)

class EvaluationCheckpoint:
    """
    The collection of all currently extracted information after every SubSet.
    n_subsets: The number of subsets in the dataset already evaluated at this checkpoint.
    calibrators: Calibrators for individual subsets
    pred_conc: Precalculated concentration predictions
    metadata: Any data one might want to pass forward for manual evaluation
    """
    def __init__(self, dataset, n_subsets, calibrators, pred_conc, metadata):
        self.dataset = dataset

        all_stages = []
        for subset in self.dataset.subsets[:n_subsets]:
            all_stages.extend(subset.stages)

        self.experiment = dataset.experiment[dataset.experiment["stage"].isin(all_stages)]
        self.calibrators = calibrators
        self.pred_conc = pred_conc
        self.n_subsets = n_subsets

        if metadata is None:
            metadata = {}

        self.metadata = metadata

class SubSetCalibrator:
    """
    Whenever new chemicals get added to the set of extracted spectra, we need a new orthogonalization from scratch and thus a recalibration. This class wraps all the info needed for that, generating the new predictions on all considered records.
    """
    def __init__(self, experiment, chemicals, extracted, concentrations, extra_components=None, linear_model=True):
        """
        experiment: Part of the experiment that contains the calibration data for these chemicals
        chemicals: Which chemicals are calibrated for
        extracted: These chemicals' spectra prior to orthogonalization
        concentrations: The chemicals' concentrations for each snapshot, as used for the calibration. Given as arrays rather than column handles because that wouldn't work for products.
        extra_components: They are included in the orthogonalization, but not in concentration calculations.
        linear_model: Switch between a linear and a proportional model y = a*x [+b] for predictions. If there is more than one new chemical in the SubSet, a multilinear model is always used.
        """
        self.experiment = experiment
        self.ir_data = np.stack(experiment["ir"].tolist())
        self.chemicals = chemicals
        self.extracted = extracted
        self.concentrations = concentrations
        self.extra_components = extra_components
        self.linear_model = linear_model

        if not linear_model and len(chemicals) > 1:
            raise ValueError("For SubSets with more than one chemical, only linear models can be used.")

        self.components = extra_components.copy() if extra_components is not None else []
        self.components += extracted

    def predict(self, experiment, other_spectra):
        """
        experiment: The data on which the model gets applied
        other_components: Other spectra against which this calibrator's spectra get orthogonalized
        """
        # Orthogonalize out the other spectra
        explainable_components, overlaps = orthogonalize_sets(self.extracted, other_spectra)

        # Calibrate on the spectra within this calibrator
        calib_scores = get_component_scores(self.ir_data, *explainable_components)
        pred_scores = get_component_scores(experiment["ir"].to_numpy(), *explainable_components).T

        if len(self.chemicals) == 1 and not self.linear_model:
            # We use a proportional model instead. This reflects how an explainable components is orthogonal to all other chemicals.
            model = LinearRegression(fit_intercept=False)
        else:
            model = LinearRegression()

        model.fit(calib_scores.T, np.array(self.concentrations).T)
        pred_conc = model.predict(pred_scores).T

        #print(f"Prediction for {self.chemicals}, component overlap: ", overlaps)

        return {chem: conc for chem, conc in zip(self.chemicals, pred_conc)}
