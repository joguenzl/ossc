import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import defaultdict
    from itertools import chain
    import os
    import random
    import traceback

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy

    import ossc
    import ossc.simulator as sim

    return defaultdict, mo, np, os, ossc, sim


@app.cell(hide_code=True)
def _(mo, os):
    # Prepare some UI elements, but don't render them yet
    spectra_kind_selector = mo.ui.dropdown(
        options=["Simplified", "Authentic"],
        value="Authentic",
        label="Select kind of spectra: "
    )

    nonprop_selector = mo.ui.dropdown(
        options=["None", "Solvent"],
        value="None",
        label="Select nonlinear absorbance-concentration: "
    )

    _keys = ["Single-step R1", "Single-step R3", "Mixture"]
    simulation_selector = mo.ui.dropdown(
        options=_keys, value=_keys[0], label="Select experiment kind:"
    )

    n_mixture = mo.ui.slider(
        start=1, stop=5, step=1, value=3, show_value=True, label="Mixture components:"
    )

    default_params = {
        "kf": 0.03,
        "yld": 80, # yield is a keyword
        "kf_nocat": 0.00057,
        "kfr3_nocat": 0.03,
    }

    r1_params = (
        mo.md(
            """
            Reaction parameters:

            - $k_f$ [1 / (%Vol $\\cdot$ s)] {kf}
            - $k_f$ [1 / (%Vol $\\cdot$ s)] (no catalyst) {kf_nocat}
            - Yield [%] {yld}
            """
        )
        .batch(
            kf=mo.ui.number(value=default_params["kf"], start=0.01, stop=0.1, step=0.001),
            kf_nocat=mo.ui.number(value=default_params["kf_nocat"], start=0, stop=0.005, step=0.0005),
            yld=mo.ui.number(value=default_params["yld"], start=10, stop=99),
        )
        .form(submit_button_label="Submit")
    )

    r3_params = (
        mo.md(
            """
            Reaction parameters:

            - $k_f$ [1 / (%Vol $\\cdot$ s)] (no catalyst) {kfr3_nocat}
            - Yield [%] {yld}
            """
        )
        .batch(
            kfr3_nocat=mo.ui.number(value=default_params["kfr3_nocat"], start=0, stop=0.1, step=0.001),
            yld=mo.ui.number(value=default_params["yld"], start=10, stop=99),
        )
        .form(submit_button_label="Submit")
    )

    def _campaign_sort(name):
        if "thio" in name:
            if "formation" in name:
                return 0
            return 1
        if "benz" in name:
            return 2

        return 10

    _options = sorted(
        [filename for filename in os.listdir(".") if filename.endswith(".h5")],
        key=_campaign_sort,
    )

    campaign_selector = mo.ui.dropdown(
        options=_options, value=_options[0], label="Select campaign: "
    )
    return (
        campaign_selector,
        default_params,
        n_mixture,
        nonprop_selector,
        r1_params,
        r3_params,
        simulation_selector,
        spectra_kind_selector,
    )


@app.cell
def _():
    ir_range = (800, 1800)
    return (ir_range,)


@app.cell(hide_code=True)
def _(defaultdict, ossc):
    # Load external calibration datasets
    def _cell():
        out = defaultdict(lambda: {}) # {Solvent -> {Analyte -> Dataset}}

        # Carbonyl compounds for mixtures
        carb_camp = ossc.read_campaign("references/carbonyls-external.h5")

        for _, exp in carb_camp.items():
            solv = ossc.get_solvent(exp)
            carb = ossc.get_new_chemical(exp, [solv])
            out[solv][carb] = exp

        # Benzaldehyde
        benz_camp = ossc.read_campaign("references/benzaldehyde-external.h5")
        benz = "Benzaldehyde"

        for _, exp in benz_camp.items():
            solv = ossc.get_solvent(exp)
            out[solv][benz] = exp

        # Aniline
        anil_camp = ossc.read_campaign("references/aniline-external.h5")
        anil = "Aniline"

        for _, exp in anil_camp.items():
            solv = ossc.get_solvent(exp)
            out[solv][anil] = exp

        return out

    external_calibrations = _cell()
    return (external_calibrations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data source
    """)
    return


@app.cell
def _(mo):
    # Menu to choose between experimental and simulated data
    _keys = ["Experiment", "Simulation"]
    source_selector = mo.ui.dropdown(
        options=_keys, value=_keys[0], label="Select data source: "
    )
    source_selector
    return (source_selector,)


@app.cell
def _(
    campaign_selector,
    mo,
    nonprop_selector,
    simulation_selector,
    source_selector,
    spectra_kind_selector,
):
    # Provide the UI to set the parameters for the simulator
    if source_selector.value == "Simulation":
        mo.output.append(spectra_kind_selector)
        mo.output.append(simulation_selector)
        mo.output.append(nonprop_selector)
    elif source_selector.value == "Experiment":
        mo.output.append(campaign_selector)
    return


@app.cell
def _(campaign_selector, ir_range, ossc, source_selector):
    if source_selector.value == "Experiment":
        campaign = ossc.read_campaign(campaign_selector.value)
        for _exp in campaign.values():
            ossc.prepare_experiment(_exp, *ir_range)
    else:
        campaign = None
    return (campaign,)


@app.cell
def _(
    campaign,
    mo,
    n_mixture,
    r1_params,
    r3_params,
    simulation_selector,
    source_selector,
):
    experiment_selector = None # Needs to be generated on the spot
    if source_selector.value == "Simulation":
        if simulation_selector.value == "Mixture":
            mo.output.append(n_mixture)
        elif simulation_selector.value == "Single-step R1":
            # Parameters for the first reaction as a combined form
            mo.output.append(r1_params)
        elif simulation_selector.value == "Single-step R3":
            # Parameters for the first reaction as a combined form
            mo.output.append(r3_params)

    elif source_selector.value == "Experiment":
        # Select the experiment to look at
        _options = [handle for handle in campaign]

        if len(_options) > 0:
            experiment_selector = mo.ui.dropdown(
                options=_options, value=_options[-1], label="Select experiment: "
            )
            if len(_options) > 1:
                mo.output.append(experiment_selector)
    return (experiment_selector,)


@app.cell
def _(experiment, mo, n_mixture, simulation_selector, source_selector):
    # Needs to be generated dynamically
    if source_selector.value == "Simulation" and simulation_selector.value == "Mixture":
        _stop = n_mixture.value
        _show = True
    else:
        _stop = len(experiment.attrs["chemicals"]) - 1
        _show = "Reaction" not in experiment["stage"].values

    n_external_mixture = mo.ui.slider(
        start=0, stop=_stop, value=0, label="Components using an external dosage: ", show_value=True
    )

    if _show:
        mo.output.append(n_external_mixture)
    return (n_external_mixture,)


@app.cell(hide_code=True)
def _(
    default_params,
    get_nonprop,
    n_mixture,
    ossc,
    r1_params,
    r3_params,
    sim,
    simulation_selector,
    spectra_kind_selector,
):
    def simulate_experiment():
        chems = list(sim.spectra[sim.SpectraKind.AUTHENTIC].keys())
        iso = "Isopropanol"
        anil = "Aniline"
        benz = "Benzaldehyde"
        thio = "Thiophene-2-carboxaldehyde"
        water = "Water"
        ti = "Thioimine"
        bi = "Benzimine"

        if spectra_kind_selector.value == "Authentic":
            selspectra = sim.spectra[sim.SpectraKind.AUTHENTIC]
        elif spectra_kind_selector.value == "Simplified":
            selspectra = sim.spectra[sim.SpectraKind.NOOVERLAP]

        if simulation_selector.value == "Mixture":
            return ossc.run_mixture(selspectra, iso, [benz, thio, ti, bi, water][:n_mixture.value], nonprop=get_nonprop())

        if simulation_selector.value == "Single-step R1":
            values = default_params if r1_params.value is None else r1_params.value
            K = (1 / (1 - values["yld"] / 100) - 1)**2
            return ossc.run_reaction(selspectra, iso, [anil, thio, ti, water], values["kf"], K, has_cat=True, kf_nocat=values["kf_nocat"], monitoring_duration=10, nonprop=get_nonprop())

        elif simulation_selector.value == "Single-step R3":
            values = default_params if r3_params.value is None else r3_params.value
            K = (1 / (1 - values["yld"] / 100) - 1)**2
            return ossc.run_reaction(selspectra, iso, [anil, benz, bi, water], values["kfr3_nocat"], K, has_cat=False, nonprop=get_nonprop())

        raise Exception(f"Simulation kind not implemented: {simulation_selector.value}")

    return (simulate_experiment,)


@app.cell
def _(
    campaign,
    experiment_selector,
    ir_range,
    ossc,
    simulate_experiment,
    source_selector,
):
    if source_selector.value == "Simulation":
        experiment = simulate_experiment()
        ossc.prepare_experiment(experiment, *ir_range)
    else:
        if experiment_selector is None:
            experiment = list(campaign.values())[0]
        else:
            experiment = campaign[experiment_selector.value]
    return (experiment,)


@app.cell
def _():
    # Evaluation
    return


@app.cell
def _(
    experiment,
    external_calibrations,
    get_nonprop,
    ir_range,
    n_external_mixture,
    n_mixture,
    ossc,
    sim,
    simulation_selector,
    source_selector,
    spectra_kind_selector,
):
    def _cell():
        """ Build the dataset, i.e. finalize all the labels and metadata """
        n_internal = 1000

        if "Reaction" not in experiment["stage"].values:
            if source_selector.value == "Simulation" and simulation_selector.value == "Mixture":
                n_chems = n_mixture.value
            else:
                n_chems = len(experiment.attrs["chemicals"]) - 1

            n_internal = n_chems - n_external_mixture.value

            products = []
        else:
            # Which one is it?
            labels = " ".join(experiment["stage"].unique()).lower()
            benz_pos = labels.find("benz")
            thio_pos = labels.find("thio")

            if benz_pos != -1 and (benz_pos < thio_pos or thio_pos < 0):
                n_internal = 1
                products = [
                    "Benzimine",
                    "Thioimine",
                ]
            else:
                products = [
                    "Thioimine",
                    "Benzimine",
                ]

        solv = ossc.get_solvent(experiment)

        # Which set of external calibrations do we need?
        if source_selector.value == "Simulation":
            if spectra_kind_selector.value == "Authentic":
                selspectra = sim.spectra[sim.SpectraKind.AUTHENTIC]
            elif spectra_kind_selector.value == "Simplified":
                selspectra = sim.spectra[sim.SpectraKind.NOOVERLAP]

            ext_calibs = {}

            # Generate the external calibrations on the spot, including nonlinearities etc
            for chem in experiment.attrs["chemicals"]:
                max_volfrac = max(experiment[f"{chem} volume fraction"])

                V_solv = 25
                if chem == solv:
                    continue
                else:
                    V_chem = V_solv * max_volfrac / (1 - max_volfrac)

                # If it is fully linear, there is no point in using more
                n_dosage_steps = 1

                if get_nonprop() == sim.NonProp.ALL:
                    n_dosage_steps = 10

                exp = sim.DigiExperiment(selspectra, nonprop=get_nonprop())

                exp.start_stage("Solvent")
                exp.add_chemical(solv, V_solv, stirring=0)

                exp.start_stage(f"{chem} dosage")
                for i in range(n_dosage_steps):
                    exp.add_chemical(chem, V_chem / n_dosage_steps)

                ext_calibs[chem] = exp.get_experiment()
        else:
            if solv in external_calibrations:
                ext_calibs = external_calibrations[solv]
            else:
                ext_calibs = None

        dataset = ossc.Dataset(experiment, ir_range, products, 
                       n_internal=n_internal,
                       catalyst="Acetic acid",
                       external_calibs=ext_calibs
        )

        # Add thiophene aldehyde as a known byproduct to the second reaction so the imine spectrum gets cleaned
        first_rct_seen = False
        for subset in dataset.subsets:
            if subset.kind == ossc.SubSet.REACTION:
                if not first_rct_seen:
                    first_rct_seen = True
                else:
                    subset.add_byproduct("Thiophene-2-carboxaldehyde")
                    break

        return dataset

    dataset = _cell()
    return (dataset,)


@app.cell
def _():
    style = {
        "Isopropanol": ("gray", "--", "Isopropanol"),
        "Aniline": ("#f5b214", "--", "Aniline"),
        "Benzaldehyde": ("#1092ee", "-.", "Benzald."),
        "Thiophene-2-carboxaldehyde": ("#ec6d1f", "--", "Thioph. ald."),
        "Benzimine": ("#0aad48", "-.", "Imine (Benzald.)"),
        "Thioimine": ("#c8054b", "--", "Imine (Thioph.)"),
        "Ethyl acetate": ("#f5b214", "--", "Ethyl acetate"),
        "N-Methyl-2-pyrrolidone": ("#c8054b", "--", "NMP"),
        "N,N-Dimethylacetamide": ("#0aad48", "--", "DMAc"),
    }
    return (style,)


@app.cell
def _(checkpoint, ossc):
    if "zeta" in checkpoint.metadata:
        ossc.render_zeta_surface(checkpoint.metadata["zeta"])
    return


@app.cell(hide_code=True)
def _(mo, ossc, style):
    def evaluate_checkpoint(checkpoint):
        """
        Render a human-readable summary of the experiment up to the checkpoint
        """
        dataset = checkpoint.dataset

        # Heading
        headings = {
            ossc.SubSet.SOLVENT: "Solvent",
            ossc.SubSet.DOSAGE: "Dosage",
            ossc.SubSet.EXT_DOSAGE: "External calibration",
            ossc.SubSet.MONITORING: "Monitoring",
            ossc.SubSet.REACTION: "Reaction",
        }
        mo.output.append(mo.md("## " + headings[dataset.subsets[checkpoint.n_subsets - 1].kind]))

        exp = checkpoint.experiment

        # Render all concentrations in equivalents / %Vol for mixtures
        refs = {chem: dataset.get_true_conc(chem) for chem in checkpoint.pred_conc.keys()}

        if not "Reaction" in dataset.experiment["stage"].values:
            # Mixture experiment
            fig, ax = ossc.render_volconc(exp["Time since start"].to_numpy(), refs, checkpoint.pred_conc, style, exclude=[dataset.solvent])

            mo.output.append(fig)
        else:
            # Actual reaction
            scaling = {}
            if exp.attrs.get("simulated", False):
                scaling = {"Benzimine": 2, "Thioimine": 2}

            starting_mats = {
                "Benzimine": ("Aniline", "Benzaldehyde"),
                "Thioimine": ("Aniline", "Thiophene-2-carboxaldehyde"),
            }

            i_subset = checkpoint.n_subsets - 1
            last_subset = dataset.subsets[i_subset]

            equ_factors = ossc.get_equ_factors(checkpoint, starting_mats=starting_mats)
            conc = {chem: pred * equ_factors[chem] for chem, pred in checkpoint.pred_conc.items()}

            is_twostep = len(last_subset.byproducts) > 0

            if not is_twostep:
                fig, ax = ossc.render_equconc(checkpoint, style, ref_scaling=scaling, starting_mats=starting_mats, exclude=[dataset.solvent])
                mo.output.append(fig)


            # If this is the second reaction, use the stoichiometry relations for better accuracy
            if is_twostep:
                mo.output.append(mo.md("Two-step reaction detected: Inferring concentrations through stoichiometry"))

                anil_conc = conc["Aniline"]
                ti_conc = conc["Thioimine"]

                for subset in dataset.subsets:
                    if "Benzaldehyde" in subset.chemicals:
                        first_step_end = subset.exp.index[0]
                        break
                else:
                    raise Exception("This seems to be a two-step reaction, but the Benzaldehyde dosage cannot be found.")

                second_step_start = last_subset.exp.index[0]

                ta_conc = 1 - ti_conc
                ta_conc[:first_step_end] = conc["Thiophene-2-carboxaldehyde"][:first_step_end]
                conc["Thiophene-2-carboxaldehyde"] = ta_conc

                ba_conc = ti_conc + anil_conc
                ba_conc[:first_step_end] = 0
                ba_conc[:second_step_start - 1] = conc["Benzaldehyde"][:second_step_start - 1]
                conc["Benzaldehyde"] = ba_conc

                bi_conc = 1 - ba_conc
                bi_conc[:first_step_end] = 0
                bi_conc[:second_step_start - 1] = conc["Benzimine"][:second_step_start - 1]
                conc["Benzimine"] = bi_conc

                del conc[dataset.solvent]

                mo.output.append(ossc.render_conc(checkpoint, conc, style))

            solv = ossc.get_solvent(exp)
            names = []
            equs = []

            for chem, pred in conc.items():
                if chem == solv:
                    continue

                try:
                    equs.append(round(pred[-1], 2))
                    names.append(chem)
                except:
                    pass

            mo.output.append(mo.ui.table({"Chemical": names, "Final concentration [eq]": equs}, selection=None))

            if checkpoint.dataset.subsets[i_subset].kind == ossc.SubSet.REACTION:
                # Highlight product yields
                for chem, equ in zip(names, equs):
                    if chem in checkpoint.dataset.products:
                        mo.output.append(f"Yield for {chem}: {100*equ:0.1f}")

        # Render all MCD spectra together in a stacked plot
        fig, ax = ossc.render_stacked_spectra(exp.attrs["ir-x"], checkpoint.metadata["mcd-spectra"], style)
        mo.output.append(fig)

    return (evaluate_checkpoint,)


@app.cell
def _():
    from ossc.deconvolution import ZetaDecomp, zeta_space

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Output
    """)
    return


@app.cell
def _(dataset, ossc):
    checkpoint = None
    checkpoints = []

    for _i in range(15):
        _chkpt = ossc.eval_subset(dataset, _i, checkpoint, verbose=True)
        if _chkpt is checkpoint:
            break
        checkpoint = _chkpt
        checkpoints.append(checkpoint)
    return checkpoint, checkpoints


@app.cell
def _(checkpoints, dataset, mo, ossc):
    def _cell():
        for chkpt in checkpoints:
            subset = dataset.subsets[chkpt.n_subsets - 1]
            if subset.kind is not ossc.SubSet.REACTION:
                continue

            meta = chkpt.metadata.get("zeta2", chkpt.metadata.get("zeta"))
            fig, ax = ossc.render_zeta_surface(meta)
            mo.output.append(fig)
            mo.output.append(mo.md("Figure: Plot of the feature count as a function of the background fraction ζ over time. The dashed line shows the values obtained by applying MCD to the corresponding spectrum, the solid line is the first principal component of all the spectra in the stage fitted to them."))

    _cell()
    return


@app.cell
def _(checkpoint, evaluate_checkpoint):
    evaluate_checkpoint(checkpoint)
    return


@app.cell(hide_code=True)
def _(checkpoint, dataset, mo, np, ossc):
    if not np.any([subset.kind == ossc.SubSet.REACTION for subset in dataset.subsets]):
        headline = f"## Average errors on the concentrations"
        data = ossc.evaluate_mixture(checkpoint)

        table_data = {}

        table_data["Chemical"] = data["Chemical"]
        table_data["Correlation (Prediction <-> Exact)"] = [round(x, 4) for x in data["Correlation (Prediction <-> Exact)"]]
        table_data["Average error [%]"] = [f"{100*x:0.1f}" for x in data["Average error"]]

        mo.output.append(mo.md(headline))
        mo.output.append(mo.ui.table(selection=None, data=table_data))
    return


@app.cell
def _(checkpoint, experiment, mo, ossc):
    if "Monitoring" in experiment["stage"].values and "Reaction" in experiment["stage"].values:
        _mon_out = ossc.evaluate_monitoring(checkpoint)
        mo.output.append(mo.md(f"$k_f$ measured during the monitoring phase: {10**6*_mon_out["kf"]:0.0f} (Vol% s)" + "$^{-1}$"))
        mo.output.append(mo.md(f"Inferred reactant consumption at end of dosage: {100*_mon_out["epsilon"]:0.2f} %"))
    return


@app.cell(hide_code=True)
def _(nonprop_selector, sim):
    def get_nonprop():
        opt = nonprop_selector.value
        if opt == "None":
            return sim.NonProp.NONE

        if opt == "Solvent":
            return sim.NonProp.SOLVENT

        if opt == "All":
            return sim.NonProp.ALL

        raise Exception("Invalid value for nonproportional relations")

    return (get_nonprop,)


if __name__ == "__main__":
    app.run()
