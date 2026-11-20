import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import defaultdict
    import copy
    from itertools import chain
    import os
    import random
    import traceback

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy

    from matplotlib.patches import FancyArrowPatch, Rectangle
    from matplotlib.lines import Line2D

    import ossc
    import ossc.simulator as sim
    import ossc.deconvolution as deconv

    if not mo.running_in_notebook():
        os.makedirs("./graphics", exist_ok=True)

    if mo.running_in_notebook():
        # The exponent -1 does not work with the article template's font Calibri on my machine in some font sizes, so replace it with ⁻¹.
        from functools import wraps
        from matplotlib.axes import Axes
    
        def replace_label(method):
            @wraps(method)
            def wrapper(self, label, *args, **kwargs):
                if isinstance(label, str):
                    label = label.replace("$^{-1}$", "⁻¹")
                return method(self, label, *args, **kwargs)
            return wrapper
    
        Axes.set_xlabel = replace_label(Axes.set_xlabel)
        Axes.set_ylabel = replace_label(Axes.set_ylabel)
    
        plt.rcParams.update({
            "font.family": "Calibri",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Calibri",
            "mathtext.it": "Calibri:italic",
            "mathtext.bf": "Calibri:bold",
            "mathtext.sf": "Calibri",
        })
    return (
        FancyArrowPatch,
        Line2D,
        Rectangle,
        copy,
        deconv,
        defaultdict,
        mo,
        np,
        ossc,
        plt,
        sim,
    )


@app.cell
def _(mo):
    if not mo.running_in_notebook():
        outformat = "pdf"
    else:
        outformat = "svg"
    return (outformat,)


@app.cell
def _():
    single_col = 6
    return (single_col,)


@app.cell
def _(outformat, plt):
    # ⁻¹
    if outformat == "pdf":
        plt.rcParams.update({
            "font.family": "Open Sans",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Open Sans",
            "mathtext.it": "Open Sans:italic",
            "mathtext.bf": "Open Sans:bold",
            "mathtext.bfit": "Open Sans:italic:bold",
            "mathtext.sf": "Open Sans",
            "mathtext.cal": "Open Sans:italic",
            "mathtext.fallback": "stix",
        })
    return


@app.cell
def _():
    _chems = ["Aniline", "Thiophene-2-carboxaldehyde", "Thioimine", "Benzaldehyde", "Benzimine"]
    colors = ["#f5b214", "#ec6d1f", "#c8054b", "#1092ee", "#0aad48"] # Rainbow
    _ls = ["--", "--", "--", "-.", "-."]
    _labels = ["Aniline", "Thioph. ald.", "Imine (Thioph.)", "Benzald.", "Imine (Benz.)"]

    rctstyle = {chem: (c, l, la) for chem, c, l, la in zip(_chems, colors, _ls, _labels)}

    _chems = ["Methanol", "Ethanol", "n-Propanol", "n-Butanol", "Isopropanol"]
    _ls = ["-", "-", "-", "-", "-"]
    _labels = ["MeOH", "EtOH", "nPrOH", "nBuOH", "iPrOH"]

    solvent_style = {chem: (c, l, la) for chem, c, l, la in zip(_chems, colors, _ls, _labels)}
    return colors, rctstyle, solvent_style


@app.cell
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
    # Illustrations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graphical abstract
    """)
    return


@app.cell
def _(np, ossc, sim):
    def _generate():
        spectra = sim.spectra[sim.SpectraKind.AUTHENTIC].copy()
        spectra["Isopropanol"] = np.zeros((501, ))

        return ossc.run_reaction(
            spectra, "Isopropanol", ["Aniline", "Thiophene-2-carboxaldehyde", "Thioimine", "Water", "Benzaldehyde", "Benzimine"], 0.02, 16, has_cat=True, kf_nocat=0.0005, kf2=0.0025, K2=0.5, kf3=0.1, K3=16, nonprop=sim.NonProp.NONE, monitoring_duration=10,
        )

    ga_experiment = _generate()
    return (ga_experiment,)


@app.cell
def _(
    FancyArrowPatch,
    ga_experiment,
    mo,
    np,
    ossc,
    outformat,
    plt,
    rctstyle,
    sim,
):
    def _render():
        # Three panels: A 3D plot of the spectra, a stacked plot of the pure spectra and the exact concentrations.
        exp = ga_experiment

        ir_data = np.array([arr for arr in exp["ir"].to_numpy()])
        time = exp["Time since start"].to_numpy() / 3600
        wvn = exp.attrs["ir-x"]

        # Render the first panel
        T, W = np.meshgrid(time, wvn, indexing="ij")

        # Build triangles for the rectangular grid.
        nt, nw = ir_data.shape
        x, y, z = T.ravel(), W.ravel(), ir_data.ravel()
        triangles = []
        for i in range(nt - 1):
            for j in range(nw - 1):
                a = i * nw + j
                b = a + 1
                c = (i + 1) * nw + j
                d = c + 1

                # Choose the cell diagonal: "ad" or "bc"
                triangles.extend([(a, b, d), (a, d, c)])
                #triangles.extend([(a, b, c), (b, d, c)])

        fig = plt.figure(figsize=(17, 4))
        outer = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.075)
        right = outer[0, 1].subgridspec(1, 2, wspace=0.35)
        ax = fig.add_subplot(outer[0, 0], projection="3d")
        ax2 = fig.add_subplot(right[0, 0])
        ax3 = fig.add_subplot(right[0, 1])

        ax.zaxis.set_rotate_label(False)

        #ax.plot_surface(T, W, ir_data, cmap="viridis")
        ax.plot_trisurf(
            x, y, z, cmap="viridis", triangles=np.asarray(triangles),
            antialiased=False
        )

        ax.set_xlabel("Time $t$ [h]")
        ax.set_ylabel("Wavenumber $ν$ [cm$^{-1}$]")
        ax.set_zlabel("Absorbance $A$", rotation=90)

        elev = 20
        azim = 60
        ax.view_init(elev=elev, azim=azim)
        ax.invert_xaxis()
        ax.set_title("In situ reaction monitoring\nwith unknown products")

        # Pure component spectra
        ax2.set_title("Pure component spectra")
        chems = ["Aniline", "Thiophene-2-carboxaldehyde", "Thioimine", "Benzaldehyde", "Benzimine"]

        labels = ["Amine", "Aldehyde 1", "Imine 1", "Aldehyde 2", "Imine 2"]    
        style = {chem: (rctstyle[chem][0], "-", la) for chem, la in zip(chems, labels)}

        spectra = {chem: sim.spectra[sim.SpectraKind.AUTHENTIC][chem] for chem in chems}

        ax2.tick_params(axis="y", which="both", left=False, labelleft=False)
        stack_shift = 1

        i_chem = 0
        for key, spec in spectra.items():
            color, _, label = style.get(key, ("black", "-", key))
            ax2.plot(wvn, i_chem*stack_shift + spec, color=color, label=label)

            i_chem += 1

        ax2.set_xlim((min(wvn), max(wvn)))

        ax2.legend(loc="upper center", ncols=2, prop={"size": 9})
        ax2.set_ylim(-0.02, 6.7)

        ax2.set_xlabel(r"Wavenumber $ν$ [cm$^{-1}$]")
        ax2.set_ylabel(r"Absorbance $A$")

        # Concentration profiles
        ax3.set_title("Concentrations over time")

        # Build a checkpoint with exact predictions
        def factor(chem):
            if "imine" in chem:
                return 2

            return 1

        pred_conc = {chem: factor(chem) * exp[f"True {chem} volume fraction"].to_numpy() for chem in chems}
        class TmpDs:
            def __init__(self, exp):
                self.experiment = exp
                self.subsets = []

            def get_true_conc(self, chem):
                return self.experiment[f"True {chem} volume fraction"].to_numpy()

        ds = TmpDs(exp)

        chkpt = ossc.EvaluationCheckpoint(ds, 0, None, pred_conc, {})
        chkpt.experiment = exp

        starting_mats = {
            "Benzimine": ["Benzaldehyde", "Aniline"],
            "Thioimine": ["Thiophene-2-carboxaldehyde", "Aniline"],
        }

        ossc.render_equconc(chkpt, style=style, inax=ax3, starting_mats=starting_mats, ref_scaling={"Thioimine": 2, "Benzimine": 2}, legend_loc="upper right")

        # Add a large arrow in each gap
        for left_ax, right_ax, delta1, delta2 in [(ax, ax2, -0.003, -0.005), (ax2, ax3, -0.0075, -0.0225)]:
            left_edge = left_ax.get_position().x1
            right_edge = right_ax.get_position().x0
            y_center = (left_ax.get_position().y0 + left_ax.get_position().y1) / 2

            arrow = FancyArrowPatch(
                (left_edge + 0.01 + delta1, y_center),
                (right_edge - 0.01 + delta2, y_center),
                transform=fig.transFigure,
                arrowstyle="simple",
                mutation_scale=45,  # increase for a taller, broader arrow
                color="black",
                clip_on=False,
            )
            fig.add_artist(arrow)

        if not mo.running_in_notebook():
            fig.savefig("graphics/graphical-abstract." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md("Figure: The presented algorithm takes the time series of spectra produced during a chemical reaction (left) and extracts the pure components' spectra (center). This includes extracting the spectra of unknown products. After orthogonalising each spectrum against all other chemicals' spectra, their concentrations at each point in time are calculated."))

    _render()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stages & Steps
    """)
    return


@app.cell
def _(sim):
    def _generate():
        V_solv = 50
        V_chem = 8
        n_steps = (10, 10)
        n_steps_second = 10
        dosage_delay = 180 # Time the pump takes to deliver a chemical

        kf = 0.01
        kf_nocat = 0.000057
        kf2 = 0.001
        kf3 = 0.02

        K = 16
        K2 = 1
        K3 = 9

        solvent = "Isopropanol"
        chemicals = ["Aniline", "Thiophene-2-carboxaldehyde", "Thioimine", "Water", "Benzaldehyde", "Benzimine"]
        cat = "Acetic acid"

        if len(chemicals) < 4:
            raise Exception("The implemented reactions require at least four chemicals!")

        reactions = [sim.SecondOrderRevReaction(kf, K, *chemicals[:4], cat=cat, kf_nocat=kf_nocat)]
        reactions.append(sim.SecondOrderRevReaction(kf2, K2, chemicals[2], chemicals[4], chemicals[1], chemicals[5]))
        reactions.append(sim.SecondOrderRevReaction(kf3, K3, chemicals[0], chemicals[4], chemicals[5], chemicals[3]))

        exp = sim.DigiExperiment(sim.spectra[sim.SpectraKind.AUTHENTIC], reactions=reactions, nonprop=sim.NonProp.NONE)
        exp.start_stage("Solvent")
        exp.add_chemical(solvent, V_solv, stirring=0)

        for chem, n_step in zip(chemicals[:2], n_steps):
            exp.start_stage(f"{chem} dosage")
            for i in range(n_step):
                exp.add_chemical(chem, V_chem / n_step, stirring=180, dosage_delay=dosage_delay)

        exp.start_stage("Monitoring")
        for i in range(120):
            exp.tick(30)
            exp.generate_snapshot()

        exp.start_stage("Acetic acid dosage")
        exp.add_chemical(cat, 1e-16, stirring=0)

        exp.start_stage("Reaction")

        rct_ticks = 100
        rct_tick = 63
        for i in range(rct_ticks):
            exp.tick(rct_tick)
            exp.generate_snapshot()

        # If there are enough chemicals, start the second step
        exp.start_stage(f"{chemicals[-2]} dosage")

        n_step = n_steps_second
        for i in range(n_step):
            exp.add_chemical(chemicals[-2], V_chem / n_step, stirring=180, dosage_delay=dosage_delay)

        exp.start_stage("Reaction2")

        for i in range(rct_ticks):
            exp.tick(rct_tick)
            exp.generate_snapshot()

        return exp.get_experiment()

    stageexp = _generate()
    return (stageexp,)


@app.cell(hide_code=True)
def _(Line2D, Rectangle, mo, np, outformat, plt, rctstyle, stageexp):
    def _cell():
        # What do I need? I basically want three horizontally aligned boxes with the concentrations, steps and stages
        x = stageexp["Time since start"].to_numpy()
        x = x / x[-1]
        # 141 for the catalyst
        phase_boundaries = [11, 21, 142, 242, 252]
        phase_boundaries = [x[i - 1] for i in phase_boundaries]
        margin = 0.05 / 20

        labels = [
            "Dosage 1", "Dosage 2", "Monitoring", "Reaction 1", "External\ncalibration", "Reaction 2",
        ]

        colors = [rctstyle[x][0] for x in ["Aniline", "Thiophene-2-carboxaldehyde", "Thioimine", "Benzaldehyde", "Benzimine"]]
        stage_colors = colors[:2] + ["gray"] + colors[2:]

        timeline_1 = [(start + margin, end - start - 2 * margin, label, color) for start, end, label, color in zip([0] + phase_boundaries, phase_boundaries + [1], labels, stage_colors)]

        timeline_2 = [
            (margin, phase_boundaries[3] - 2 * margin, "Reaction step 1", "white"),
            (phase_boundaries[3] + margin, 1 - phase_boundaries[3] - 2 * margin, "Reaction step 2", "white"),
        ]

        # Helper for drawing labelled rectangles
        def draw_timeline(ax, intervals, y=0.5, height=0.8):
            bbox = ax.get_window_extent()
            aspect = bbox.height / bbox.width
            for start, duration, label, color in intervals:
                rect = Rectangle(
                    (start, y - height / 2),
                    duration,
                    height,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.9,
                    #boxstyle="round,pad=0.3",
                    #mutation_aspect=1/aspect
                )
                ax.add_patch(rect)

                # Center the label inside the rectangle
                ax.text(
                    start + duration / 2,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                    clip_on=True,
                )

            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.grid(axis="x", linestyle=":", alpha=0.5)

        # Figure and axes
        factor = 2
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(170 / 25.4 * factor, 2 * factor),
            sharex=True,
            gridspec_kw={
                "height_ratios": [3, 1, 1],
                "hspace": 0.08,
            },
        )

        ax_function, ax_timeline_1, ax_timeline_2 = axes

        # First row: function
        used_vol = stageexp["Used volume"].to_numpy()
        thio_vol = stageexp["Thiophene-2-carboxaldehyde volume"].to_numpy()[-1]
        anil_vol = stageexp["Aniline volume"].to_numpy()[-1]
        benz_vol = stageexp["Benzaldehyde volume"].to_numpy()[-1]

        ax_function.plot(x, np.nan_to_num(stageexp["True Aniline volume fraction"].to_numpy() * used_vol / anil_vol), label="Aniline", color=colors[0])
        ax_function.plot(x, np.nan_to_num(stageexp["True Thiophene-2-carboxaldehyde volume fraction"].to_numpy() * used_vol / thio_vol), label="Thioph. ald.", color=colors[1])
        ax_function.plot(x, 2 * np.nan_to_num(stageexp["True Thioimine volume fraction"].to_numpy() * used_vol / (anil_vol + thio_vol)), label="Product 1", color=colors[2])
        ax_function.plot(x, np.nan_to_num(stageexp["True Benzaldehyde volume fraction"].to_numpy() * used_vol / (benz_vol)), label="Benzald.", color=colors[3])
        ax_function.plot(x, 2 * np.nan_to_num(stageexp["True Benzimine volume fraction"].to_numpy() * used_vol / (anil_vol + benz_vol)), label="Product 2", color=colors[4])

        # Separate stages
        for x_at in phase_boundaries:
            ax_function.axvline(x_at, color="black", ls="--")

        #ax_function.axhline(0, color="gray", linewidth=0.8)
        ax_function.set_ylabel("Concentration $c$ [equ.]")
        #ax_function.grid(True, linestyle=":", alpha=0.5)

        fig.canvas.draw()

        # Second and third rows: timelines
        draw_timeline(ax_timeline_1, timeline_1)
        draw_timeline(ax_timeline_2, timeline_2)

        ax_timeline_1.set_ylabel("Stages")
        ax_timeline_2.set_ylabel("Steps")

        # Shared x-axis
        #ax_timeline_2.set_xlabel("Time")
        ax_timeline_2.set_xlim(0, 1)
        ax_timeline_2.set_xticks([])

        # Hide duplicated x-axis labels on upper plots
        ax_function.tick_params(axis="x", labelbottom=False)
        ax_timeline_1.tick_params(axis="x", labelbottom=False)

        # Clean up borders
        for ax in axes[1:]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

        # The legend for the triptych colors
        labels = ["Amine", "Aldehyde 1", "Imine 1", "Aldehyde 2", "Imine 2"]

        handles = [
            Line2D([0], [0], color=color, linewidth=2)
            for color in colors
        ]

        ax_function.legend(
            handles=handles,
            labels=labels,
            ncols=5,          # one item per column
            loc="upper right",
            #frameon=False
        )

        if not mo.running_in_notebook():
            fig.savefig("graphics/stages-steps." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md("Figure: Schematic representation of a two-step experiment and its stages. The experiment starts by taking calibration curves for the reactants in the dosage stages. It determines the rate constant of their spontaneous reaction in the monitoring phase. The addition of the catalyst starts the first reaction phase. Once equilibrium is reached, a new reactant is added to start the second reaction step. Since the catalyst is already added, an external calibration dataset must be used in addition to the in situ spectra. During the second reaction stage, the mixture is again monitored until it reaches equilibrium."))

    _cell()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MCD explanation graphic
    """)
    return


@app.cell
def _(deconv):
    yellow =  "#b58900"
    orange =  "#cb4b16"
    red =     "#dc322f"
    magenta = "#d33682"
    violet =  "#6c71c4"
    blue =    "#268bd2"
    cyan =    "#2aa198"
    green =   "#859900"

    PT = deconv.PeakTrace
    pt_colors = {
        PT.DISAPPEARING: red,
        PT.SHADOW: blue,
        PT.STABLE: yellow,
        PT.APPEARING: green,
        PT.NOISE: "gray",
        PT.UNCLASSIFIED: "black",
    }
    return PT, blue, cyan, green, magenta, pt_colors, red, violet


@app.cell
def _(thio_chkpts):
    def _load():
        out = {}
        # We reuse the data from a lower graphic
        solv = "MeOH"
        chkpt = thio_chkpts[solv][3]
        dataset = chkpt.dataset
        exp = dataset.experiment

        i_snap = -21

        zeta = chkpt.metadata["zeta"]

        out["wvn"] = exp.attrs["ir-x"]
        out["solv"] = exp["ir"].iloc[0]
        out["full-solv"] = exp["ir-raw"].iloc[0]
        out["ref"] = zeta["solvent-meta"]["mixture"]
        out["ref-solv-factor"] = zeta["solvent-meta"]["zeta"]

        out["target"] = zeta["snapshot-meta"][i_snap]["mixture"]
        out["target-solv-factor"] = 0
        out["zeta"] = [0.3, zeta["snapshot-meta"][i_snap]["zeta"], 0.7]
        out["topout-zeta"] = zeta["snapshot-meta"][i_snap]["topout-zeta"]


        return out

    mcdfig_data = _load()
    return (mcdfig_data,)


@app.cell(hide_code=True)
def _(
    Line2D,
    PT,
    blue,
    green,
    magenta,
    mcd_inputs,
    mcd_opt,
    mcd_triptych,
    mo,
    outformat,
    plt,
    pt_colors,
    red,
):
    # Combine all three small plots into one
    def _mcd_plot():
        factor = 2
        fig = plt.figure(figsize=(170 / 25.4 * factor, 2.5 * factor))

        outer = fig.add_gridspec(
            nrows=4,
            ncols=2,
            width_ratios=[1, 1],
            height_ratios=[1, 1, 0.2, 0.097],
            wspace=0.05,
            hspace=0.05,
        )

        left_top = fig.add_subfigure(outer[0, 0])
        left_middle = fig.add_subfigure(outer[1, 0])
        left_equation = fig.add_subfigure(outer[2, 0])

        # One right-hand subfigure spanning all three rows
        right = fig.add_subfigure(outer[:, 1])

        mcd_inputs(left_top)
        mcd_opt(left_middle)
        mcd_triptych(right)

        for subfig, label in zip((left_top, left_middle, right), "abc"):
            x = 0.05
            y = 0.87

            if label == "c":
                x = 0.02
                y = 0.943

            label = label + ")"
            subfig.text(
                x, y, label,
                ha="left",
                va="top",
                #fontsize=12,
                fontweight="bold"
            )

        # Decomposition equation
        parts = [
            (r"$A_{\mathrm{p}}$", blue),
            (r"$=$", "black"),
            (r"$\;A(t)$", magenta),
            (r"$-$", "black"),
            (r"$\;A_{\mathrm{const}}$", green),
            (r"$-\;\zeta(t)\cdot$", "black"),
            (r"$\;A_{\mathrm{r}}$", red),
        ]
        eq_ax = left_equation.add_axes([0, 0, 1, 1])
        eq_ax.set_axis_off()
        x, y = 0.336, -0.45
        fontsize = 16

        for text, color in parts:
            t = eq_ax.text(
                x,
                y,
                text,
                color=color,
                transform=eq_ax.transAxes,
                fontsize=fontsize,
                va="baseline",
                ha="left",
            )

            # Render so Matplotlib can calculate the text width
            fig.canvas.draw()

            renderer = fig.canvas.get_renderer()
            text_bbox = t.get_window_extent(renderer=renderer)
            axes_bbox = eq_ax.get_window_extent()

            # Convert pixel width to axes coordinates
            x += text_bbox.width / axes_bbox.width    

        # The legend for the triptych colors
        order = [PT.STABLE, PT.DISAPPEARING, PT.SHADOW, PT.NOISE]
        colors = [pt_colors[x] for x in order]
        labels = ["Stable", "Reactant", "Artefact", "Unclear"]

        handles = [
            Line2D([0], [0], color=color, linewidth=2)
            for color in colors
        ]

        right.legend(
            handles=handles,
            labels=labels,
            ncols=5,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
            frameon=False
        )

        fig.canvas.draw()

        dy = +0.068
        for ax in right.axes:
            pos = ax.get_position()
            ax.set_position((pos.x0, pos.y0 + dy, pos.width, pos.height))

        if not mo.running_in_notebook():
            fig.savefig("graphics/mcd-explanation." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md("Figure: Steps of the minimum-complexity deconvolution algorithm. a) As input, the algorithm only requires the mixture spectrum to be deconvolved and a spectrum of the background signals, which are usually reactant spectra. The solvent spectrum is already subtracted and signal around 1000 cm-1 is removed due to spectrometer saturation. b) Varying the background fraction ζ, different candidate spectra are obtained. Some background features are highlighted with vertical lines for easier comparison. If ζ is too low, background features remain (i). If it is too high, the subtraction of background features leads to spurious negative signals (iii). At the correct background fraction, the background signals are mostly removed and the number of features remaining in the spectrum is minimal (ii). c) Plot of the features identified by the algorithm and their count as the background fraction is varied. The centre plot marks whether individual features are observed or not at a given value. Blue: Stable signals, either strong analyte signal or overlapping background and analyte signals. Green: Background features that disappear. Magenta: Spurious minima due to excessive subtraction. Red: Analyte features that appear at higher ζ. They are not considered as features because they are often artefacts of peak shifts. Grey: Shallow or unclear signals. Bottom: Number of detected features, counting only green and magenta signals. The minimum corresponds to the ideal background fraction."))

    _mcd_plot()
    return


@app.cell(hide_code=True)
def _(Line2D, deconv, mcdfig_data, np, ossc, plt, pt_colors):
    def mcd_triptych(infig = None):
        # The third part: Traces of the peaks as zeta progresses, colored for their roles
        data = mcdfig_data

        # Count the number of maxima with a given prominence or minima that come out of them.
        wvn = data["wvn"]
        start = data["ref"]
        mixture = data["target"]
        include_appearing = False

        # Just to wrap it in the editor
        if True:
            peak_traces, minimum_traces, extremum_pairs = ossc.deconvolution.get_peak_traces(start, mixture)

            n_over_zeta = np.zeros(len(ossc.zeta_space), dtype=int)

            for trace in peak_traces:
                if trace.group in [deconv.PeakTrace.SHADOW, deconv.PeakTrace.DISAPPEARING] or include_appearing and trace.group == deconv.PeakTrace.APPEARING:
                    zeta0 = min(trace.y[0], trace.y[-1])
                    zeta1 = max(trace.y[0], trace.y[-1])

                    # Optional: If there is a corresponding minimum but no gap due to shifts, open it artificially
                    if True:
                        if trace.group == deconv.PeakTrace.DISAPPEARING:
                            negative = extremum_pairs.get(trace, None)
                            if negative is not None and len(negative) + len(trace) >= len(deconv.zeta_space):
                                # Open a gap until twice the prominence at the end of the disappearing trace
                                # Since it is always disappearing, we know the order
                                zeta1 = trace.y[np.argmin(np.abs((trace.prom - 2*trace.prom[-1])))]

                    for i_zeta, zeta in enumerate(deconv.zeta_space):
                        if zeta0 <= zeta <= zeta1:
                            n_over_zeta[i_zeta] += 1

            # Check whether there is one global minimum. If not, take the one closer to the topout value.

            def find_minima(a):
                a = np.asarray(a)

                if a.size == 0:
                    return []

                m = int(a.min())
                is_min = (a == m).astype(int)
                diff = np.diff(is_min)
                starts = np.nonzero(diff == 1)[0]
                ends   = np.nonzero(diff == -1)[0]

                # Both ends are now the first within the valley
                starts = starts + 1

                if len(starts) > len(ends):
                    # One valley at the far end
                    ends = np.array([len(a) - 1])
                elif len(ends) < len(starts):
                    starts = np.array([0])

                return [(deconv.zeta_space[int(s)], deconv.zeta_space[int(e)]) for s,e in zip(starts, ends)]

            minima_valleys = find_minima(n_over_zeta)

            if len(minima_valleys) < 1:
                return None

            best_valley = minima_valleys[0]

            vert_lines = []

            if len(minima_valleys) > 1:
                best_distance = np.inf

                for valley in minima_valleys:
                    dist = min(abs(valley[0] - data["topout-zeta"]), abs(valley[1] - data["topout-zeta"]))
                    if dist < best_distance:
                        best_distance = dist
                        best_valley = valley

                vert_lines.extend(best_valley)


        if infig is None:
            fig = plt.figure(figsize=(10, 6))
        else:
            fig = infig

        gs = plt.GridSpec(2, 3, width_ratios=[0.15, 1.0, 0.15], height_ratios=[0.7, 0.2], figure=fig, hspace=0, wspace=0)

        # Top row: three axes that share y
        ax_zeta0  = fig.add_subplot(gs[0, 0])
        ax_traces   = fig.add_subplot(gs[0, 1], sharey=ax_zeta0)
        ax_zeta1 = fig.add_subplot(gs[0, 2], sharey=ax_zeta0)

        # Bottom row: three axes; make the left/right narrow and invisible
        ax_bot_left  = fig.add_subplot(gs[1, 0])
        ax_count   = fig.add_subplot(gs[1, 1], sharex=ax_traces)
        ax_bot_right = fig.add_subplot(gs[1, 2])

        # Hide bottom-left and bottom-right but keep their space narrow
        ax_bot_left.set_axis_off()
        ax_bot_right.set_axis_off()


        # Remove ticks and labels everywhere first
        ax_traces.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_traces.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        ax_zeta0.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_zeta1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_zeta1.tick_params(axis="y", which="both", left=False, labelleft=False, right=True, labelright=True)

        ax_zeta0.set_ylabel("Wavenumber $ν$ [cm$^{-1}$]")
        ax_zeta1.set_ylabel("Wavenumber $ν$ [cm$^{-1}$]", rotation=270, labelpad=15, va="bottom")
        ax_zeta1.yaxis.set_label_position("right")

        zeta0_label = Line2D(
            [], [], 
            linestyle="none",
            marker=None,
            label="$ζ$=0"
        )

        zeta1_label = Line2D(
            [], [], 
            linestyle="none",
            marker=None,
            label="$ζ$=1"
        )

        ax_zeta0.plot(mixture, data["wvn"], color="k")
        ax_zeta1.plot(-mixture + start, data["wvn"], color="k")

        ax_zeta0.legend(loc="upper center", handles=[zeta0_label], handlelength=0, handletextpad=0,)
        ax_zeta1.legend(loc="upper center", handles=[zeta1_label], handlelength=0, handletextpad=0)
        ax_zeta0.set_ylim((min(data["wvn"]), max(data["wvn"])))

        ax_count.set_xlim(0, 1)
        ax_count.plot(deconv.zeta_space, n_over_zeta, color="k")
        ax_count.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_count.set_ylabel("Features")

        ax_count.set_xlabel("Reactant fraction $ζ$ [1]")

        # Plot the PeakTraces
        class_colors = pt_colors
        for trace in peak_traces:
            color = class_colors.get(trace.group)
            ax_traces.plot(trace.y, wvn[trace.x], color=color)

            if trace.group == deconv.PeakTrace.DISAPPEARING:
                ax_zeta0.axhline(data["wvn"][trace.x[0]], color=class_colors[deconv.PeakTrace.DISAPPEARING], alpha=0.5)
            elif trace.group == deconv.PeakTrace.SHADOW:
                ax_zeta1.axhline(data["wvn"][trace.x[0]], color=class_colors[deconv.PeakTrace.SHADOW], alpha=0.5)
            elif trace.group == deconv.PeakTrace.STABLE:
                ax_zeta0.axhline(data["wvn"][trace.x[-1]], color=class_colors[deconv.PeakTrace.STABLE], alpha=0.5)
                ax_zeta1.axhline(data["wvn"][trace.x[0]], color=class_colors[deconv.PeakTrace.STABLE], alpha=0.5)

        #if infig is None:
        #    mo.output.append(fig)

    mcd_triptych()
    return (mcd_triptych,)


@app.cell(hide_code=True)
def _(blue, cyan, mcdfig_data, plt, violet):
    def mcd_opt(infig=None):
        data = mcdfig_data
        wvn = data["wvn"]

        #fig = plt.figure(figsize=(10, 4))
        if infig is None:
            fig = plt.figure(figsize=(6, 3))
        else:
            fig = infig

        gs = fig.add_gridspec(3, hspace=0)
        axes = gs.subplots(sharex=True)

        ax0 = axes[0]
        ax0.set_xlim((min(wvn), max(wvn)))

        ref = data["ref"] - data["ref-solv-factor"] * data["solv"]
        target = data["target"] - data["target-solv-factor"] * data["solv"]

        features = [1216, 1500, 1660]

        axes[2].axhline(0, color="black", ls="-.")

        cols = [violet, blue, cyan]

        for i_zeta, zeta in enumerate(data["zeta"]):
            ax = axes[i_zeta]
            ax.plot(wvn, target - zeta * ref, label=f"$\\zeta={zeta:0.2f}$", color=cols[i_zeta])
            for f in features:
                ax.axvline(f, color="gray", alpha=0.5, ls="--")
            ax.legend(loc="upper right")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            if i_zeta != len(data["zeta"]) - 1:
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

            ax.text(
                0.005, 0.97, (i_zeta + 1) * "i" + ")",
                transform=ax.transAxes,
                ha="left", va="top",
            )

        axes[-1].set_xlabel(r"Wavenumber $ν$ [cm$^{-1}$]")
        fig.supylabel("Absorbance $A$", x=0.09)

        #if infig is None:
        #    mo.output.append(fig)

    mcd_opt()
    return (mcd_opt,)


@app.cell(hide_code=True)
def _(green, magenta, mcdfig_data, plt, red):
    def mcd_inputs(infig=None):
        data = mcdfig_data
        wvn = data["wvn"]

        # 10, 2 for the current article figure
        # 6, 3 for the poster
        if infig is None:
            fig = plt.figure(figsize=(6, 2))
        else:
            fig = infig

        gs = fig.add_gridspec(3, hspace=0)
        axes = gs.subplots(sharex=True)

        solv_axis = axes[0]
        solv_axis.set_xlim((min(wvn), max(wvn)))

        legend_pos = "upper right"

        solv_axis.plot(wvn, data["solv"], label="Solvent", color=green)
        solv_axis.plot(wvn, data["full-solv"], label="Noisy region", color=green, ls="--")
        solv_axis.legend(loc=legend_pos)


        educt_axis = axes[1]
        educt_axis.plot(wvn, data["ref"] - data["ref-solv-factor"] * data["solv"], label="Pre-reaction", color=red)
        educt_axis.legend(loc=legend_pos)
        educt_axis.set_ylim([None, 0.8])

        target_axis = axes[2]
        target_axis.set_ylim([None, 0.8])

        target_axis.plot(wvn, data["target"], label="Reaction", color=magenta)
        target_axis.legend(loc=legend_pos)
        target_axis.set_xlabel(r"Wavenumber $ν$ [cm$^{-1}$]")
        fig.supylabel("Absorbance $A$", x=0.09)

        solv_axis.tick_params(axis="y", which="both", left=False, labelleft=False)
        educt_axis.tick_params(axis="y", which="both", left=False, labelleft=False)
        target_axis.tick_params(axis="y", which="both", left=False, labelleft=False)

        solv_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        educt_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        #if infig is None:
        #    mo.output.append(fig)

    mcd_inputs()
    return (mcd_inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectra for simulated experiments
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, outformat, plt, rctstyle, sim, single_col):
    def _cell():
        fig = plt.figure(figsize=(single_col, 3.5))
        gs = fig.add_gridspec(2, hspace=0)
        axes = gs.subplots(sharex=True)

        chemicals = ["Isopropanol", "Benzaldehyde", "Thiophene-2-carboxaldehyde", "Aniline", "Thioimine", "Benzimine"]

        def render_set(axis, set):
            axis.tick_params(axis="y", which="both", left=False, labelleft=False)

            for i_chem, chem in enumerate(chemicals):
                color, _, label = rctstyle.get(chem, ("k", "-", chem))
                spec = set[chem]

                axis.plot(wvn, len(chemicals) - i_chem - 1 + spec, color=color, label=label)

        wvn = np.linspace(800, 1800, 501)

        render_set(axes[0], sim.spectra[sim.SpectraKind.NOOVERLAP])
        render_set(axes[1], sim.spectra[sim.SpectraKind.AUTHENTIC])

        axes[0].set_xlim((min(wvn), max(wvn)))

        legend_pos = "upper right"
        axes[0].legend(loc=legend_pos)

        axes[-1].set_xlabel(r"Wavenumber $\nu$ [cm$^{-1}$]")
        fig.subplots_adjust(left=0.06)
        fig.supylabel(r"Absorbance $A$")

        axes[0].text(
            0.005, 0.98, "a)",
            transform=axes[0].transAxes,
            ha="left", va="top"
        )

        axes[1].text(
            0.005, 0.98, "b)",
            transform=axes[1].transAxes,
            ha="left", va="top"
        )

        if not mo.running_in_notebook():
            fig.savefig("graphics/simulation-spectra." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md("Figure: Pure component spectra used by the digital twin: a) Non-overlapping and b) extracted from experimental data"))

    _cell()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Solvent subtraction
    """)
    return


@app.cell
def _(ossc):
    def _load():
        ba_campaign = ossc.read_campaign("references/benzaldehyde-external.h5")

        out = {}

        for handle, experiment in ba_campaign.items():
            ossc.prepare_experiment(experiment, 800, 1800)
            solv = ossc.get_solvent(experiment)

            dataset = ossc.Dataset(experiment, (800, 1800))

            out[handle] = ossc.eval_dataset(dataset, verbose=True)

        return out

    ba_campaign = _load()
    return (ba_campaign,)


@app.cell
def _():
    import matplotlib as mpl

    return (mpl,)


@app.cell(hide_code=True)
def _(ba_campaign, mo, mpl, np, outformat, plt, single_col):
    def _solvent_subtraction():
        stack_offset = 0.5
        wavenumbers = np.linspace(800, 1800, 501)

        fig = plt.figure(figsize=(single_col, 3))
        ax = fig.subplots(1)

        labels = ["MeOH", "EtOH", "iPrOH", "nPrOH", "nBuOH", "DMF", "DMAc", "NMP", "Dioxane", "Acetonitrile", "PhCN"]
        colors = ["#002b36", "#586e75", "#8ba0a4"] + list(mpl.style.library["Solarize_Light2"]['axes.prop_cycle'].by_key()['color'])

        for i_solv, handle in enumerate(labels):
            checkpoint = ba_campaign[handle]
            spec = checkpoint.metadata["mcd-spectra"]["Benzaldehyde"]

            ax.plot(wavenumbers, i_solv * stack_offset + spec, label=handle, gapcolor="k", color=colors[i_solv % len(labels)])

        ax.set_xlim(800, 1800)

        fig.legend(loc="upper left", ncol=4, bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)

        ax.set_xlabel(r"Wavenumber $ν$ [cm$^{-1}$]")
        ax.set_ylabel(r"Absorbance $A$")
        ax.set_yticks([])

        plt.tight_layout()

        if not mo.running_in_notebook():
            fig.savefig("graphics/solvent-subtraction." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md("Figure: Spectra of benzaldehyde dissolved in eleven different solvents after background subtraction via minimum complexity deconvolution. While the extracted peaks are always identical, their shapes and relative amplitudes are affected by solution effects. Some signal is cut due to saturation by solvent signal."))

    _solvent_subtraction()
    return


@app.cell
def _(ba_campaign, mo, np, ossc, outformat, plt, single_col):
    def _feature_plot():
        mecn = ba_campaign["Acetonitrile"]
        nmp = ba_campaign["NMP"]

        wavenumbers = np.linspace(800, 1800, 501)

        mecn_counts = mecn.metadata["mcd-meta"]["Benzaldehyde"]
        nmp_counts = nmp.metadata["mcd-meta"]["Benzaldehyde"]

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            sharex=True,
            figsize=(single_col, 3),
            gridspec_kw={"hspace": 0}
        )

        # Example data
        x = ossc.zeta_space
        ax1.plot(x, mecn_counts["n-over-zeta"], color="k", label="Acetonitrile")
        ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax1.legend(loc="lower left")
        ax1.set_xlim(0, 1)

        ax2.plot(x, nmp_counts["n-over-zeta"], color="k", label="NMP")
        ax2.set_xlabel("Solvent fraction $ζ$ [1]")
        ax2.legend(loc="lower left")
        ax2.set_xlim(0, 1)

        fig.supylabel("Feature count $n$")

        if not mo.running_in_notebook():
            fig.savefig("graphics/zeta-plot-solvents." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md("Figure: Plot of the number of background features in the subtraction of solvent spectra of benzaldehyde dissolved in acetonitrile (top) and N-Methyl-2-pyrrolidone (NMP, bottom). Since NMP has more peaks in its spectrum, the algorithm can more precisely narrow down the optimal background fraction, which is visible from the narrower and deeper valley in its plot."))

    _feature_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5-component mixture
    """)
    return


@app.cell
def _(ossc):
    mixture_exp = ossc.read_campaign("mixture-5-compounds.h5")["exp"]
    return (mixture_exp,)


@app.cell
def _(
    colors,
    copy,
    external_calibrations,
    mixture_exp,
    mo,
    ossc,
    outformat,
    plt,
    single_col,
):
    def eval_mixture(ir_range, n_external=0, render_figure=False):
        exp = copy.deepcopy(mixture_exp)

        n_internal = len(exp.attrs["chemicals"]) - 1 - n_external

        ossc.prepare_experiment(exp, *ir_range)
        mixture_ds = ossc.Dataset(exp, ir_range, external_calibs=external_calibrations[ossc.get_solvent(exp)], n_internal=n_internal)
        chkpt = ossc.eval_dataset(mixture_ds, False)

        if n_external > 0:
            external_str = f"external dosage for {n_external} component"
            if n_external > 1:
                external_str += "s"
        else:
            external_str = f"internal dosage"

        headline = f"## 5-component mixture results with an {external_str} on {ir_range[0]}-{ir_range[1]} " + "cm$^{-1}$"

        table_data = {}
        data = ossc.evaluate_mixture(chkpt)

        table_data["Chemical"] = data["Chemical"]
        table_data["Correlation (Prediction <-> Exact)"] = [round(x, 4) for x in data["Correlation (Prediction <-> Exact)"]]
        table_data["RMSD [Vol%]"] = [f"{100*x:0.1f}" for x in data["Average error"]]
        mo.output.append(mo.md(headline))
        mo.output.append(mo.md("Table: Correlation between exact values and calculated concentrations for each chemical as well as their root mean square deviation."))
        mo.output.append(mo.ui.table(selection=None, data=table_data))

        if not render_figure:
            return

        refs = {chem: mixture_ds.get_true_conc(chem) for chem in chkpt.pred_conc.keys()}
        time = exp["Time since start"].to_numpy() / 3600

        # Plot the trends separately
        fig = plt.figure(figsize=(single_col, 4))
        gs = fig.add_gridspec(len(chkpt.pred_conc) - 1, hspace=0)
        axes = gs.subplots(sharex=True)

        axes[0].set_xlim((min(time), max(time)))
        axes[-1].set_xlabel(r"Time $t$ [h]")
        axes[len(axes) // 2].set_ylabel(r"Volume fraction $φ$ [%]")

        legend_pos = "lower right"

        for i_chem, (chem, prediction) in enumerate(chkpt.pred_conc.items()):
            if i_chem == 0:
                # Skip solvent
                continue

            truth = chkpt.dataset.get_true_conc(chem)
            max_conc = max(truth)
            axis = axes[i_chem - 1]

            color = colors[i_chem - 1]
            axis.plot(time, 100 * truth, color=color, alpha=0.5)
            axis.plot(time, 100 * prediction, label=chem, color=color, ls="--")
            axis.legend(loc=legend_pos)

            if i_chem + 1 != len(chkpt.pred_conc):
                axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        if not mo.running_in_notebook():
            fig.savefig(f"graphics/5-comp-mixture-{ir_range[0]}-{ir_range[1]}-{external_str}." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md(f"Figure: Concentration curves for an experiment that mixes chemicals without a chemical reaction. Solid lines show the true concentrations, dashed lines the predictions by OSSC. At the start of the experiment, the analytes are added separately and in small portions. After dilution to stay within the calibration range, arbitrary amounts of the analytes are added to create a diverse test set."))

    return (eval_mixture,)


@app.cell
def _(eval_mixture):
    eval_mixture((800, 1800), render_figure=True)
    return


@app.cell
def _(eval_mixture):
    eval_mixture((800, 1800), n_external=1, render_figure=True)
    return


@app.cell
def _(eval_mixture):
    eval_mixture((800, 1800), n_external=2, render_figure=True)
    return


@app.cell
def _(eval_mixture):
    eval_mixture((800, 1800), n_external=5, render_figure=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reaction R1+R2
    """)
    return


@app.cell
def _(external_calibrations, mo, ossc):
    def _load_reactions():
        reactions = ossc.read_campaign("thioimine-exchanges.h5")

        out = {}

        for handle, exp in mo.status.progress_bar(
            reactions.items(),
            total=len(reactions),
            title="Processing two-step reactions",
        ):
            products = ["Thioimine", "Benzimine"]

            ossc.prepare_experiment(exp, 800, 1800)
            solv = ossc.get_solvent(exp)
            ext_calibs = external_calibrations[solv]

            dataset = ossc.Dataset(exp, (800, 1800), products, 
                   n_internal=1000,
                   catalyst="Acetic acid",
                   external_calibs=ext_calibs
            )

            dataset.subsets[-1].add_byproduct("Thiophene-2-carboxaldehyde")

            chkpts = []
            checkpoint = None

            for i in range(6):
                checkpoint = ossc.eval_subset(dataset, i, checkpoint, verbose=True)
                chkpts.append(checkpoint)

            out[handle] = chkpts

        return out

    thio_chkpts = _load_reactions()
    return (thio_chkpts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First reaction step
    """)
    return


@app.cell
def _(mo, np, ossc, outformat, plt, rctstyle, sim, solvent_style, thio_chkpts):
    def _cell():
        ti_yields = {}
        ti_spectra = {}
        kfs = {}
        epsilons = {}

        for chem, style in solvent_style.items():
            handle = style[2]
            chkpts = thio_chkpts[handle]

            chkpt = chkpts[3]
            dataset = chkpt.dataset
            solvent = dataset.solvent
            mo.output.append(mo.md(f"### Reaction performed in {solvent}"))

            exp = chkpt.experiment

            # Actual reaction
            scaling = {}
            if exp.attrs.get("simulated", False):
                scaling = {"Benzimine": 2, "Thioimine": 2}

            starting_mats = {
                "Benzimine": ("Aniline", "Benzaldehyde"),
                "Thioimine": ("Aniline", "Thiophene-2-carboxaldehyde"),
            }

            fig, ax = ossc.render_equconc(chkpt, rctstyle, ref_scaling=scaling, starting_mats=starting_mats, exclude=[dataset.solvent], legend_loc="center right", highlight=["Acetic acid"])

            if not mo.running_in_notebook():
                fig.savefig(f"graphics/r1-conc-{handle}." + outformat, bbox_inches="tight")

            mo.output.append(fig)
            mo.output.append(mo.md(f"Figure: Concentrations of reactants and products for reaction R1 performed in {solvent}"))

            equ_factors = ossc.get_equ_factors(chkpt, starting_mats=starting_mats)
            conc = {chem: pred * equ_factors[chem] for chem, pred in chkpt.pred_conc.items()}

            monitoring = ossc.evaluate_monitoring(chkpt)

            ti_yields[handle] = f"{100 * conc["Thioimine"][-1]:0.1f}"
            ti_spectra[handle] = chkpt.metadata["mcd-spectra"]["Thioimine"]
            kfs[handle] = f"{10**6 * monitoring["kf"]:0.0f}"
            epsilons[handle] = f"{100 * monitoring["epsilon"]:0.2f}"

            fig, ax = ossc.render_zeta_surface(chkpt.metadata["zeta"])
            mo.output.append(fig)

            if not mo.running_in_notebook():
                fig.savefig(f"graphics/r1-zeta-{handle}." + outformat, bbox_inches="tight")

            mo.output.append(mo.md(f"Figure: Feature plot over time for reaction R1 performed in {solvent} with values from decomposing single spectra (dashed) and smoothed to the first PC (solid line)"))

            prod_spec = chkpt.metadata["zeta"]["product-smooth"]
            fig, ax = ossc.render_3d(prod_spec[0], np.array(prod_spec[1]) / 3600, prod_spec[2], xlabel="Wavenumber $ν$ [cm$^{-1}$", ylabel="Absorbance $A$", zlabel="Time $t$ [h]")
            ax.set_xlim(800, 1800)

            mo.output.append(fig)

            if not mo.running_in_notebook():
                fig.savefig(f"graphics/r1-product-{handle}." + outformat, bbox_inches="tight")

            mo.output.append(mo.md(f"Figure: Product spectrum as extracted from every individual snapshot of reaction R1 in {handle} using the scaled score of the first principal component to separate the raw spectra. The product spectrum is obtained as the average of the latter half of these spectra."))

        mo.output.append(mo.md("Table: Yields for the first reaction step R1 performed in different solvents"))
        data = {
            "Solvent": list(ti_yields.keys()),
            "Yield [%]": list(ti_yields.values()),
            "k_f [10^6 / (%Vol s)]": list(kfs.values()),
            "epsilon [%]": list(epsilons.values()),
        }
        mo.output.append(mo.ui.table(selection=None, data=data))


        # Stacked plot of all product spectra
        fig, ax = plt.subplots(figsize=(6, 4))
        wvn = np.linspace(800, 1800, 501)

        ax.set_xlim((min(wvn), max(wvn)))
        ax.set_ylabel(r"Absorbance $A$")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        ax.set_xlabel(r"Wavenumber $ν$ [cm$^{-1}$]")

        legend_pos = "upper right"

        ax.plot(wvn, sim.spectra[sim.SpectraKind.AUTHENTIC]["Thioimine"], color="k", label="Reference", zorder=len(ti_spectra))

        for i_solv, (handle, spectrum) in enumerate(ti_spectra.items()):
            # Subtract the solvent via another MCD
            solv = ossc.get_solvent(thio_chkpts[handle][3].experiment)
            solvent, _ = ossc.get_solvent_spectrum(thio_chkpts[handle][3].experiment)

            #_, spectrum, _ = ossc.mcd(solvent, spectrum)

            color, _, label = solvent_style[solv]
            ax.plot(wvn, spectrum + 0.5*(i_solv + 1), color=color, label=label, zorder=len(ti_spectra) - 1 - i_solv)

        ax.legend(loc="upper left", ncol=3)
        ax.set_ylim(-0.05, 5.1)

        if not mo.running_in_notebook():
            fig.savefig(f"graphics/r1-product-thioimine." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md("Figure: Spectra of the product N-(Thiophene-2-ylmethylene)aniline extracted from spectra of reaction R1 in five different solvents. There are some artefacts, mostly due to solvent peaks that moved and could thus not be subtracted perfectly, but even small spectral features are reliably extracted."))

    _cell()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Second reaction step
    """)
    return


@app.cell
def _(mo, np, ossc, outformat, plt, rctstyle, sim, solvent_style, thio_chkpts):
    def _cell():
        ti_yields = {}
        bi_yields = {}
        bi_spectra = {}

        for chem, style in solvent_style.items():
            handle = style[2]
            chkpts = thio_chkpts[handle]

            chkpt = chkpts[-1]
            dataset = chkpt.dataset
            solvent = dataset.solvent
            mo.output.append(mo.md(f"### Reaction performed in {solvent}"))

            exp = chkpt.experiment

            # Actual reaction
            scaling = {}
            if exp.attrs.get("simulated", False):
                scaling = {"Benzimine": 2, "Thioimine": 2}

            starting_mats = {
                "Benzimine": ("Aniline", "Benzaldehyde"),
                "Thioimine": ("Aniline", "Thiophene-2-carboxaldehyde"),
            }

            equ_factors = ossc.get_equ_factors(chkpt, starting_mats=starting_mats)
            conc = {chem: pred * equ_factors[chem] for chem, pred in chkpt.pred_conc.items()}

            anil_conc = conc["Aniline"]
            ti_conc = conc["Thioimine"]

            for subset in dataset.subsets:
                if "Benzaldehyde" in subset.chemicals:
                    first_step_end = subset.exp.index[0]
                    break
            else:
                raise Exception("This seems to be a two-step reaction, but the Benzaldehyde dosage cannot be found.")

            second_step_start = dataset.subsets[-1].exp.index[0]

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

            fig, ax = ossc.render_conc(chkpt, conc, rctstyle, legend_loc="upper right", highlight=["Benzaldehyde"])

            if not mo.running_in_notebook():
                fig.savefig(f"graphics/r2-conc-stoichio-{handle}." + outformat, bbox_inches="tight")

            mo.output.append(fig)
            mo.output.append(mo.md(f"Figure: Concentrations over time predicted for the two-step reaction R1+R2 performed in {handle}. The concentrations are for the aldehydes and N-benzylideneaniline are derived from the reactions' stoichiometry."))

            ti_yields[handle] = f"{100 * conc["Thioimine"][-1]:0.1f}"
            bi_yields[handle] = f"{100 * conc["Benzimine"][-1]:0.1f}"
            bi_spectra[handle] = chkpt.metadata["mcd-spectra"]["Benzimine"]

        mo.output.append(mo.md("Table: Yields after the second reaction step in different solvents"))
        data = {
            "Solvent": list(ti_yields.keys()),
            "Yield R1 [%]": list(ti_yields.values()),
            "Yield R3 [%]": list(bi_yields.values()),
        }
        mo.output.append(mo.ui.table(selection=None, data=data))

        # Stacked plot of all product spectra
        fig, ax = plt.subplots(figsize=(6, 4))
        wvn = np.linspace(800, 1800, 501)

        ax.set_xlim((min(wvn), max(wvn)))
        ax.set_ylabel(r"Absorbance $A$")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        ax.set_xlabel(r"Wavenumber $ν$ [cm$^{-1}$]")

        legend_pos = "upper right"

        ax.plot(wvn, sim.spectra[sim.SpectraKind.AUTHENTIC]["Benzimine"] * 1.1 * 0.5, color="k", label="Reference", zorder=len(bi_spectra))

        for i_solv, (handle, spectrum) in enumerate(bi_spectra.items()):
            # Subtract the solvent via another MCD
            solv = ossc.get_solvent(thio_chkpts[handle][3].experiment)

            color, _, label = solvent_style[solv]
            spectrum = np.where(np.logical_or(spectrum > 0, np.isnan(spectrum)), spectrum, 0)
            ax.plot(wvn, spectrum + 0.5*(i_solv + 1), color=color, label=label, zorder=len(bi_spectra) - 1 - i_solv)

        ax.legend(loc="upper left", ncol=3)
        ax.set_ylim(-0.05, 3.9)

        if not mo.running_in_notebook():
            fig.savefig(f"graphics/r2-product-benzimine." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md("Figure: Spectra of the final product N-benzylidene aniline extracted from the two-step reaction R1+R2. Their quality is similar to that observed for the first step products (figure 6): The spectral features of the analyte are recovered well, with artefacts of shifting peaks visible in some locations. The algorithm returns negative peaks for some wave numbers, which were set to zero. "))

    _cell()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulations
    """)
    return


@app.cell
def _(ossc, sim):
    def _load():
        exp = sim.run_reaction(sim.spectra[sim.SpectraKind.AUTHENTIC], "Isopropanol", ["Aniline", "Thiophene-2-carboxaldehyde", "Thioimine", "Water"], 0.02, 16, has_cat=True, kf_nocat=0.0005, monitoring_duration=10)

        ossc.prepare_experiment(exp, 800, 1800)
        solv = ossc.get_solvent(exp)
        ext_calibs = sim.spectra[sim.SpectraKind.AUTHENTIC]

        dataset = ossc.Dataset(exp, (800, 1800), ["Thioimine"], 
               n_internal=1000,
               catalyst="Acetic acid",
               external_calibs=sim.spectra[sim.SpectraKind.AUTHENTIC]
        ) 

        single_chkpt = ossc.eval_dataset(dataset)

        experiment = ossc.run_reaction(
            sim.spectra[sim.SpectraKind.NOOVERLAP], "Isopropanol", ["Aniline", "Thiophene-2-carboxaldehyde", "Thioimine", "Water", "Benzaldehyde", "Benzimine"], 0.02, 16, has_cat=True, kf_nocat=0.0005, kf2=0.0025, K2=0.5, kf3=0.1, K3=16, monitoring_duration=10, nonprop=sim.NonProp.NONE
            )

        ossc.prepare_experiment(experiment, 800, 1800)
        solv = ossc.get_solvent(experiment)

        selspectra = sim.spectra[sim.SpectraKind.NOOVERLAP]

        ext_calibs = {}

        # Generate the external calibrations on the spot
        for chem in experiment.attrs["chemicals"]:
            max_volfrac = max(experiment[f"{chem} volume fraction"])

            V_solv = 25
            if chem == solv:
                continue
            else:
                V_chem = V_solv * max_volfrac / (1 - max_volfrac)

            # If it is fully linear, there is no point in using more
            exp = sim.DigiExperiment(selspectra, nonprop=sim.NonProp.NONE)

            exp.start_stage("Solvent")
            exp.add_chemical(solv, V_solv, stirring=0)

            exp.start_stage(f"{chem} dosage")
            exp.add_chemical(chem, V_chem)

            ext_calibs[chem] = exp.get_experiment()

        dataset = ossc.Dataset(experiment, (800, 1800), ["Thioimine", "Benzimine"], 
               n_internal=1000,
               catalyst="Acetic acid",
               external_calibs=ext_calibs
        )

        dataset.subsets[-1].add_byproduct("Thiophene-2-carboxaldehyde")

        two_chkpt = ossc.eval_dataset(dataset)

        return single_chkpt, two_chkpt

    single_step_sim, two_step_sim = _load()
    return single_step_sim, two_step_sim


@app.cell
def _(mo, ossc, outformat, rctstyle, single_step_sim, two_step_sim):
    def _render():
        scaling = {"Benzimine": 2, "Thioimine": 2}

        starting_mats = {
            "Benzimine": ("Aniline", "Benzaldehyde"),
            "Thioimine": ("Aniline", "Thiophene-2-carboxaldehyde"),
        } 

        fig, ax = ossc.render_equconc(single_step_sim, rctstyle, ref_scaling=scaling, starting_mats=starting_mats, legend_loc="center right", highlight=["Acetic acid"])

        if not mo.running_in_notebook():
            fig.savefig(f"graphics/r1-conc-simulated." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md("Figure: Exact concentrations (solid) and predictions by OSSC (dashed) on synthetic data simulating a single-step reaction"))

        fig, ax = ossc.render_equconc(two_step_sim, rctstyle, ref_scaling=scaling, starting_mats=starting_mats, highlight=["Benzaldehyde"], legend_loc="upper right")

        if not mo.running_in_notebook():
            fig.savefig(f"graphics/r2-conc-simulated." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md("Figure: Exact concentrations (solid) and predictions by OSSC (dashed) on synthetic data simulating a two-step reaction"))

    _render()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # External calibrations for reaction R3
    """)
    return


@app.cell
def _(external_calibrations, mo, ossc):
    def _load():
        reactions = ossc.read_campaign("benzimine-formations.h5")

        out = {}

        for handle, exp in mo.status.progress_bar(
            reactions.items(),
            total=len(reactions),
            title="Processing benzimine formations",
        ):
            products = ["Benzimine"]

            ossc.prepare_experiment(exp, 800, 1800)
            solv = ossc.get_solvent(exp)
            ext_calibs = external_calibrations[solv]

            dataset = ossc.Dataset(exp, (800, 1800), products, 
                   n_internal=1,
                   external_calibs=ext_calibs
            )

            chkpts = []
            checkpoint = None

            try:
                for i in range(4):
                    checkpoint = ossc.eval_subset(dataset, i, checkpoint, verbose=True)
                    chkpts.append(checkpoint)

                out[handle] = chkpts
            except:
                mo.output.append(ossc.render_spectra(exp[exp["stage"] == "Reaction"]))

        return out

    r3_chkpts = _load()
    return (r3_chkpts,)


@app.cell
def _():
    from sklearn.linear_model import LinearRegression

    return (LinearRegression,)


@app.cell
def _(
    LinearRegression,
    mo,
    np,
    ossc,
    outformat,
    plt,
    r3_chkpts,
    rctstyle,
    sim,
    solvent_style,
    thio_chkpts,
):
    def _cell():
        bi_yields = {}
        bi_spectra = {}
        ref_yields = {}

        for chem, style in solvent_style.items():
            handle = style[2]
            chkpts = r3_chkpts[handle]

            chkpt = chkpts[3]
            dataset = chkpt.dataset
            solvent = dataset.solvent
            mo.output.append(mo.md(f"### Reaction performed in {solvent}"))

            exp = chkpt.experiment

            # Actual reaction
            scaling = {}
            if exp.attrs.get("simulated", False):
                scaling = {"Benzimine": 2, "Thioimine": 2}

            starting_mats = {
                "Benzimine": ("Aniline", "Benzaldehyde"),
                "Thioimine": ("Aniline", "Thiophene-2-carboxaldehyde"),
            }

            fig, ax = ossc.render_equconc(chkpt, rctstyle, ref_scaling=scaling, starting_mats=starting_mats, exclude=[dataset.solvent])

            peak_wvn = 1700

            # Add the linear regression of the BA carbonyl peak
            ext_calib = dataset.external_calibs["Benzaldehyde"]
            ext_peak_index = np.argmin(np.abs(ext_calib.attrs["ir-x"] - peak_wvn))
            ext_peak_height = np.array([x[ext_peak_index] for x in ext_calib["ir"].to_numpy()])
            ext_peak_height -= ext_peak_height[0]

            ext_conc = ext_calib["Benzaldehyde volume fraction"].to_numpy()

            model = LinearRegression()
            model.fit(np.atleast_2d(ext_peak_height).T, ext_conc)

            peak_index = np.argmin(np.abs(exp.attrs["ir-x"] - peak_wvn))
            peak_height = np.array([x[peak_index] for x in exp["ir"].to_numpy()])
            peak_height -= peak_height[0]

            equ_factors = ossc.get_equ_factors(chkpt, starting_mats={"Benzimine": ["Aniline", "Benzaldehyde"]})
            lin_conc = model.predict(np.atleast_2d(peak_height).T) * equ_factors["Benzaldehyde"][-1]
            ref_yields[handle] = f"{100 * (1 - lin_conc[-1]):0.1f}"

            time = exp["Time since start"].to_numpy()

            ax.plot(time / 3600, lin_conc, color=rctstyle["Benzaldehyde"][0], alpha=0.5, zorder=-10)
            ax.legend(loc="center right")

            if not mo.running_in_notebook():
                fig.savefig(f"graphics/r3-conc-{handle}." + outformat, bbox_inches="tight")

            mo.output.append(fig)
            mo.output.append(mo.md(f"Figure: Concentrations for reaction R3 performed in {solvent}"))

            equ_factors = ossc.get_equ_factors(chkpt, starting_mats=starting_mats)
            conc = {chem: pred * equ_factors[chem] for chem, pred in chkpt.pred_conc.items()}

            bi_yields[handle] = f"{100 * conc["Benzimine"][-1]:0.1f}"
            bi_spectra[handle] = chkpt.metadata["mcd-spectra"]["Benzimine"]

        mo.output.append(mo.md("Table: Yields for reaction R3 in different solvents"))
        data = {
            "Solvent": list(bi_yields.keys()),
            "Yield [%]": list(bi_yields.values()),
            "Yield, linear model [%]": list(ref_yields.values()),
        }
        mo.output.append(mo.ui.table(selection=None, data=data))

        # Stacked plot of all product spectra
        fig, ax = plt.subplots(figsize=(6, 4))
        wvn = np.linspace(800, 1800, 501)

        ax.set_xlim((min(wvn), max(wvn)))
        ax.set_ylabel(r"Absorbance $A$")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        ax.set_xlabel(r"Wavenumber [cm$^{-1}$]")

        legend_pos = "upper right"

        ax.plot(wvn, sim.spectra[sim.SpectraKind.AUTHENTIC]["Benzimine"] * 1.1, color="k", label="Reference", zorder=len(bi_spectra))

        for i_solv, (handle, spectrum) in enumerate(bi_spectra.items()):
            # Subtract the solvent via another MCD
            solv = ossc.get_solvent(thio_chkpts[handle][3].experiment)
            solvent, _ = ossc.get_solvent_spectrum(thio_chkpts[handle][3].experiment)

            #_, spectrum, _ = ossc.mcd(solvent, spectrum)

            color, _, label = solvent_style[solv]
            ax.plot(wvn, spectrum + 0.5*(i_solv + 1), color=color, label=label, zorder=len(bi_spectra) - 1 - i_solv)

        ax.legend(loc="upper left", ncol=3)
        ax.set_ylim(-0.05, 4.3)

        if not mo.running_in_notebook():
            fig.savefig(f"graphics/r3-product-benzimine." + outformat, bbox_inches="tight")

        mo.output.append(fig)
        mo.output.append(mo.md("Figure: Spectra of N-benzylidene aniline extracted from the single-step reaction R3 compared to a reference spectrum"))

    _cell()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gauge spontaneous reaction through simulations
    """)
    return


@app.cell
def _(mo, ossc, sim):
    def _simulations():
        kfs = [0, 570, 1155, 1750, 2365, 2990]

        epsilon_true = []
        delta_yield = []
        kf_meas = []
        epsilon_pred = []
        delta_epsilon = []

        for kf in mo.status.progress_bar(
            kfs,
            total=len(kfs),
            title="Simulating reactions",
        ):
            exp = sim.run_reaction(sim.spectra[sim.SpectraKind.NOOVERLAP], "Isopropanol", ["Aniline", "Thiophene-2-carboxaldehyde", "Thioimine", "Water"], 0.05, 16, has_cat=True, kf_nocat=kf / 10**6, monitoring_duration=10)
            anil_true = exp[exp["stage"] == "Thiophene-2-carboxaldehyde dosage"]["True Aniline volume fraction"].iloc[-1]
            anil_nominal = exp[exp["stage"] == "Thiophene-2-carboxaldehyde dosage"]["Aniline volume fraction"].iloc[-1]
            epsilon_true.append(1 - anil_true / anil_nominal)

            products = ["Thioimine", "Benzimine"]

            ossc.prepare_experiment(exp, 800, 1800)
            solv = ossc.get_solvent(exp)
            ext_calibs = sim.spectra[sim.SpectraKind.NOOVERLAP]

            dataset = ossc.Dataset(exp, (800, 1800), products, 
                   n_internal=1000,
                   catalyst="Acetic acid",
                   external_calibs=ext_calibs
            ) 

            chkpt = ossc.eval_dataset(dataset)

            anil_true = exp["True Aniline volume fraction"].iloc[-1]
            anil_nominal = exp["Aniline volume fraction"].iloc[-1]
            anil_pred = chkpt.pred_conc["Aniline"][-1]
            delta_yield.append((anil_pred - anil_true) / anil_nominal)

            monitoring = ossc.evaluate_monitoring(chkpt)
            kf_measured = monitoring["kf"]
            kf_meas.append(kf_measured)

            sim_exp = sim.run_reaction(sim.spectra[sim.SpectraKind.NOOVERLAP], "Isopropanol", ["Aniline", "Thiophene-2-carboxaldehyde", "Thioimine", "Water"], 0.05, 16, has_cat=True, kf_nocat=kf_measured)

            anil_true = sim_exp[sim_exp["stage"] == "Thiophene-2-carboxaldehyde dosage"]["True Aniline volume fraction"].iloc[-1]
            anil_nominal = sim_exp[sim_exp["stage"] == "Thiophene-2-carboxaldehyde dosage"]["Aniline volume fraction"].iloc[-1]
            epsilon_pred.append(1 - anil_true / anil_nominal)

            delta_epsilon.append(epsilon_true[-1] - epsilon_pred[-1])

        data = {
            "ε (true)": [f"{100*x:0.0f}" for x in epsilon_true],
            "k_f (true)": kfs,
            "Error on yield Δη": [f"{100*x:0.1f}" for x in delta_yield],
            "k_f (measured)": [f"{10**6*x:0.0f}" for x in kf_meas],
            "ε (predicted)": [f"{100*x:0.2f}" for x in epsilon_pred],
            "Δε": [f"{100*x:0.2f}" for x in delta_epsilon],
        }

        mo.output.append(mo.md("Table: Setup values and results for simulated experiments to gauge the error in calibration due to reaction prior to the catalyst addition. ε is the fraction of starting material that is already consumed while the second reactant is added to the reactor. The rate constant k_f refers to the reaction without catalyst. The true k_f is chosen as a parameter for the digital experiment to match the desired ε. To minimize other error sources, the digital experiments use the overlap-free spectra."))
        mo.output.append(mo.ui.table(selection=None, data=data))

    _simulations()
    return


if __name__ == "__main__":
    app.run()
