
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pint

from ossc.common import get_equ_factors, get_solvent
from ossc.deconvolution import ZetaDecomp, zeta_space

# Fade ground truths a bit for easier comparison
truth_alpha = 0.5

ureg = pint.UnitRegistry() # To render units to strings

def render_3d(x, z, data, figsize=(6, 4), xlabel=None, zlabel=None, ylabel=None):
    """Plot data in 2D, with line color determined by z."""
    cmap = plt.colormaps["viridis_r"]
    norm = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    fig, ax = plt.subplots(figsize=figsize)
    for z_val, row in zip(z, np.array(data)):
        ax.plot(x, row, color=cmap(norm(z_val)))

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None:
        fig.colorbar(mappable, ax=ax, label=zlabel)
        
    return fig, ax

def render_spectra(df, color_trend="Time since start", figsize=(6, 4)):
    """Render spectra with a viridis_r color gradient based on `color_trend`."""
    zlabel = f"{color_trend} [{ureg.Unit(df.attrs["units"][color_trend]):~#P}]"
    x = np.asarray(df.attrs["ir-x"])
    values = np.asarray(df[color_trend], dtype=float)
    spectra = [np.asarray(spectrum) for spectrum in df["ir"]]

    return render_3d(x, values, spectra, figsize=figsize, xlabel="Wavenumber [cm$^{-1}$]", zlabel=zlabel, ylabel="Absorbance")

def render_true_conc(exp, *chemicals):
    if not exp.attrs.get("simulated", False):
        return None, None
    
    fig, ax = plt.subplots()
    ax.set_xlabel("Time since start [s]")
    ax.set_ylabel("Volume fraction [Vol%]")

    if len(exp) == 0:
        return fig, ax

    x = exp["Time since start"]

    for chem in chemicals:
        y = exp[f"True {chem} volume fraction"]
        ax.plot(x, y, label=chem)

    ax.legend()

    return fig, ax

def visualize_mcd(metadata, wavenumbers, figsize=(6, 4)):
    if metadata is None or not "kind" in metadata:
        return None

    if metadata["kind"] == ZetaDecomp.TOPOUT:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, hspace=0)
        axes = gs.subplots(sharex=True)

        background = metadata["background"]
        mixture = metadata["mixture"]

        axes[0].plot(wavenumbers, background, label="Background")
        axes[1].plot(wavenumbers, mixture, label="Mixture")
        axes[2].plot(wavenumbers, metadata["result"], label=f"$\\zeta={metadata["zeta"]:0.2f}$")

        for ax in axes:
            ax.axvline(x=wavenumbers[metadata["topout-index"]])
            ax.legend(loc="upper right")

        axes[0].set_xlabel("Wavernumber [cm$^{-1}$]")
        fig.supylabel("Absorbance")

        return fig

    elif metadata["kind"] == ZetaDecomp.MCD:
        fig = plt.figure(figsize=figsize)
        outer_gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)

        # Existing upper figure: four shared-x axes
        top_gs = outer_gs[0].subgridspec(3, 1, hspace=0)
        axes = top_gs.subplots(sharex=True)

        background = metadata["background"]
        mixture = metadata["mixture"]

        axes[0].plot(wavenumbers, background, label="Background")
        axes[1].plot(wavenumbers, mixture, label="Mixture")
        axes[2].plot(wavenumbers, metadata["result"], label=f"$\\zeta={metadata["zeta"]:0.2f}$")

        for ax in axes:
            ax.legend(loc="upper right")

        axes[0].set_xlabel("Wavenumber [cm$^{-1}$]")
        axes[1].set_ylabel("Absorbance")

        axes[-1].tick_params(axis="x", labelbottom=True)

        # New lower plot
        zax = fig.add_subplot(outer_gs[1])
        zax.set_xlabel("Background fraction [1]")
        zax.set_ylabel("Features")
        zax.plot(zeta_space, metadata["n-over-zeta"])

        return fig

def render_equconc(chkpt, style, y_ticks=None, ref_scaling={}, starting_mats=[], exclude=[], include_solv=False, highlight=[], legend_loc="center left", figsize=(6, 4), inax=None):
    """
    ref_scaling: Manual scaling factors for the ground truths because byproducts aren't separated out (water).
    starting_mats: Keys are products, values lists of the starting materials. Used to define what one equ is for products.
    exclude: Skip this chemical. The solvent is excluded by default.
    highlight: List of chemicals; if their dosage is a single snapshot (catalyst), it is highlighted with a line; if it is multiple snapshots long, it is highlighted with a backdrop
    """
    exp = chkpt.experiment
    time = exp["Time since start"].to_numpy() / 3600

    if inax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = inax

    ax.set_xlim((min(time), max(time)))
    ax.set_xlabel(r"Time $t$ [h]")
    ax.set_ylabel(r"Concentration $c$ [equ.]")

    # Highlight reaction onsets
    for chem in highlight:
        subexp = exp[exp["stage"] == f"{chem} dosage"]

        if subexp.shape[0] == 0:
            raise Exception(f"Should highlight the dosage of {chem}, but didn't find it at the current checkpoint.")

        if subexp.shape[0] == 1:
            # E.g. the catalyst
            x = exp["Time since start"].iloc[subexp.index[0] - 1]
            ax.axvline(x / 3600, ls="--", color="gray")
        else:
            # E.g. benzaldehyde dosages
            ax.axvspan(
                exp["Time since start"].iloc[subexp.index[0] - 1] / 3600,
                subexp["Time since start"].iloc[-1] / 3600,
                color="gray",
                alpha=0.2,
                zorder=0,
            )

    equ_factors = get_equ_factors(chkpt, starting_mats)

    used_vol = chkpt.experiment["Used volume"].to_numpy()
    solvent = get_solvent(exp)

    for i_chem, (chem, prediction) in enumerate(chkpt.pred_conc.items()):
        if chem in exclude or (not include_solv and chem == solvent):
            continue

        truth = chkpt.dataset.get_true_conc(chem)[:len(chkpt.experiment)]

        equ_factor = equ_factors[chem]
        
        color, line, label = style.get(chem, ("black", "-", chem))
        if exp.attrs.get("simulated", False):
            ax.plot(time, truth * equ_factor * ref_scaling.get(chem, 1), color=color, alpha=truth_alpha)
        ax.plot(time, prediction * equ_factor, label=label, color=color, ls=line)

    if y_ticks is not None:
        ax.set_yticks(np.array(y_ticks))

    ax.legend(loc=legend_loc)

    if inax is None:
        return fig, ax
    else:
        return inax

def render_volconc(time, ref, pred, style, exclude=[]):
    """ Render concentration trends as provided and with a reference. """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    time = time / 3600

    for key, trend in pred.items():
        if key in exclude:
            continue

        color, line, label = style.get(key, ("black", "-", key))

        if ref:
            truth = ref[key]
            ax.plot(time, 100 * truth, color=color, label=None)
        ax.plot(time, 100 * trend, ls=line, color=color, label=label)

    ax.set_xlim((min(time), max(time)))

    legend_pos = "upper right"

    ax.legend(loc=legend_pos)

    ax.set_xlabel(r"Time $t$ [h]")
    ax.set_ylabel(r"Volume fraction $φ$ [%]")

    return fig, ax

def render_stacked_spectra(wavenumber, spectra, style, stack_shift=1, inax=None):
    if inax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        ax = inax

    ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    i_chem = len(spectra) - 1
    for key, spec in spectra.items():
        color, _, label = style.get(key, ("black", "-", key))
        ax.plot(wavenumber, i_chem*stack_shift + spec, color=color, label=label)

        i_chem -= 1

    ax.set_xlim((min(wavenumber), max(wavenumber)))

    legend_pos = "upper right"

    ax.legend(loc=legend_pos)

    ax.set_xlabel(r"Wavenumber [cm$^{-1}$]")
    ax.set_ylabel(r"Absorbance")

    if inax is None:
        return fig, ax
    else:
        return ax

def render_zeta_surface(data):
    surface = data["surface"]
    time = np.array(surface[1]) / 3600
    zeta_axis = surface[0]
    features = np.array(surface[2]).T

    zeta = data["plot"]
    zeta_insta = data["plot-instantaneous"]

    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = ax.pcolormesh(time, zeta_axis, features, shading="auto", cmap="viridis")
    ax.set_xlabel("Time $t$ [h]")
    ax.set_ylabel("Background fraction $ζ$ [1]")
    fig.colorbar(mesh, ax=ax, label="Feature count $n$")

    ax.plot(time, zeta, color="orange", label="Smoothed")
    ax.plot(time, zeta_insta, color="orange", ls="--", label="Single frame")
    ax.legend()

    return fig, ax


def render_conc(chkpt, trends, style, y_ticks=None, highlight=[], legend_loc="center left", figsize=(6, 4)):
    """
    trends: Dict of chemical -> array of the concentrations as determined outside
    highlight: List of chemicals; if their dosage is a single snapshot (catalyst), it is highlighted with a line; if it is multiple snapshots long, it is highlighted with a backdrop
    """
    exp = chkpt.experiment
    time = exp["Time since start"].to_numpy() / 3600

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim((min(time), max(time)))
    ax.set_xlabel(r"Time $t$ [h]")
    ax.set_ylabel(r"Concentration $c$ [equ.]")

    # Highlight reaction onsets
    for chem in highlight:
        subexp = exp[exp["stage"] == f"{chem} dosage"]

        if subexp.shape[0] == 0:
            raise Exception(f"Should highlight the dosage of {chem}, but didn't find it at the current checkpoint.")

        if subexp.shape[0] == 1:
            # E.g. the catalyst
            x = exp["Time since start"].iloc[subexp.index[0] - 1] / 3600
            ax.axvline(x, ls="--", color="gray")
        else:
            # E.g. benzaldehyde dosages
            ax.axvspan(
                exp["Time since start"].iloc[subexp.index[0] - 1] / 3600,
                subexp["Time since start"].iloc[-1] / 3600,
                color="gray",
                alpha=0.2,
                zorder=0,
            )

    for chem, conc in trends.items():
        color, line, label = style.get(chem, ("black", "-", chem))
        
        ax.plot(time, conc, label=label, color=color, ls=line)

    if y_ticks is not None:
        ax.set_yticks(np.array(y_ticks))

    ax.legend(loc=legend_loc)

    return fig, ax
