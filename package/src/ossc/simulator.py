
"""
Simulate experiments.
"""

from collections import defaultdict
import copy
import csv
from enum import Enum
import importlib.resources
import random

import numpy as np
import pandas as pd
import pint

class SpectraKind(Enum):
    AUTHENTIC = 1
    NOOVERLAP = 2
    OVERLAP = 3

class NonProp(Enum):
    NONE = 1
    SOLVENT = 2
    ALL = 3

wavenumbers = np.linspace(800, 1800, 501)

# The chemicals for which there are experimental spectra
chemicals = [
    "Isopropanol"
    "Benzaldehyde"
    "Thiophene-2-carboxaldehyde"
    "Thioimine"
    "Benzimine"
    "Water"
    "Acetic acid"
]

def load_spectra():
    """
    Load or generate the used spectra.
    Executed when the module is loaded.
    """
    path = importlib.resources.files("ossc").joinpath("external-spectra.csv")

    with path.open(newline="", encoding="utf-8") as f:
        # Load the experimental spectra from disk
        authentic = {}
        for row in csv.reader(f):
            chem = row[0]
            authentic[chem] = np.array(row[1:], dtype=float)

    # Generate single-peak spectra for testing
    def lorentzian(nu, nu0, w=10):
        x = (nu - nu0) / (w / 2)
        return 1 / (1 + x**2)

    ov_lorentzians = {}
    noov_lorentzians = {}

    for i_chem, handle in enumerate(authentic.keys()):
        spectrum = lorentzian(wavenumbers, 850 + 100 * i_chem)
        spectrum[(wavenumbers <= 800 + 100 * i_chem) | (wavenumbers >= 800 + 100 * (i_chem + 1))] = 0

        noov_lorentzians[handle] = spectrum

    for i_chem, handle in enumerate(authentic.keys()):
        spectrum = lorentzian(wavenumbers, 850 + 100 * i_chem, w=50)

        ov_lorentzians[handle] = spectrum

    return {
        SpectraKind.AUTHENTIC: authentic,
        SpectraKind.NOOVERLAP: noov_lorentzians,
        SpectraKind.OVERLAP: ov_lorentzians,
    }

spectra = load_spectra()

class DigiExperiment:
    def __init__(self, spectra, reactions=None, nonprop=NonProp.NONE):
        if reactions is None:
            reactions = []

        self.spectra = spectra
        self.reactions = reactions
        self.nonprop = nonprop

        self.snapshots = []
        self.volumes = defaultdict(lambda: 0)
        self.nominal_volumes = defaultdict(lambda: 0)
        self.elapsed = 0
        self.stage = "Solvent"
       
    def add_chemical(self, chem, vol, stirring=30, dosage_delay=None):
        if dosage_delay is None:
            self.volumes[chem] += vol
            self.nominal_volumes[chem] += vol
        else:
            # Simulate a slow addition of the analyte for accurate kinetics.
            n_steps = 10
            for i_step in range(n_steps):
                self.volumes[chem] += vol / n_steps
                self.nominal_volumes[chem] += vol / n_steps
                self.tick(dosage_delay / n_steps)

        if stirring is not None:
            self.tick(stirring)
            self.generate_snapshot() 
        
    def generate_snapshot(self):
        used_vol = sum(self.nominal_volumes.values())
        
        row = {
            "stage": self.stage,
            "Time since start": self.elapsed,
            "Reactor temperature": 30,
            "Stirring rate": 400,
            "Used volume": used_vol,
        }
    
        # Add chemicals and build spectrum
        spectrum = np.zeros_like(wavenumbers)
        
        for i_chem, (chem, vol) in enumerate(self.volumes.items()):
            true_conc = vol / used_vol
            nom_vol = self.nominal_volumes[chem]
            row[f"{chem} volume"] = nom_vol
            row[f"{chem} volume fraction"] = nom_vol / used_vol
            row[f"True {chem} volume fraction"] = true_conc
    
            if self.nonprop == NonProp.ALL \
                or self.nonprop == NonProp.SOLVENT and i_chem == 0:
                effective_conc = -0.5 * true_conc**2 + 1.5 * true_conc            
            else:
                effective_conc = true_conc

            spectrum += self.spectra[chem] * effective_conc

        row["ir"] = spectrum
    
        self.snapshots.append(row)

    def start_stage(self, stage):
        self.stage = stage
        
    def tick(self, time):
        """ Simulate the progress of ongoing reactions over the given time interval. """

        if len(self.reactions) == 0:
            pass
        elif len(self.reactions) == 1:
            self.reactions[0].tick(self, time)
        else:
            # Bisect the time until the concentrations converge
            volumes = copy.deepcopy(self.volumes)

            convergence_threshold = 0.001
            max_depth = 6

            last_volumes = None

            for exponent in range(max_depth):
                self.volumes = copy.deepcopy(volumes)

                time_step = time / 2**exponent

                for i_step in range(2**exponent):
                    for reaction in self.reactions:
                        reaction.tick(self, time_step)

                if last_volumes is None:
                    last_volumes = self.volumes
                else:
                    delta = sum([abs(last_volumes[key] - self.volumes[key]) for key in volumes])
                    last_volumes = self.volumes

                    if delta < convergence_threshold:
                        break
            else:
                #print("Did not converge.")
                pass

        self.elapsed += time

    def get_experiment(self):
        df = pd.DataFrame(self.snapshots)

        ureg = pint.UnitRegistry()
        
        units = {
            "Time since start": ureg.s,
            "Reactor temperature": ureg.celsius,
            "Stirring rate": ureg.rpm,
            "Used volume": ureg.ml,
            "ir": ureg.dimensionless,
            "ir-x": ureg.reciprocal_centimeter,
            "Reactor volume": ureg.ml,
        }

        for chem in self.volumes.keys():
            units[f"{chem} volume"] = ureg.ml
            units[f"{chem} volume fraction"] = ureg.dimensionless
            units[f"True {chem} volume fraction"] = ureg.dimensionless

        units = {key: str(val) for key, val in units.items()}

        metadata = {
            "ir-x": wavenumbers,
            "Reactor volume": 100,
            "units": units,
            "chemicals": list(self.volumes.keys()),
            "simulated": True,
        }

        df.replace(np.nan, 0, inplace=True)
        df.attrs = metadata

        return df

class SecondOrderRevReaction:
    def __init__(self, kf, K, A, B, C, D, cat=None, kf_nocat=0):
        """
        If a catalyst is named, kf_nocat is used if it is not present, and kf otherwise. The catalyst if present if it appears in the volumes dictionary, even with 0 ml.
        """
        self.kf = kf
        self.K = K
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.cat = cat
        self.kf_nocat = kf_nocat

    def tick(self, exp, duration):
        kf = self.kf

        if self.cat and self.cat not in exp.volumes:
            kf = self.kf_nocat

        tot_vol = sum(exp.volumes.values())

        A0 = exp.volumes[self.A] / tot_vol
        B0 = exp.volumes[self.B] / tot_vol
        C0 = exp.volumes[self.C] / tot_vol
        D0 = exp.volumes[self.D] / tot_vol

        if (A0 == 0 or B0 == 0) and (C0 == 0 or D0 == 0) or kf == 0:
            return

        A1, Ainf = second_order_rev_kinetics(duration, kf, kf / self.K, A0, B0, C0, D0)

        exp.volumes[self.A] = A1 * tot_vol
        exp.volumes[self.B] += (A1 - A0) * tot_vol
        exp.volumes[self.C] -= (A1 - A0) * tot_vol
        exp.volumes[self.D] -= (A1 - A0) * tot_vol

def run_reaction(spectra, solvent, chemicals, kf, K, has_cat=True, kf_nocat=0, monitoring_duration=0, reaction_tick=50, n_reaction_tick=100, kf2=None, K2=None, kf3=None, K3=None, reaction_tick2=50, n_reaction_tick2=100, nonprop=NonProp.NONE):
    V_solv = 50
    V_chem = 8
    n_steps = (10, 5)
    n_steps_second = 3
    dosage_delay = 30 # Time the pump takes to deliver a chemical

    cat = None
    if has_cat:
        cat = "Acetic acid"

    if len(chemicals) < 4:
        raise Exception("The implemented reactions require at least four chemicals!")

    reactions = [SecondOrderRevReaction(kf, K, *chemicals[:4], cat=cat, kf_nocat=kf_nocat)]

    if kf2 is not None:
        if len(chemicals) < 6:
            raise Exception("The implemented two-step reaction requires at least six chemicals!")
        # Add the second step reaction
        if K2 is None:
            K2 = K

        reactions.append(SecondOrderRevReaction(kf2, K2, chemicals[2], chemicals[4], chemicals[1], chemicals[5]))

    if kf3 is not None:
        if len(chemicals) < 6:
            raise Exception("The implemented two-step reaction requires at least six chemicals!")
        # Add the second step reaction
        if K3 is None:
            K3 = K

        reactions.append(SecondOrderRevReaction(kf3, K3, chemicals[0], chemicals[4], chemicals[5], chemicals[3]))

    exp = DigiExperiment(spectra, reactions=reactions, nonprop=nonprop)
    exp.start_stage("Solvent")
    exp.add_chemical(solvent, V_solv, stirring=0)

    for chem, n_step in zip(chemicals[:2], n_steps):
        exp.start_stage(f"{chem} dosage")
        for i in range(n_step):
            exp.add_chemical(chem, V_chem / n_step, stirring=20, dosage_delay=dosage_delay)

    if monitoring_duration > 0:
        # Let it react for a bit to gauge how much reacted during the dosage
        exp.start_stage("Monitoring")
        for i in range(monitoring_duration):
            exp.tick(30)
            exp.generate_snapshot()

    if has_cat:
        exp.start_stage("Acetic acid dosage")
        exp.add_chemical(cat, 1e-16, stirring=0)

    exp.start_stage("Reaction")

    for i in range(n_reaction_tick):
        exp.tick(reaction_tick)
        exp.generate_snapshot()

    if kf2 is None:
        return exp.get_experiment()

    # If there are enough chemicals, start the second step
    exp.start_stage(f"{chemicals[-2]} dosage")

    n_step = n_steps_second
    for i in range(n_step):
        exp.add_chemical(chemicals[-2], V_chem / n_step, stirring=30, dosage_delay=dosage_delay)

    exp.start_stage("Reaction2")

    for i in range(n_reaction_tick2):
        exp.tick(reaction_tick2)
        exp.generate_snapshot()

    return exp.get_experiment()

def run_mixture(spectra, solvent, chems, nonprop=NonProp.NONE):
    V_solv = 25
    V_chem = 5
    n_dosage_steps = 5

    exp = DigiExperiment(spectra, nonprop=nonprop)

    exp.start_stage("Solvent")
    exp.add_chemical(solvent, V_solv, stirring=0)

    for chem in chems:
        exp.start_stage(f"{chem} dosage")
        for i in range(n_dosage_steps):
            exp.add_chemical(chem, V_chem / n_dosage_steps)

    exp.start_stage("Monitoring")

    # Dilute to stay within calibration range
    for i in range(10):
        exp.add_chemical(solvent, V_solv / 5)

    # Randomly add so much of each chemical that it matches the calibration range
    total = 2 * V_chem
    steps = {}

    for chem in chems:
        y = np.random.uniform(0, 1, size=5) + 1
        normalized = y / y.sum() * total    

        steps[chem] = normalized

    for i_step in range(5):
        random.shuffle(chems)
        for chem in chems:
            exp.add_chemical(chem, steps[chem][i_step])

    return exp.get_experiment()

def coth(x):
    return np.cosh(x) / np.sinh(x)

def arcoth(x):
    return 0.5 * np.log((x+1) / (x-1))

def second_order_rev_kinetics(t, kf, kr, A0, B0, C0, D0):
    """
    Given the starting concentrations A0-D0, calculate the concentration of A after every time in t. Also returns the equilibrium concentration if this was the only reaction.
    """
    from numpy import sqrt, exp, tanh, arctanh as artanh

    if kr == kf:
        """
        kf == kr: dx/dt = -f x*(x+(B0-A0)) + f (C0 + A0 - x)^2
        """
        Ainf = (A0 + C0)*(A0 + D0) / (A0 + B0 + C0 + D0)
        return Ainf + (A0 - Ainf) * (np.exp(-kf*t*(A0 + B0 + C0 + D0))), Ainf

    sqrtarg = (kf*(A0 - B0) + kr*(C0 - D0))**2 + 2*(2*A0*B0 + A0*C0 + B0*C0 + A0*D0 + B0*D0 + 2*C0*D0)*kf*kr
    sqrtterm = sqrt(sqrtarg)

    ararg = (kf*(A0 + B0) + kr*(C0 + D0))/sqrtterm

    if ararg < 1 and ararg > -1:
        hypterm = tanh(t*sqrtterm / 2 + artanh(ararg))
    else:
        hypterm = coth(t*sqrtterm / 2 + arcoth(ararg))

    trace = (sqrtterm * hypterm + kf * (A0 - B0) - kr*(C0 + D0 + 2*A0))/(2*(kf - kr))

    # Same, just with the hyperbolic term at t -> inf, i.e. of value 1
    Ainf = (sqrtterm + (kf * (A0 - B0) - kr*(C0 + D0 + 2*A0)))/(2*(kf - kr))

    return trace, Ainf

