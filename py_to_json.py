import importlib.util
import json
import sys
import os
import re
from argparse import ArgumentParser
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HELPERSTUFF_DIR = os.path.join(SCRIPT_DIR, "helperstuff")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if HELPERSTUFF_DIR not in sys.path:
    sys.path.insert(0, HELPERSTUFF_DIR)

from observables import observables
from binning import binning
from paths import path

LHSCANS_DIR = os.path.join(SCRIPT_DIR, "LHScans")
JSONS_DIR = os.path.join(SCRIPT_DIR, "jsons")

#SPECIAL_OBS = {"massZ1","massZ2","costhetaZ1","costhetaZ2","costhetastar","phi","phi1"}
SPECIAL_OBS = {}

UNBLIND = True

JET_DISPLAY_WIDTH_VARS = {
    "pTj1",
    "pTj2",
    "mjj",
    "absdetajj",
    "dphijj",
    "pTHj",
    "mHj",
    "pTHjj",
    "TBjmax",
    "TCjmax",
}
JET_FIRST_BIN_FRACTION = 1.0 / 9.0


def get_options():
    parser = ArgumentParser()
    parser.add_argument("variable_name", help="Observable name")
    parser.add_argument("year", help="Data-taking era")
    parser.add_argument(
        "--ZZfloating",
        dest="ZZ",
        action="store_true",
        default=False,
        help="Use the zzfloating version of the observable when available",
    )
    return parser.parse_args()


def canonicalize_variable_name(variable_name):
    variable_name = variable_name.strip()
    if " vs " in variable_name:
        return variable_name.replace(" vs ", "_")
    return variable_name


def resolve_variable_name(variable_name, use_zzfloating):
    variable_name = canonicalize_variable_name(variable_name)
    if not use_zzfloating or "zzfloating" in variable_name:
        return variable_name

    zz_variable = f"{variable_name}_zzfloating"
    expected_scan = os.path.join(LHSCANS_DIR, f"resultsXS_LHScan_expected_{zz_variable}_v3.py")
    observed_scan = os.path.join(LHSCANS_DIR, f"resultsXS_LHScan_observed_{zz_variable}_v3.py")

    if os.path.exists(expected_scan) or os.path.exists(observed_scan):
        return zz_variable

    return variable_name


def get_binning_variable_name(variable_name):
    base_variable = variable_name
    if base_variable.endswith("_zzfloating"):
        base_variable = base_variable[: -len("_zzfloating")]

    if base_variable.count("_") == 1:
        left, right = base_variable.split("_", 1)
        return f"{left} vs {right}"

    return base_variable


def get_base_variable_name(variable_name):
    if variable_name.endswith("_zzfloating"):
        return variable_name[: -len("_zzfloating")]
    return variable_name


def set_jet_display_xlim(variable_name, bins, x_lim):
    if get_base_variable_name(variable_name) not in JET_DISPLAY_WIDTH_VARS:
        return x_lim
    first_bin_high = float(bins[1])
    x_high = float(x_lim[1])
    x_low = (first_bin_high - JET_FIRST_BIN_FRACTION * x_high) / (1.0 - JET_FIRST_BIN_FRACTION)
    return [x_low, x_high]


def load_results(file_path):
    # Dynamically import the Python file to get the 'resultsXS' dictionary
    spec = importlib.util.spec_from_file_location("results_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resultsXS

def _get_key(var, ch, i):
    # v4 has SM_125_<var>_<ch>_genbin<i> for ch in {4l,2e2mu}
    # and SM_125_<var>_genbin<i> is NOT what you showed for v4 special obs
    return f"SM_125_{var}_{ch}_genbin{i}"

def _get_stat_key(key):
    return f"{key}_statOnly"

def _quad(a, b):
    return float(np.sqrt(float(a)**2 + float(b)**2))

def _get_key_v4(var, ch, i):
    return f"SM_125_{var}_{ch}_genbin{i}"

def _get_key_v3_inclusive(var, i):
    return f"SM_125_{var}_genbin{i}"

def _get_zzkey_v3_inclusive(var, i):
    return f"SM_125_{var}_zznorm_genbin{i}"


def _physical_num_bins(variable_name):
    bins, doubleDiff = binning(get_binning_variable_name(variable_name))
    if variable_name in {"mass4l", "mass4l_zzfloating"}:
        return 4 if variable_name == "mass4l_zzfloating" else 4
    return len(bins) if doubleDiff else len(bins) - 1

def _theory_suffix(variable_name, channel):
    if variable_name in SPECIAL_OBS:
        if channel == "4l":
            return ""
        if channel == "2e2mu":
            return "_2e2mu"
        if channel == "4e4mu":
            return "_4e4mu"
    return "" if channel == "4l" else f"_{channel}"


def _collect_zzfloating_results(variable_name, version):
    resultsXS_exp = load_results(os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_expected_{variable_name}_{version}.py'))
    zz_obs_tag = "observed" if UNBLIND else "expected"
    resultsXS_obs = load_results(os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_{zz_obs_tag}_{variable_name}_{version}.py'))

    pattern = re.compile(rf"^SM_125_{re.escape(variable_name)}_zznorm_genbin(\d+)$")
    zz_indices = sorted(
        int(match.group(1))
        for key in resultsXS_exp
        for match in [pattern.match(key)]
        if match is not None
    )

    zznorm_exp, zznorm_up_exp, zznorm_down_exp = [], [], []
    zznorm_stat_up_exp, zznorm_stat_down_exp = [], []
    zznorm_obs, zznorm_up_obs, zznorm_down_obs = [], [], []
    zznorm_stat_up_obs, zznorm_stat_down_obs = [], []

    for i in zz_indices:
        zzkey = _get_zzkey_v3_inclusive(variable_name, i)
        zzstat_key = f"{zzkey}_statOnly"

        if zzkey in resultsXS_exp and zzstat_key in resultsXS_exp:
            zz_central_exp = resultsXS_exp[zzkey]['central']
            zz_up_exp = abs(resultsXS_exp[zzkey]['uncerUp'])
            zz_dn_exp = abs(resultsXS_exp[zzkey]['uncerDn'])
            zz_statUp_exp = abs(resultsXS_exp[zzstat_key]['uncerUp'])
            zz_statDn_exp = abs(resultsXS_exp[zzstat_key]['uncerDn'])
            zznorm_exp.append(round(zz_central_exp, 3))
            zznorm_up_exp.append(round(zz_up_exp, 3))
            zznorm_down_exp.append(round(zz_dn_exp, 3))
            zznorm_stat_up_exp.append(round(zz_statUp_exp, 3))
            zznorm_stat_down_exp.append(round(zz_statDn_exp, 3))

        if zzkey in resultsXS_obs and zzstat_key in resultsXS_obs:
            zz_central_obs = resultsXS_obs[zzkey]['central']
            zz_up_obs = abs(resultsXS_obs[zzkey]['uncerUp'])
            zz_dn_obs = abs(resultsXS_obs[zzkey]['uncerDn'])
            zz_statUp_obs = abs(resultsXS_obs[zzstat_key]['uncerUp'])
            zz_statDn_obs = abs(resultsXS_obs[zzstat_key]['uncerDn'])
            zznorm_obs.append(round(zz_central_obs, 3))
            zznorm_up_obs.append(round(zz_up_obs, 3))
            zznorm_down_obs.append(round(zz_dn_obs, 3))
            zznorm_stat_up_obs.append(round(zz_statUp_obs, 3))
            zznorm_stat_down_obs.append(round(zz_statDn_obs, 3))

    return (
        zznorm_exp,
        zznorm_up_exp,
        zznorm_down_exp,
        zznorm_stat_up_exp,
        zznorm_stat_down_exp,
        zznorm_obs,
        zznorm_up_obs,
        zznorm_down_obs,
        zznorm_stat_up_obs,
        zznorm_stat_down_obs,
    )

def parse_results(channel, variable_name, year):

    # --- Choose scan output model by observable/channel ---
    if variable_name in SPECIAL_OBS:
        # per your rule:
        #   inclusive (4e+4mu+2e2mu) -> v3 4l
        #   2e2mu -> v4 2e2mu
        #   4e+4mu -> v4 4l
        if channel == "4l":
            if (UNBLIND): input_file = os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_observed_{variable_name}_v3.py')
            else: input_file = os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_expected_{variable_name}_v3.py')
        elif channel in ("2e2mu", "4e4mu"):
            if (UNBLIND): input_file = os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_observed_{variable_name}_v4.py')
            else: input_file = os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_expected_{variable_name}_v4.py')
        else:
            raise ValueError(f"Unsupported channel for SPECIAL_OBS: {channel}")

    elif variable_name == "mass4l" or variable_name == "mass4l_zzfloating":
        if (UNBLIND):
            input_file = os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_observed_{variable_name}_v3.py') if channel == "4l" else os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_observed_{variable_name}_v2.py')
        else:
            input_file = os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_expected_{variable_name}_v3.py') if channel == "4l" else os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_expected_{variable_name}_v2.py')

    else:
        if (UNBLIND):
            input_file = os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_observed_{variable_name}_v3.py') if channel == "4l" else os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_observed_{variable_name}_v2.py')
        else:
            input_file = os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_expected_{variable_name}_v3.py') if channel == "4l" else os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_expected_{variable_name}_v2.py')

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)

    print(input_file)
    resultsXS = load_results(input_file)

    # --- Count bins correctly for each format ---
    physical_num_bins = _physical_num_bins(variable_name)

    if variable_name in SPECIAL_OBS:
        if channel == "4l":
            # v3 inclusive: old-style keys
            num_bins = len([k for k in resultsXS
                            if k.startswith(f"SM_125_{variable_name}_genbin")
                            and "_statOnly" not in k])
        else:
            # v4: channelized keys; count using the actual v4 channel you're reading
            v4ch = "2e2mu" if channel == "2e2mu" else "4l"  # 4e4mu uses v4 "4l"
            num_bins = len([k for k in resultsXS
                            if k.startswith(f"SM_125_{variable_name}_{v4ch}_genbin")
                            and "_statOnly" not in k])

    elif (variable_name == "mass4l" or variable_name == "mass4l_zzfloating") and channel != "4l":
        num_bins = 1
    else:
        num_bins = len([k for k in resultsXS if k.startswith(f"SM_125_{variable_name}_genbin") and "_statOnly" not in k])

    if channel == "4l" and "zzfloating" in variable_name and variable_name not in {"mass4l_zzfloating"}:
        if num_bins > physical_num_bins:
            print(
                f"Info: trimming {variable_name} signal bins from {num_bins} to "
                f"{physical_num_bins}; extra bins are ZZ floating POIs."
            )
        num_bins = min(num_bins, physical_num_bins)

    bins, doubleDiff = binning(get_binning_variable_name(variable_name))

    exp_xs, err_up, err_down = [], [], []
    stat_up, stat_down = [], []

    zznorm_exp, zznorm_up_exp, zznorm_down_exp = [], [], []
    zznorm_stat_up_exp, zznorm_stat_down_exp = [], []
    zznorm_obs, zznorm_up_obs, zznorm_down_obs = [], [], []
    zznorm_stat_up_obs, zznorm_stat_down_obs = [], []

    if (variable_name == "mass4l" or variable_name == "mass4l_zzfloating") and channel != "4l":
        # unchanged mass4l special handling
        key = f"SM_125_{variable_name}_{channel}_genbin0"
        stat_key = f"{key}_statOnly"
        if key in resultsXS and stat_key in resultsXS:
            central = resultsXS[key]['central']
            main_up = abs(resultsXS[key]['uncerUp'])
            main_dn = abs(resultsXS[key]['uncerDn'])
            statUp  = abs(resultsXS[stat_key]['uncerUp'])
            statDn  = abs(resultsXS[stat_key]['uncerDn'])

            exp_xs.append(round(central, 3))
            err_up.append(round(main_up, 3))
            err_down.append(round(abs(main_dn), 3))
            stat_up.append(round(statUp, 3))
            stat_down.append(round(abs(statDn), 3))

        if "zzfloating" in variable_name:
            zzkey = f"SM_125_{variable_name}_zznorm{channel}_genbin0"
            zzstat_key = f"{zzkey}_statOnly"
            resultsXS_exp = load_results(os.path.join(LHSCANS_DIR, 'resultsXS_LHScan_expected_mass4l_zzfloating_v2.py'))
            zz_obs_tag = "observed" if UNBLIND else "expected"
            resultsXS_obs = load_results(os.path.join(LHSCANS_DIR, f'resultsXS_LHScan_{zz_obs_tag}_mass4l_zzfloating_v2.py'))
            if zzkey in resultsXS_exp and zzstat_key in resultsXS_exp:
                zz_central_exp = resultsXS_exp[zzkey]['central']
                zz_up_exp = abs(resultsXS_exp[zzkey]['uncerUp'])
                zz_dn_exp = abs(resultsXS_exp[zzkey]['uncerDn'])
                zz_statUp_exp  = abs(resultsXS_exp[zzstat_key]['uncerUp'])
                zz_statDn_exp  = abs(resultsXS_exp[zzstat_key]['uncerDn'])
                zznorm_exp.append(round(zz_central_exp, 3))
                zznorm_up_exp.append(round(zz_up_exp, 3))
                zznorm_down_exp.append(round(zz_dn_exp, 3))
                zznorm_stat_up_exp.append(round(zz_statUp_exp, 3))
                zznorm_stat_down_exp.append(round(zz_statDn_exp, 3))
            if zzkey in resultsXS_obs and zzstat_key in resultsXS_obs:
                zz_central_obs = resultsXS_obs[zzkey]['central']
                zz_up_obs = abs(resultsXS_obs[zzkey]['uncerUp'])
                zz_dn_obs = abs(resultsXS_obs[zzkey]['uncerDn'])
                zz_statUp_obs  = abs(resultsXS_obs[zzstat_key]['uncerUp'])
                zz_statDn_obs  = abs(resultsXS_obs[zzstat_key]['uncerDn'])
                zznorm_obs.append(round(zz_central_obs, 3))
                zznorm_up_obs.append(round(zz_up_obs, 3))
                zznorm_down_obs.append(round(zz_dn_obs, 3))
                zznorm_stat_up_obs.append(round(zz_statUp_obs, 3))
                zznorm_stat_down_obs.append(round(zz_statDn_obs, 3))
        else:
            print(f"Warning: Missing keys for channel {channel} in mass4l")

    else:
        for i in range(num_bins):

            if variable_name in SPECIAL_OBS:
                if channel == "4l":
                    # v3 inclusive
                    key = _get_key_v3_inclusive(variable_name, i)
                elif channel == "2e2mu":
                    # v4 2e2mu
                    key = _get_key_v4(variable_name, "2e2mu", i)
                elif channel == "4e4mu":
                    # v4 4l == 4e+4mu
                    key = _get_key_v4(variable_name, "4l", i)
                else:
                    raise ValueError(f"Unsupported channel for SPECIAL_OBS: {channel}")

                stat_key = f"{key}_statOnly"

                if key in resultsXS and stat_key in resultsXS:
                    central = resultsXS[key]['central']
                    main_up = abs(resultsXS[key]['uncerUp'])
                    main_dn = abs(resultsXS[key]['uncerDn'])
                    statUp  = abs(resultsXS[stat_key]['uncerUp'])
                    statDn  = abs(resultsXS[stat_key]['uncerDn'])

                    exp_xs.append(round(central, 3))
                    err_up.append(round(main_up, 3))
                    err_down.append(round(main_dn, 3))
                    stat_up.append(round(statUp, 3))
                    stat_down.append(round(statDn, 3))
                else:
                    print(f"Warning: Missing keys for {channel}, bin {i}: {key} / {stat_key}")

            else:
                # your existing non-SPECIAL_OBS logic (unchanged)
                if channel in ["4l"]:
                    key = f"SM_125_{variable_name}_genbin{i}"
                    stat_key = f"{key}_statOnly"

                    #if i < num_bins/2:

                    if key in resultsXS and stat_key in resultsXS:
                        central = resultsXS[key]['central']
                        main_up = abs(resultsXS[key]['uncerUp'])
                        main_dn = abs(resultsXS[key]['uncerDn'])
                        statUp  = abs(resultsXS[stat_key]['uncerUp'])
                        statDn  = abs(resultsXS[stat_key]['uncerDn'])
                        exp_xs.append(round(central, 3))
                        err_up.append(round(main_up, 3))
                        err_down.append(round(main_dn, 3))
                        stat_up.append(round(statUp, 3))
                        stat_down.append(round(statDn, 3))          
                    else:
                        print(f"Warning: Missing keys for {channel}, bin {i}: {key} / {stat_key}")

                else:
                    raise ValueError(f"Unsupported channel: {channel}")

        if channel == "4l" and "zzfloating" in variable_name:
            version = "v2" if "mass4l" in variable_name else "v3"
            (
                zznorm_exp,
                zznorm_up_exp,
                zznorm_down_exp,
                zznorm_stat_up_exp,
                zznorm_stat_down_exp,
                zznorm_obs,
                zznorm_up_obs,
                zznorm_down_obs,
                zznorm_stat_up_obs,
                zznorm_stat_down_obs,
            ) = _collect_zzfloating_results(variable_name, version)

    do_log = 1
    x_unit = " (GeV)"
    y_unit = " (fb/GeV)" 
    y_lim_bottom = 10e-5
    y_lim_top = 10
        
    if doubleDiff:
        last_bin_center = -99
        first_bin_center = -99
        x_lim = [-99, -99]
        y_lim_bottom = 10e-6
        y_lim_top = 10e4
    else: 
        if variable_name == "mass4l" or variable_name == "mass4l_zzfloating":
            do_log = 0
            x_lim = [0, 4]
            y_lim_bottom = 0
            y_lim_top = 6
            x_unit = ""
            y_unit = " (fb)"
        elif variable_name == "pT4l" or variable_name == "pT4l_zzfloating":
            x_lim = [0, 200]
            y_lim_bottom = 1e-4
            y_lim_top = 5
        elif variable_name == "rapidity4l" or variable_name == "rapidity4l_zzfloating":
            do_log = 0
            x_unit = ""
            y_unit = " (fb)"
            x_lim = [0, 2.5]
            y_lim_bottom = 0
            y_lim_top = 5
        elif variable_name == "massZ1" or variable_name == "massZ1_zzfloating":
            x_lim = [40, 120]
            y_lim_bottom = 10e-4
            y_lim_top = 100
        elif variable_name == "massZ2" or variable_name == "massZ2_zzfloating":
            x_lim = [0, 65]
            y_lim_bottom = 10e-4
            y_lim_top = 100
        elif variable_name == "Nj" or variable_name == "Nj_zzfloating":
            x_unit = ""
            y_unit = " (fb)"
            x_lim = [0, 5]
            y_lim_bottom = 10e-4
            y_lim_top = 1000
        elif variable_name == "pTj1" or variable_name == "pTj1_zzfloating":
            x_lim = [-30, 240]
        elif variable_name == "pTj2" or variable_name == "pTj2_zzfloating":
            x_lim = [-10, 140]
        elif variable_name == "mjj" or variable_name == "mjj_zzfloating":
            x_lim = [-100, 525]
        elif variable_name == "absdetajj" or variable_name == "absdetajj_zzfloating":
            x_unit = ""
            y_unit = " (fb)"
            x_lim = [-2, 10]
            y_lim_bottom = 10e-4
        elif variable_name == "dphijj" or variable_name == "dphijj_zzfloating":
            x_unit = ""
            y_unit = " (fb)"
            x_lim = [-5, 3.14159]
            y_lim_bottom = 10e-3
        elif variable_name == "mHj" or variable_name == "mHj_zzfloating":
            x_lim = [-110, 880]
        elif variable_name == "pTHj" or variable_name == "pTHj_zzfloating":
            x_lim = [-25, 200]
        elif variable_name == "pTHjj" or variable_name == "pTHjj_zzfloating":
            x_lim = [-25, 100]
        elif variable_name == "costhetastar" or variable_name == "costhetaZ1" or variable_name == "costhetaZ2" or variable_name == "costhetastar_zzfloating" or variable_name == "costhetaZ1_zzfloating" or variable_name == "costhetaZ2_zzfloating":
            x_lim = [-1, 1]
            y_lim_bottom = 10e-3
            y_lim_top = 1000
            x_unit = ""
            y_unit = " (fb)"
        elif variable_name == "phi" or variable_name == "phi1" or variable_name == "phi_zzfloating" or variable_name == "phi1_zzfloating":
            x_lim = [-3.14159, 3.14159]
            y_lim_bottom = 10e-3
            y_lim_top = 1000
            x_unit = ""
            y_unit = " (fb)"
        elif variable_name == "TCjmax" or variable_name == "TCjmax_zzfloating":
            x_lim = [-20, 80]
        elif variable_name == "TBjmax" or variable_name == "TBjmax_zzfloating":
            x_lim = [-20, 80]
        else:
            x_lim = [-1000, 1000]

        x_lim = set_jet_display_xlim(variable_name, bins, x_lim)
        first_bin_center =  x_lim[0] + (bins[1] - x_lim[0])/2
        last_bin_center = bins[-2] + (x_lim[1] - bins[-2])/2

    channel_tag = _theory_suffix(variable_name, channel)

    if (UNBLIND): data = 1
    else: data = 0

    return {
        variable_name: {
            "ggh_xs": f"fidXS_NNLOPS_{variable_name}_ggH{channel_tag}_{year}",
            "vbf_xs": f"fidXS_{variable_name}_VBFH{channel_tag}_{year}",
            "vh_xs": f"fidXS_{variable_name}_VH{channel_tag}_{year}",
            "tth_xs": f"fidXS_{variable_name}_ttH{channel_tag}_{year}",
            "xh_xs": f"fidXS_{variable_name}_xH{channel_tag}_{year}",
            "ggh_powheg_xs": f"fidXS_{variable_name}_ggH{channel_tag}_{year}",
            "exp_xs": exp_xs,
            "err_up": err_up,
            "err_down": err_down,
            "stat_up": stat_up,
            "stat_down": stat_down,
            "zznorm_exp": zznorm_exp,
            "zznorm_up_exp": zznorm_up_exp,
            "zznorm_down_exp": zznorm_down_exp,
            "zznorm_stat_up_exp": zznorm_stat_up_exp,
            "zznorm_stat_down_exp": zznorm_stat_down_exp,
            "zznorm_obs": zznorm_obs,
            "zznorm_up_obs": zznorm_up_obs,
            "zznorm_down_obs": zznorm_down_obs,
            "zznorm_stat_up_obs": zznorm_stat_up_obs,
            "zznorm_stat_down_obs": zznorm_stat_down_obs,
            "output_name": variable_name,
            "is_data": data,
            "last_bin_center": last_bin_center,
            "first_bin_center": first_bin_center,
            "plot_log": do_log,
            "variable": variable_name,
            "x_unit": x_unit,
            "y_unit": y_unit,
            "x_lim": x_lim,
            "y_lim_bottom": y_lim_bottom,
            "y_lim_top": y_lim_top,
            "y_lim": [-1,3],
            "pvalue": -99
        }
    }

def main():
    args = get_options()

    variable_name = resolve_variable_name(args.variable_name, args.ZZ)
    year = args.year

    if variable_name == "mass4l" or variable_name == "mass4l_zzfloating":
        channels = ["4l", "4e", "4mu", "2e2mu"]
    elif variable_name in SPECIAL_OBS:
        channels = ["4l", "2e2mu", "4e4mu"]
    else:
        channels = ["4l"]

    for channel in channels:

        channel_tag = "" if channel == "4l" else f"_{channel}"

        result_dict = parse_results(channel, variable_name, year)
        json_file = os.path.join(JSONS_DIR, f"{variable_name}_results{channel_tag}_{year}.json")

        with open(json_file, "w") as f:
            json.dump(result_dict, f, indent=4)

        print(f"JSON saved to {json_file}")

if __name__ == "__main__":
    main()
