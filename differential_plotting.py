import importlib
import json
import os
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator, MultipleLocator
from matplotlib.transforms import Bbox

sys.path.append("helperstuff/")
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from fit.createDatacard import get_zzfloating_merged_bin_groups
from paths import path

plt.style.use(hep.style.CMS)
sys.path.append(os.path.join(os.path.dirname(__file__), "fidXS"))

SPECIAL_OBS = {}
SPECIAL_BIN_WIDTH_VARS = {
    "pTj1", "pTj1_zzfloating",
    "pTj2", "pTj2_zzfloating",
    "mjj", "mjj_zzfloating",
    "absdetajj", "absdetajj_zzfloating",
    "dphijj", "dphijj_zzfloating",
    "pTHj", "pTHj_zzfloating",
    "pTHjj", "pTHjj_zzfloating",
    "mHj", "mHj_zzfloating",
    "TCjmax", "TCjmax_zzfloating",
    "TBjmax", "TBjmax_zzfloating",
}
SPECIAL_BIN_WIDTH_BASE_VARS = {v.replace("_zzfloating", "") for v in SPECIAL_BIN_WIDTH_VARS}
JET_FIRST_BIN_FRACTION = 1.0 / 9.0

VAR_LABELS = {
    "mass4l": r"m_{4\ell}",
    "mass4l_zzfloating": r"m_{4\ell}^{\text{ZZ floating}}",
    "pT4l": r"p^T_{4\ell}",
    "rapidity4l": r"|y_{4\ell}|",
    "rapidity4l_zzfloating": r"|y_{4\ell}|^{\text{ZZ floating}}",
    "massZ1": r"m_{Z_1}",
    "massZ2": r"m_{Z_2}",
    "Nj": r"N_{jets}",
    "pTj1": r"p^T_{j_1}",
    "pTj1_zzfloating": r"p_{j_1}^{T \text{ ZZ floating}}",
    "pTj2": r"p^T_{j_2}",
    "mjj": r"m_{jj}",
    "absdetajj": r"|\Delta\eta_{jj}|",
    "dphijj": r"\Delta\Phi_{jj}",
    "pTHj": r"p^T_{Hj}",
    "pTHjj": r"p^T_{Hjj}",
    "mHj": r"m_{Hj}",
    "TCjmax": r"\mathcal{T}_C^{max}",
    "TBjmax": r"\mathcal{T}_B^{max}",
    "rapidity4l_pT4l": r"|y_{H}| p^T_{4\ell}",
    "massZ1_massZ2": r"m_{Z1} m_{Z2}",
    "Nj_pT4l": r"N_{jets} p^T_{4\ell}",
    "pTj1_pTj2": r"p^T_{j1} p^T_{j2}",
    "pTHj_pT4l": r"p^T_{Hj} p^T_{4\ell}",
    "TCjmax_pT4l": r"\mathcal{T}_C^{max} p^T_{4\ell}",
    "pT4l_pTHj": r"p^T_{H} p^T_{Hj}",
    "absdetajj_mjj": r"|\Delta\eta_{jj}| m_{jj}",
    "costhetaZ1": r"cos \theta_{Z_1}",
    "costhetaZ2": r"cos \theta_{Z_2}",
    "costhetastar": r"cos \theta^{*}_{ZZ}",
    "phi": r"\Phi",
    "phi1": r"\Phi_1",
}


def get_var_label(variable):
    label = VAR_LABELS.get(variable)
    if label is not None:
        return label

    if variable.endswith("_zzfloating"):
        base_variable = variable[:-len("_zzfloating")]
        base_label = VAR_LABELS.get(base_variable, base_variable)
        return r"{" + base_label + r"}^{\text{ZZ floating}}"

    return variable


def get_base_variable_name(variable):
    if variable.endswith("_zzfloating"):
        return variable[:-len("_zzfloating")]
    return variable


def set_jet_display_xlim(variable, bins_plot, cfg):
    if get_base_variable_name(variable) not in SPECIAL_BIN_WIDTH_BASE_VARS:
        return
    first_bin_high = float(bins_plot[1])
    x_high = float(cfg["x_lim"][1])
    x_low = (first_bin_high - JET_FIRST_BIN_FRACTION * x_high) / (1.0 - JET_FIRST_BIN_FRACTION)
    cfg["x_lim"] = [x_low, x_high]
    cfg["first_bin_center"] = x_low + 0.5 * (first_bin_high - x_low)
    cfg["last_bin_center"] = float(bins_plot[-2]) + 0.5 * (x_high - float(bins_plot[-2]))


LUMI_BY_ERA = {
    "2022": 7.9804,
    "2022EE": 26.6728,
    "2023preBPix": 17.794,
    "2023postBPix": 9.451,
    "2022full": 34.6,
    "2023full": 27.2,
    "2022_2023": 62,
    "Run3": 171,
}

IMPORT_ATTRS = [
    "fidXS",
    "fidXS_scale_up",
    "fidXS_scale_dn",
    "fidXS_pdf_up",
    "fidXS_pdf_dn",
    "fidXS_alpha_up",
    "fidXS_alpha_dn",
    "Boundaries",
]

DOUBLE_DIFF_CUSTOM_XTICKS = {

    "rapidity4l_pT4l": [
        r"$|y_H| \in [0,0.2)$  $p_T^H \in [0,50)$",
        r"$|y_H| \in [0.2,0.4)$  $p_T^H \in [0,50)$",
        r"$|y_H| \in [0.4,0.65)$  $p_T^H \in [0,50)$",
        r"$|y_H| \in [0.65,0.9)$  $p_T^H \in [0,50)$",
        r"$|y_H| \in [0.9,1.2)$  $p_T^H \in [0,50)$",
        r"$|y_H| \in [1.2,2.5]$  $p_T^H \in [0,50)$",
        r"$|y_H| \in [0,0.5)$  $p_T^H \in [50,105)$",
        r"$|y_H| \in [0.5,1.15)$  $p_T^H \in [50,105)$",
        r"$|y_H| \in [1.15,2.5]$  $p_T^H \in [50,105)$",
        r"$|y_H| \in [0,0.45)$  $p_T^H \in [105,\infty)$",
        r"$|y_H| \in [0.45,1)$  $p_T^H \in [105,\infty)$",
        r"$|y_H| \in [1,2.5]$  $p_T^H \in [105,\infty)$",
    ],

    "massZ1_massZ2": [
        r"$m_{Z1}\in[40,88)$  $m_{Z2}\in[12,28)$",
        r"$m_{Z1}\in[40,88)$  $m_{Z2}\in[28,34)$",
        r"$m_{Z1}\in[40,88)$  $m_{Z2}\in[34,40)$",
        r"$m_{Z1}\in[40,88)$  $m_{Z2}\in[40,65)$",
        r"$m_{Z1}\in[88,120)$  $m_{Z2}\in[12,25)$",
        r"$m_{Z1}\in[88,120)$  $m_{Z2}\in[25,28)$",
        r"$m_{Z1}\in[88,120)$  $m_{Z2}\in[28,65)$",
    ],

    "pTj1_pTj2": [
        r"$0j$",
        r"$1j$",
        r"$p_T^{j1}\in[30,80)$  $p_T^{j2}\in[30,50)$",
        r"$p_T^{j1}\in[80,\infty)$  $p_T^{j2}\in[30,50)$",
        r"$p_T^{j1}\in[30,\infty)$  $p_T^{j2}\in[50,\infty)$",
    ],

    "Nj_pT4l": [
        r"$N_{jets}=0$  $p_T^H\in[0,12)$",
        r"$N_{jets}=0$  $p_T^H\in[12,18)$",
        r"$N_{jets}=0$  $p_T^H\in[18,24)$",
        r"$N_{jets}=0$  $p_T^H\in[24,32)$",
        r"$N_{jets}=0$  $p_T^H\in[32,46)$",
        r"$N_{jets}=0$  $p_T^H\in[46,\infty)$",
        r"$N_{jets}=1$  $p_T^H\in[0,45)$",
        r"$N_{jets}=1$  $p_T^H\in[45,65)$",
        r"$N_{jets}=1$  $p_T^H\in[65,100)$",
        r"$N_{jets}=1$  $p_T^H\in[100,\infty)$",
        r"$N_{jets} \geq 2$  $p_T^H\in[0,155)$",
        r"$N_{jets} \geq 2$  $p_T^H\in[155,\infty)$",
    ],

    "pT4l_pTHj": [
        r"$p_T^H\in[0,\infty)$  $p_T^{Hj}\in[-100,0)$",
        r"$p_T^H\in[0,45)$  $p_T^{Hj}\in[0,50)$",
        r"$p_T^H\in[45,70)$  $p_T^{Hj}\in[0,50)$",
        r"$p_T^H\in[70,95)$  $p_T^{Hj}\in[0,50)$",
        r"$p_T^H\in[95,\infty)$  $p_T^{Hj}\in[0,50)$",
        r"$p_T^H\in[0,175)$  $p_T^{Hj}\in[50,\infty)$",
        r"$p_T^H\in[175,\infty)$  $p_T^{Hj}\in[50,\infty)$",
    ],

    "TCjmax_pT4l": [
        r"$\mathcal{T}_C^{max}\in[-100,0)$  $p_T^H\in[0,\infty)$",
        r"$\mathcal{T}_C^{max}\in[0,30)$  $p_T^H\in[0,35)$",
        r"$\mathcal{T}_C^{max}\in[0,30)$  $p_T^H\in[35,55)$",
        r"$\mathcal{T}_C^{max}\in[0,30)$  $p_T^H\in[55,70)$",
        r"$\mathcal{T}_C^{max}\in[0,30)$  $p_T^H\in[70,90)$",
        r"$\mathcal{T}_C^{max}\in[0,30)$  $p_T^H\in[90,125)$",
        r"$\mathcal{T}_C^{max}\in[0,30)$  $p_T^H\in[125,\infty)$",
        r"$\mathcal{T}_C^{max}\in[30,\infty)$  $p_T^H\in[0,250)$",
        r"$\mathcal{T}_C^{max}\in[30,\infty)$  $p_T^H\in[250,\infty)$",
    ],

    "absdetajj_mjj": [
        r"$0/1j$",
        r"$|\Delta\eta_{jj}|\in[0,3)$  $m_{jj}\in[0,\infty)$",
        r"$|\Delta\eta_{jj}|\in[3,10)$  $m_{jj}\in[0,450)$",
        r"$|\Delta\eta_{jj}|\in[3,10)$  $m_{jj}\in[450,\infty)$",
    ],
}

PVALUE_MAP = {
    #"mass4l": 0.97, #FINAL
    "mass4l": 0.84, # WITH K1 K2
    "pT4l": 0.87, #FINAL
    "rapidity4l": 0.72, #FINAL
    "massZ1": 0.87, #FINAL
    "massZ2": 0.83, #FINAL
    "costhetaZ1": 0.84, #FINAL
    "costhetaZ2": 0.28, #FINAL
    "costhetastar": 0.84, #FINAL
    "phi": 0.15, #FINAL
    "phi1": 0.4, #FINAL
    "pTj1": 0.25, #FINAL
    "pTj2": 0.99, #FINAL
    "Nj": 0.99, #FINAL
    "mjj": 0.99, #FINAL
    "absdetajj": 0.81, #FINAL
    "dphijj": 0.98, #FINAL
    "mHj": 0.42, #FINAL
    "pTHj": 0.32, #FINAL
    "pTHjj": 0.79, #FINAL
    "TCjmax": 0.86, #FINAL
    "TBjmax": 0.79, #FINAL
    "rapidity4l_pT4l": 0.33, #FINAL
    "pT4l_pTHj": 0.15, #FINAL
    "massZ1_massZ2": 0.63, #FINAL
    "pTj1_pTj2": 0.89, #FINAL
    "absdetajj_mjj": 0.72, #FINAL
    "Nj_pT4l": 0.92, #FINAL
    "TCjmax_pT4l": 0.92, #FINAL
    
    "mass4l_zzfloating": 0.97,
    "pT4l_zzfloating": 0.77,
    "rapidity4l_zzfloating": 0.86,
    "massZ1_zzfloating": 0.71,
    "massZ2_zzfloating": 0.87,
    "costhetaZ1_zzfloating": 0.85,
    "costhetaZ2_zzfloating": 0.29,
    "costhetastar_zzfloating": 0.83,
    "phi_zzfloating": 0.17,
    "phi1_zzfloating": 0.43,
    "pTj1_zzfloating": 0.06,
    "pTj2_zzfloating": 0.68,
    "Nj_zzfloating": 0.47,
    "mjj_zzfloating": 0.99,
    "absdetajj_zzfloating": 0.58,
    "dphijj_zzfloating": 0.95,
    "mHj_zzfloating": 0.25,
    "pTHj_zzfloating": 0.45,
    "pTHjj_zzfloating": 0.55,
    "TCjmax_zzfloating": 0.81,
    "TBjmax_zzfloating": 0.66,
    "rapidity4l_pT4l_zzfloating": 0.3,
    "pT4l_pTHj_zzfloating": 0.05,
    "massZ1_massZ2_zzfloating": 0.52,
    "pTj1_pTj2_zzfloating": 0.56,
    "absdetajj_mjj_zzfloating": 0.97,
    "Nj_pT4l_zzfloating": 0.96,
    "TCjmax_pT4l_zzfloating": 0.83,
}

def reorder_legend(handles, labels):
    desired_order = [
        r"Data (stat $\oplus$ sys unc.)",
        "Systematic Uncertainty",
        "VBF fit (non-VBF fixed)",
        "VBF fit (all extrapolated fixed)",
        "ggH (NNLOPS + JHUGen + Pythia) + xH",
        "ggH (POWHEG + JHUGen + Pythia) + xH",
        "xH = ttH + VH + VBF (POWHEG + JHUGen + Pythia)",
        "Fitted ZZ / Predicted ZZ",
    ]
    order_map = {label: i for i, label in enumerate(desired_order)}
    pairs = sorted(zip(handles, labels), key=lambda x: order_map.get(x[1], 999))
    return zip(*pairs)

def get_options():
    parser = ArgumentParser()
    parser.add_argument(
        "--variables",
        dest="variables",
        required=True,
        default="",
        help="Comma-separated variables or 'all'",
    )
    parser.add_argument(
        "--config-json",
        dest="config_json",
        required=False,
        default="./config_unblinded.json",
        help="Path to config JSON",
    )
    parser.add_argument(
        "--no-preliminary",
        dest="no_preliminary",
        required=False,
        default=False,
        action="store_true",
        help="Disable CMS Preliminary label",
    )
    parser.add_argument(
        "--year",
        dest="YEAR",
        required=True,
        default="",
        help="Data-taking era",
    )
    parser.add_argument(
        "--ZZfloating",
        dest="ZZ",
        required=False,
        default=False,
        action="store_true",
        help="Use the zzfloating version of each requested observable when available",
    )
    return parser.parse_args()


def canonicalize_variable_name(variable):
    variable = variable.strip()
    if " vs " in variable:
        return variable.replace(" vs ", "_")
    return variable


def resolve_plot_variable(variable, config, use_zzfloating, era):
    variable = canonicalize_variable_name(variable)
    if not use_zzfloating or "zzfloating" in variable:
        return variable

    zz_variable = f"{variable}_zzfloating"
    if zz_variable in config:
        return zz_variable
    if os.path.exists(_default_result_json_path(zz_variable, era)):
        return zz_variable

    return variable


def resolve_plot_variable_list(raw_variables, config, use_zzfloating, era):
    variable_list = [canonicalize_variable_name(v) for v in raw_variables.split(",") if v.strip()]
    if "all" in variable_list:
        variable_list = list(config.keys())
        if use_zzfloating:
            base_only = []
            zz_vars = []
            for variable in variable_list:
                if variable.endswith("_zzfloating"):
                    zz_vars.append(variable)
                elif not os.path.exists(_default_result_json_path(f"{variable}_zzfloating", era)) and f"{variable}_zzfloating" not in config:
                    base_only.append(variable)
            return zz_vars + base_only
        return variable_list

    resolved = []
    seen = set()
    for variable in variable_list:
        current = resolve_plot_variable(variable, config, use_zzfloating, era)
        if current not in seen:
            resolved.append(current)
            seen.add(current)
    return resolved


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def load_var_config(json_path, variable):
    js = load_json(json_path)
    if variable not in js:
        raise KeyError(f"'{variable}' key not found in {json_path}")
    return js[variable]


def _mass4l_json_paths(config_json_path, era):
    d = os.path.dirname(config_json_path)
    base = os.path.basename(config_json_path)
    suffix = f"_{era}.json"

    if not base.endswith(suffix):
        stem, ext = os.path.splitext(base)
        return [
            os.path.join(d, base),
            os.path.join(d, f"{stem}_2e2mu{ext}"),
            os.path.join(d, f"{stem}_4mu{ext}"),
            os.path.join(d, f"{stem}_4e{ext}"),
        ]

    stem = base[:-len(suffix)]
    return [
        os.path.join(d, base),
        os.path.join(d, f"{stem}_2e2mu{suffix}"),
        os.path.join(d, f"{stem}_4mu{suffix}"),
        os.path.join(d, f"{stem}_4e{suffix}"),
    ]


def load_mass4l_final_states(var, inclusive_mass4l_json_path, era):
    paths = _mass4l_json_paths(inclusive_mass4l_json_path, era)
    final_state_labels = ["4l", "2e2mu", "4mu", "4e"]

    per_fs = []
    for p in paths:
        js = load_json(p)
        if "mass4l" not in js and "mass4l_zzfloating" not in js:
            raise KeyError(f"'mass4l' or 'mass4l_zzfloating' key not found in {p}")
        per_fs.append(js[var])

    def one(x):
        return float(x[0])

    combined = dict(per_fs[0])
    combined["exp_xs"] = [one(cfg["exp_xs"]) for cfg in per_fs]
    combined["err_up"] = [one(cfg["err_up"]) for cfg in per_fs]
    combined["err_down"] = [one(cfg["err_down"]) for cfg in per_fs]
    combined["stat_up"] = [one(cfg["stat_up"]) for cfg in per_fs]
    combined["stat_down"] = [one(cfg["stat_down"]) for cfg in per_fs]

    if var == "mass4l_zzfloating":
        for key in [
            "zznorm_exp",
            "zznorm_up_exp",
            "zznorm_down_exp",
            "zznorm_stat_up_exp",
            "zznorm_stat_down_exp",
            "zznorm_obs",
            "zznorm_up_obs",
            "zznorm_down_obs",
            "zznorm_stat_up_obs",
            "zznorm_stat_down_obs",
        ]:
            combined[key] = [one(cfg[key]) for cfg in per_fs]

    combined["_per_fs_configs"] = per_fs
    combined["_final_state_labels"] = final_state_labels

    bins_plot = np.arange(0, 5, 1, dtype=float)
    bins_c = 0.5 * (bins_plot[1:] + bins_plot[:-1])
    bin_w = np.ones(4, dtype=float)

    return combined, bins_plot, bins_c, bin_w


def _specialobs_json_paths(variable, era):
    return [
        f"jsons/{variable}_results_{era}.json",
        f"jsons/{variable}_results_2e2mu_{era}.json",
        f"jsons/{variable}_results_4e4mu_{era}.json",
    ]


def _default_result_json_path(variable, era):
    return os.path.join(os.path.dirname(__file__), "jsons", f"{variable}_results_{era}.json")


def _normalize_plot_cfg(cfg, variable):
    cfg = dict(cfg)
    cfg["variable"] = variable
    cfg["output_name"] = variable
    return cfg


def import_xs_module(module_name):
    return importlib.import_module(module_name)


def is_double_diff(cfg):
    return cfg["first_bin_center"] == -99 and cfg["last_bin_center"] == -99


def get_bins(variable, cfg):
    ggh_xs = import_xs_module(cfg["ggh_xs"])
    if is_double_diff(cfg):
        bin_number = len(ggh_xs.Boundaries)
        bins_plot = np.array([25 * i for i in range(bin_number + 1)], dtype=float)
        bins_c = 0.5 * (bins_plot[1:] + bins_plot[:-1])
        return bins_plot, bins_c, np.diff(bins_plot), bin_number

    bins_plot = np.array(ggh_xs.Boundaries, dtype=float)
    set_jet_display_xlim(variable, bins_plot, cfg)
    bins_plot[0] = cfg["x_lim"][0]
    bins_plot[-1] = cfg["x_lim"][-1]
    bins_c = 0.5 * (bins_plot[1:] + bins_plot[:-1])
    bins_c[0] = cfg["first_bin_center"]
    bins_c[-1] = cfg["last_bin_center"]
    return bins_plot, bins_c, np.diff(bins_plot), None


def import_mode_array(per_fs_configs, mode_key):
    out = []
    for cfg in per_fs_configs:
        mod = import_xs_module(cfg[mode_key])
        out.append(float(mod.fidXS[0]))
    return np.array(out, dtype=float)


def import_unc_array(per_fs_configs, mode_key, source_key):
    out = []
    for cfg in per_fs_configs:
        mod = import_xs_module(cfg[mode_key])
        out.append(abs(float(getattr(mod, source_key)[0]) - float(mod.fidXS[0])))
    return np.array(out, dtype=float)


def build_theory_mass4l(cfg, bin_w):
    per_fs = cfg["_per_fs_configs"]

    xs = {
        "ggh": import_mode_array(per_fs, "ggh_xs"),
        "vbf": import_mode_array(per_fs, "vbf_xs"),
        "vh": import_mode_array(per_fs, "vh_xs"),
        "tth": import_mode_array(per_fs, "tth_xs"),
    }

    ggh_powheg_vals = import_mode_array(per_fs, "ggh_powheg_xs")

    ggh_xs_norm = xs["ggh"] / bin_w
    vbf_xs_norm = xs["vbf"] / bin_w
    vh_xs_norm = xs["vh"] / bin_w
    tth_xs_norm = xs["tth"] / bin_w
    ggh_powheg_xs_norm = ggh_powheg_vals / bin_w
    xh_xs_norm = vbf_xs_norm + vh_xs_norm + tth_xs_norm

    sources_up = ["fidXS_scale_up", "fidXS_pdf_up", "fidXS_alpha_up"]
    sources_dn = ["fidXS_scale_dn", "fidXS_pdf_dn", "fidXS_alpha_dn"]

    ggh_up = np.stack([import_unc_array(per_fs, "ggh_xs", s) for s in sources_up], axis=0)
    vbf_up = np.stack([import_unc_array(per_fs, "vbf_xs", s) for s in sources_up], axis=0)
    vh_up = np.stack([import_unc_array(per_fs, "vh_xs", s) for s in sources_up], axis=0)
    tth_up = np.stack([import_unc_array(per_fs, "tth_xs", s) for s in sources_up], axis=0)

    ggh_dn = np.stack([import_unc_array(per_fs, "ggh_xs", s) for s in sources_dn], axis=0)
    vbf_dn = np.stack([import_unc_array(per_fs, "vbf_xs", s) for s in sources_dn], axis=0)
    vh_dn = np.stack([import_unc_array(per_fs, "vh_xs", s) for s in sources_dn], axis=0)
    tth_dn = np.stack([import_unc_array(per_fs, "tth_xs", s) for s in sources_dn], axis=0)

    ggh_powheg_up = np.stack([import_unc_array(per_fs, "ggh_powheg_xs", s) for s in sources_up], axis=0)
    ggh_powheg_dn = np.stack([import_unc_array(per_fs, "ggh_powheg_xs", s) for s in sources_dn], axis=0)

    madgraph_up = ggh_up + vbf_up + vh_up + tth_up
    madgraph_dn = ggh_dn + vbf_dn + vh_dn + tth_dn
    powheg_up = ggh_powheg_up + vbf_up + vh_up + tth_up
    powheg_dn = ggh_powheg_dn + vbf_dn + vh_dn + tth_dn

    total_xs = xs["ggh"] + xs["vbf"] + xs["vh"] + xs["tth"]

    unc_th_up = (
        np.sqrt((np.sqrt(np.sum(madgraph_up**2, axis=0)) / total_xs) ** 2 + 0.02**2) * total_xs
    ) / bin_w
    unc_th_dn = (
        np.sqrt((np.sqrt(np.sum(madgraph_dn**2, axis=0)) / total_xs) ** 2 + 0.02**2) * total_xs
    ) / bin_w

    unc_th_powheg_up = (
        np.sqrt((np.sqrt(np.sum(powheg_up**2, axis=0)) / total_xs) ** 2 + 0.02**2) * total_xs
    ) / bin_w
    unc_th_powheg_dn = (
        np.sqrt((np.sqrt(np.sum(powheg_dn**2, axis=0)) / total_xs) ** 2 + 0.02**2) * total_xs
    ) / bin_w

    return {
        "xs": xs,
        "ggh_xs_norm": ggh_xs_norm,
        "ggh_powheg_xs_norm": ggh_powheg_xs_norm,
        "xh_xs_norm": xh_xs_norm,
        "unc_th_up": unc_th_up,
        "unc_th_dn": unc_th_dn,
        "unc_th_powheg_up": unc_th_powheg_up,
        "unc_th_powheg_dn": unc_th_powheg_dn,
    }


def build_theory_standard(variable, cfg, bin_w):
    xs = {}

    ggh_xs = import_xs_module(cfg["ggh_xs"])
    vbf_xs = import_xs_module(cfg["vbf_xs"])
    vh_xs = import_xs_module(cfg["vh_xs"])
    tth_xs = import_xs_module(cfg["tth_xs"])
    ggh_powheg_xs = import_xs_module(cfg["ggh_powheg_xs"])

    xs["ggh"] = np.array(ggh_xs.fidXS, dtype=float)
    xs["vbf"] = np.array(vbf_xs.fidXS, dtype=float)
    xs["vh"] = np.array(vh_xs.fidXS, dtype=float)
    xs["tth"] = np.array(tth_xs.fidXS, dtype=float)

    ggh_xs_norm = xs["ggh"] / bin_w
    vbf_xs_norm = xs["vbf"] / bin_w
    vh_xs_norm = xs["vh"] / bin_w
    tth_xs_norm = xs["tth"] / bin_w
    ggh_powheg_xs_norm = np.array(ggh_powheg_xs.fidXS, dtype=float) / bin_w

    if variable in SPECIAL_BIN_WIDTH_VARS:
        ggh_xs_norm[0] *= bin_w[0]
        vbf_xs_norm[0] *= bin_w[0]
        vh_xs_norm[0] *= bin_w[0]
        tth_xs_norm[0] *= bin_w[0]
        ggh_powheg_xs_norm[0] *= bin_w[0]

    xh_xs_norm = vbf_xs_norm + vh_xs_norm + tth_xs_norm

    sources_up = ["fidXS_scale_up", "fidXS_pdf_up", "fidXS_alpha_up"]
    sources_dn = ["fidXS_scale_dn", "fidXS_pdf_dn", "fidXS_alpha_dn"]

    ggh_up = [abs(np.array(getattr(ggh_xs, s)) - np.array(ggh_xs.fidXS)) for s in sources_up]
    ggh_powheg_up = [abs(np.array(getattr(ggh_powheg_xs, s)) - np.array(ggh_powheg_xs.fidXS)) for s in sources_up]
    vbf_up = [abs(np.array(getattr(vbf_xs, s)) - np.array(vbf_xs.fidXS)) for s in sources_up]
    vh_up = [abs(np.array(getattr(vh_xs, s)) - np.array(vh_xs.fidXS)) for s in sources_up]
    tth_up = [abs(np.array(getattr(tth_xs, s)) - np.array(tth_xs.fidXS)) for s in sources_up]

    ggh_dn = [abs(np.array(getattr(ggh_xs, s)) - np.array(ggh_xs.fidXS)) for s in sources_dn]
    ggh_powheg_dn = [abs(np.array(getattr(ggh_powheg_xs, s)) - np.array(ggh_powheg_xs.fidXS)) for s in sources_dn]
    vbf_dn = [abs(np.array(getattr(vbf_xs, s)) - np.array(vbf_xs.fidXS)) for s in sources_dn]
    vh_dn = [abs(np.array(getattr(vh_xs, s)) - np.array(vh_xs.fidXS)) for s in sources_dn]
    tth_dn = [abs(np.array(getattr(tth_xs, s)) - np.array(tth_xs.fidXS)) for s in sources_dn]

    madgraph_up = np.sum([ggh_up, vbf_up, vh_up, tth_up], axis=0)
    powheg_up = np.sum([ggh_powheg_up, vbf_up, vh_up, tth_up], axis=0)
    madgraph_dn = np.sum([ggh_dn, vbf_dn, vh_dn, tth_dn], axis=0)
    powheg_dn = np.sum([ggh_powheg_dn, vbf_dn, vh_dn, tth_dn], axis=0)

    total_xs = xs["ggh"] + xs["vbf"] + xs["vh"] + xs["tth"]

    unc_th_up = (
        np.sqrt((np.sqrt(np.sum(np.square(madgraph_up), axis=0)) / total_xs) ** 2 + 0.02**2) * total_xs
    ) / bin_w
    unc_th_dn = (
        np.sqrt((np.sqrt(np.sum(np.square(madgraph_dn), axis=0)) / total_xs) ** 2 + 0.02**2) * total_xs
    ) / bin_w
    unc_th_powheg_up = (
        np.sqrt((np.sqrt(np.sum(np.square(powheg_up), axis=0)) / total_xs) ** 2 + 0.02**2) * total_xs
    ) / bin_w
    unc_th_powheg_dn = (
        np.sqrt((np.sqrt(np.sum(np.square(powheg_dn), axis=0)) / total_xs) ** 2 + 0.02**2) * total_xs
    ) / bin_w

    if variable in SPECIAL_BIN_WIDTH_VARS:
        unc_th_up[0] *= bin_w[0]
        unc_th_dn[0] *= bin_w[0]
        unc_th_powheg_up[0] *= bin_w[0]
        unc_th_powheg_dn[0] *= bin_w[0]

    return {
        "xs": xs,
        "ggh_xs_norm": ggh_xs_norm,
        "ggh_powheg_xs_norm": ggh_powheg_xs_norm,
        "xh_xs_norm": xh_xs_norm,
        "unc_th_up": unc_th_up,
        "unc_th_dn": unc_th_dn,
        "unc_th_powheg_up": unc_th_powheg_up,
        "unc_th_powheg_dn": unc_th_powheg_dn,
    }


def build_measurement(variable, cfg, bin_w):
    exp_xs_raw = np.array(cfg["exp_xs"], dtype=float)
    err_up_raw = np.array(cfg["err_up"], dtype=float)
    err_down_raw = np.array(cfg["err_down"], dtype=float)
    stat_up_raw = np.array(cfg["stat_up"], dtype=float)
    stat_down_raw = np.array(cfg["stat_down"], dtype=float)

    if len(exp_xs_raw) != len(bin_w):
        if "zzfloating" in variable and len(exp_xs_raw) > len(bin_w):
            print(
                f"Info: trimming duplicated ZZ POIs from measurement arrays for {variable}: "
                f"{len(exp_xs_raw)} -> {len(bin_w)}"
            )
            exp_xs_raw = exp_xs_raw[:len(bin_w)]
            err_up_raw = err_up_raw[:len(bin_w)]
            err_down_raw = err_down_raw[:len(bin_w)]
            stat_up_raw = stat_up_raw[:len(bin_w)]
            stat_down_raw = stat_down_raw[:len(bin_w)]
        else:
            raise ValueError(
                f"Measurement array length mismatch for {variable}: "
                f"{len(exp_xs_raw)} values for {len(bin_w)} bins."
            )

    exp_xs = exp_xs_raw / bin_w
    err_up = err_up_raw / bin_w
    err_down = err_down_raw / bin_w
    stat_up = stat_up_raw / bin_w
    stat_down = stat_down_raw / bin_w

    sys_up = np.sqrt(np.maximum(0.0, err_up**2 - stat_up**2))
    sys_down = np.sqrt(np.maximum(0.0, err_down**2 - stat_down**2))

    if variable in SPECIAL_BIN_WIDTH_VARS:
        exp_xs[0] = cfg["exp_xs"][0]

    return {
        "exp_xs": exp_xs,
        "err_up": err_up,
        "err_down": err_down,
        "stat_up": stat_up,
        "stat_down": stat_down,
        "sys_up": sys_up,
        "sys_down": sys_down,
    }


def create_plot_jobs(variable, config, args):
    if variable in {"mass4l", "mass4l_zzfloating"}:
        current_config, bins_plot, bins_c, bin_w = load_mass4l_final_states(variable, args.config_json, args.YEAR)
        current_config = _normalize_plot_cfg(current_config, variable)
        return [("mass4l", current_config, "", bins_plot, bins_c, bin_w)]

    if variable in SPECIAL_OBS:
        jobs = []
        for p, lab in zip(_specialobs_json_paths(variable, args.YEAR), ["4l", "2e2mu", "4e4mu"]):
            cfg = _normalize_plot_cfg(load_var_config(p, variable), variable)
            bins_plot, bins_c, bin_w, _ = get_bins(variable, cfg)
            jobs.append((lab, cfg, f"_{lab}", bins_plot, bins_c, bin_w))
        return jobs

    result_json_path = _default_result_json_path(variable, args.YEAR)
    if os.path.exists(result_json_path):
        cfg = load_var_config(result_json_path, variable)
    else:
        cfg = config[variable]
    cfg = _normalize_plot_cfg(cfg, variable)
    bins_plot, bins_c, bin_w, _ = get_bins(variable, cfg)
    return [("4l", cfg, "", bins_plot, bins_c, bin_w)]


def setup_figure(variable):
    fig = plt.figure(figsize=(10, 8), dpi=120)
    if "zzfloating" in variable:
        frame1 = fig.add_axes((0.1, 0.41, 0.8, 0.53))
        frame2 = fig.add_axes((0.1, 0.22, 0.8, 0.16))
        frame3 = fig.add_axes((0.1, 0.03, 0.8, 0.16))
        frame1.set_xticklabels([])
        frame2.set_xticklabels([])
    else:
        frame1 = fig.add_axes((0.1, 0.35, 0.8, 0.60))
        frame2 = fig.add_axes((0.1, 0.05, 0.8, 0.25))
        frame3 = None
        frame1.set_xticklabels([])
    return fig, frame1, frame2, frame3


def apply_cms_label(cfg, args):
    cms_label = "" if args.no_preliminary else "Preliminary"
    lumi = round(LUMI_BY_ERA[args.YEAR], 2)
    hep.cms.label(cms_label, data=cfg["is_data"], lumi=lumi, fontsize=20, com=13.6)


def plot_theory_main(ax, bins_plot, bins_c, bin_w, theory):
    plt.sca(ax)

    total_nom = theory["ggh_xs_norm"] + theory["xh_xs_norm"]
    total_powheg = theory["ggh_powheg_xs_norm"] + theory["xh_xs_norm"]

    plt.stairs(total_nom, bins_plot, linewidth=2, label="ggH (NNLOPS + JHUGen + Pythia) + xH", color="purple")
    plt.stairs(total_powheg, bins_plot, linewidth=2, label="ggH (POWHEG + JHUGen + Pythia) + xH", color="tab:blue")
    plt.stairs(theory["xh_xs_norm"], bins_plot, linewidth=2, color="green")
    plt.stairs(
        theory["xh_xs_norm"],
        bins_plot,
        linewidth=2,
        label="xH = ttH + VH + VBF (POWHEG + JHUGen + Pythia)",
        alpha=0.2,
        color="green",
        fill=True,
    )

    plt.rcParams["hatch.linewidth"] = 2

    for center, value, err_low, err_high, width in zip(
        bins_c, total_nom, theory["unc_th_dn"], theory["unc_th_up"], bin_w
    ):
        plt.gca().add_patch(
            plt.Rectangle(
                (center - width / 4, value - err_low),
                width / 8,
                err_low + err_high,
                fill=False,
                lw=0,
                color="purple",
                hatch="///",
            )
        )

    for center, value, err_low, err_high, width in zip(
        bins_c, total_powheg, theory["unc_th_powheg_dn"], theory["unc_th_powheg_up"], bin_w
    ):
        plt.gca().add_patch(
            plt.Rectangle(
                (center + width / 10, value - err_low),
                width / 8,
                err_low + err_high,
                fill=False,
                lw=0,
                color="tab:blue",
                hatch="///",
            )
        )


def plot_data_main(ax, bins_c, measurement):
    plt.sca(ax)
    plt.errorbar(
        bins_c,
        measurement["exp_xs"],
        yerr=[measurement["err_down"], measurement["err_up"]],
        marker="o",
        linestyle="None",
        color="k",
        linewidth=2,
        ms=5,
        capsize=4,
        label=r"Data (stat $\oplus$ sys unc.)",
    )
    plt.errorbar(
        bins_c,
        measurement["exp_xs"],
        yerr=[measurement["sys_down"], measurement["sys_up"]],
        marker="None",
        linestyle="None",
        color="red",
        linewidth=6,
        ms=5,
        capsize=4,
        label="Systematic Uncertainty",
    )


def plot_vbf_scans(ax, bins_c, bin_w, cfg):
    plt.sca(ax)
    scan_data = cfg.get("vbf_scans", {})
    scans = [
        ("total_minus_vbf", -0.12, "dimgrey", "VBF fit (non-VBF fixed)"),
        ("all_extrap", 0.12, "darkgrey", "VBF fit (all extrapolated fixed)"),
    ]

    for key, offset, color, label in scans:
        scan = scan_data.get(key)
        if not scan:
            raise KeyError(f"Missing vbf_scans.{key} in the plotting configuration")

        x = bins_c + offset * bin_w
        y = np.asarray(scan["xs"], dtype=float) / bin_w
        total_up = np.asarray(scan["err_up"], dtype=float) / bin_w
        total_down = np.asarray(scan["err_down"], dtype=float) / bin_w
        stat_up = np.asarray(scan["stat_up"], dtype=float) / bin_w
        stat_down = np.asarray(scan["stat_down"], dtype=float) / bin_w
        sys_up = np.sqrt(np.maximum(0.0, total_up**2 - stat_up**2))
        sys_down = np.sqrt(np.maximum(0.0, total_down**2 - stat_down**2))

        plt.errorbar(
            x,
            y,
            yerr=[total_down, total_up],
            marker="o",
            linestyle="None",
            color=color,
            linewidth=2,
            ms=5,
            capsize=4,
            label=label,
            zorder=5,
        )
        plt.errorbar(
            x,
            y,
            yerr=[sys_down, sys_up],
            marker="None",
            linestyle="None",
            color="red",
            linewidth=6,
            capsize=4,
            zorder=4,
        )


def get_ylabel(variable, cfg, double_diff):
    var_label = get_var_label(cfg["variable"])
    if variable in {"mass4l", "mass4l_zzfloating"}:
        return r"$\sigma_{\text{fid}}$ (fb)"
    if double_diff:
        return r"$d^2\sigma_{\text{fid}} / d" + var_label + r"\;" + cfg["y_unit"] + "$"
    return r"$d\sigma_{\text{fid}} / d" + var_label + r"\;" + cfg["y_unit"] + "$"


def get_y_lim_top(variable, cfg, args):
    y_lim_top = cfg.get("y_lim_top")
    if y_lim_top is None:
        return None
    if not args.ZZ or "zzfloating" not in variable:
        return y_lim_top
    if get_base_variable_name(variable) == "mass4l":
        return y_lim_top
    if get_base_variable_name(variable) == "rapidity4l":
        return y_lim_top + 1
    return y_lim_top * 10


def style_main_y_ticks(ax, variable, cfg):
    ax.minorticks_on()

    if cfg["plot_log"]:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ytick_fs = 15 if "zzfloating" in variable else 20
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        labelleft=True,
        labelsize=ytick_fs,
        length=6,
        width=1.0,
    )
    ax.tick_params(axis="y", which="minor", left=True, right=False, length=3, width=0.8)


def style_main_panel(ax, variable, cfg, bins_plot, args):
    plt.sca(ax)

    plt.ylabel(get_ylabel(variable, cfg, is_double_diff(cfg)), fontsize=20)

    if cfg["plot_log"]:
        plt.yscale("log")

    y_lim_top = get_y_lim_top(variable, cfg, args)
    if y_lim_top is not None:
        plt.ylim(top=y_lim_top)
    if "y_lim_bottom" in cfg:
        plt.ylim(bottom=cfg["y_lim_bottom"])

    plt.xlim(bins_plot[0], bins_plot[-1])
    plt.xticks(fontsize=20)

    style_main_y_ticks(ax, variable, cfg)


def plot_ratio_panel(ax, bins_plot, bins_c, bin_w, theory, measurement, variable):
    plt.sca(ax)

    denom = theory["ggh_xs_norm"] + theory["xh_xs_norm"]
    ratio_xs = measurement["exp_xs"] / denom
    ratio_powheg = (theory["ggh_powheg_xs_norm"] + theory["xh_xs_norm"]) / denom
    ratio_madgraph = denom / denom

    ratio_err_up = measurement["err_up"] / denom
    ratio_err_down = measurement["err_down"] / denom
    ratio_sys_up = measurement["sys_up"] / denom
    ratio_sys_down = measurement["sys_down"] / denom

    ratio_unc_up = theory["unc_th_up"] / denom
    ratio_unc_dn = theory["unc_th_dn"] / denom
    ratio_unc_powheg_up = theory["unc_th_powheg_up"] / denom
    ratio_unc_powheg_dn = theory["unc_th_powheg_dn"] / denom

    plt.errorbar(
        bins_c,
        ratio_xs,
        yerr=[ratio_err_down, ratio_err_up],
        marker="o",
        linestyle="None",
        color="k",
        linewidth=2,
        ms=5,
        capsize=4,
        label="Toy Data (Stat + Syst)",
    )
    plt.errorbar(
        bins_c,
        ratio_xs,
        yerr=[ratio_sys_down, ratio_sys_up],
        marker="None",
        linestyle="None",
        color="red",
        linewidth=6,
        ms=5,
        capsize=4,
        label="Systematic error",
    )

    plt.hlines(1, -1000, 1000, color="purple")

    for center, value, err_low, err_high, width in zip(
        bins_c, ratio_madgraph, ratio_unc_dn, ratio_unc_up, bin_w
    ):
        plt.gca().add_patch(
            plt.Rectangle(
                (center - width / 4, value - err_low),
                width / 8,
                err_low + err_high,
                fill=False,
                lw=0,
                color="purple",
                hatch="/////",
            )
        )

    for center, value, err_low, err_high, width in zip(
        bins_c, ratio_powheg, ratio_unc_powheg_dn, ratio_unc_powheg_up, bin_w
    ):
        plt.gca().add_patch(
            plt.Rectangle(
                (center + width / 10, value - err_low),
                width / 8,
                err_low + err_high,
                fill=False,
                lw=0,
                color="tab:blue",
                hatch="/////",
            )
        )

    plt.stairs(ratio_powheg, bins_plot, linewidth=2, color="tab:blue")

    for b in bins_plot:
        ax.axvline(x=b, color="gray", ls="dashed", lw=1, alpha=0.5)

    ax.set_ylabel(
        "Ratio to\nNNLOPS+POWHEG",
        fontsize=12 if "zzfloating" in variable else 15,
        #rotation=90,
        va="center",
        ha="center",
        multialignment="center",
        labelpad=18,
    )
    ax.set_xlim(bins_plot[0], bins_plot[-1])
    ax.set_ylim(0, 2)

    if "zzfloating" in variable:
        ax.set_yticks([0.0, 1.0, 2.0])
        ax.set_yticklabels(["0", "1", "2"], fontsize=15)
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    else:
        ax.tick_params(axis="y", labelsize=20)


def apply_custom_xticks(axis, variable, bins_c, bins_plot):
    custom_map = {
        "mass4l": (["$4\\ell$", "$2e2\\mu$", "$4\\mu$", "$4e$"], [0.5, 1.5, 2.5, 3.5], None),
        "mass4l_zzfloating": (["$4\\ell$", "$2e2\\mu$", "$4\\mu$", "$4e$"], [0.5, 1.5, 2.5, 3.5], None),
        "Nj": (["0", "1", "2", "3", "$\\geq4$"], [0.5, 1.5, 2.5, 3.5, 4.5], None),
        "pTj1": ([r"$\sigma(\it{N}_{\rm{Jets}}=0)$", "30", "60", "90", "120", "150", "180", "210", "240"], [10, 30, 60, 90, 120, 150, 180, 210, 240], 30),
        "pTj1_zzfloating": ([r"$\sigma(\it{N}_{\rm{Jets}}=0)$", "30", "60", "90", "120", "150", "180", "210", "240"], [10, 30, 60, 90, 120, 150, 180, 210, 240], 30),
        "pTj2": ([r"$\sigma(\it{N}_{\rm{Jets}} \leq 1)$", "40", "60", "80", "100", "120", "140"], [15, 40, 60, 80, 100, 120, 140], 30),
        "mjj": ([r"$\sigma(\it{N}_{\rm{Jets}} \leq 1)$", "0", "75", "150", "225", "300", "375", "450", "525"], [-50, 0, 75, 150, 225, 300, 375, 450, 525], 0),
        "absdetajj": ([r"$\sigma(\it{N}_{\rm{Jets}} \leq 1)$", "0", "2", "4", "6", "8", "10"], [-1, 0, 2, 4, 6, 8, 10], 0),
        "dphijj": ([r"$\sigma(\it{N}_{\rm{Jets}} \leq 1)$", "-3", "-2", "-1", "0", "1", "2", "3"], [-3.57, -3, -2, -1, 0, 1, 2, 3], -3.14),
        "mHj": ([r"$\sigma(\it{N}_{\rm{Jets}}=0)$", "0", "110", "220", "330", "440", "550", "660", "770", "880"], [-50, 0, 110, 220, 330, 440, 550, 660, 770, 880], 0),
        "pTHj": ([r"$\sigma(\it{N}_{\rm{Jets}}=0)$", "0", "25", "50", "75", "100", "125", "150", "175", "200"], [-12.5, 0, 25, 50, 75, 100, 125, 150, 175, 200], 0),
        "pTHjj": ([r"$\sigma(\it{N}_{\rm{Jets}}\leq 1)$", "0", "25", "50", "75", "100"], [-12.5, 0, 25, 50, 75, 100], 0),
        "TCjmax": ([r"$\sigma(\it{N}_{\rm{Jets}}=0)$", "0", "20", "40", "60", "80"], [-10, 0, 20, 40, 60, 80], 0),
        "TBjmax": ([r"$\sigma(\it{N}_{\rm{Jets}}=0)$", "0", "20", "40", "60", "80"], [-10, 0, 20, 40, 60, 80], 0),
    }


    custom_variable = variable if variable in custom_map else get_base_variable_name(variable)
    if custom_variable in custom_map:
        labels, ticks, vline = custom_map[custom_variable]
        if variable in SPECIAL_BIN_WIDTH_VARS:
            ticks = list(ticks)
            ticks[0] = bins_c[0]
            vline = bins_plot[1]
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
        axis.xaxis.set_minor_locator(plt.NullLocator())
        if vline is not None:
            axis.axvline(x=vline, color="black", ls="dashed", lw=1, alpha=1)
        if variable in SPECIAL_BIN_WIDTH_VARS:
            axis.get_xticklabels()[0].set_rotation(30)
            axis.get_xticklabels()[0].set_ha("right")
        return

    double_diff_variable = variable if variable in DOUBLE_DIFF_CUSTOM_XTICKS else get_base_variable_name(variable)
    if double_diff_variable in DOUBLE_DIFF_CUSTOM_XTICKS:
        labels = DOUBLE_DIFF_CUSTOM_XTICKS[double_diff_variable]
        ticks = np.asarray(bins_c[:len(labels)], dtype=float)

        axis.set_xlim(bins_plot[0], bins_plot[-1])
        axis.set_xticks(ticks, labels=labels)
        axis.xaxis.set_minor_locator(plt.NullLocator())

        for label in axis.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
            label.set_va("top")
            label.set_fontsize(4)

        return

    if variable in SPECIAL_OBS:
        return

    axis.xaxis.set_minor_locator(plt.NullLocator())


def apply_double_diff_xticks(axis, bin_number, bins_c):
    if bin_number is None:
        return
    axis.set_xticklabels([f"Bin {i}" for i in range(1, bin_number + 1)])
    axis.set_xticks(bins_c)
    axis.xaxis.set_minor_locator(plt.NullLocator())


def get_zzfloating_panel_positions(variable, bins_c, bin_w):
    groups = get_zzfloating_merged_bin_groups(variable, len(bins_c))
    if len(groups) == len(bins_c):
        return np.array(bins_c, dtype=float), np.array(bin_w, dtype=float)

    centers = []
    widths = []
    for group in groups:
        group_centers = [bins_c[i] for i in group]
        group_widths = [bin_w[i] for i in group]
        total_width = sum(group_widths)
        if total_width > 0:
            weighted_center = sum(c * w for c, w in zip(group_centers, group_widths)) / total_width
        else:
            weighted_center = np.mean(group_centers)
        centers.append(weighted_center)
        widths.append(total_width)
    return np.array(centers, dtype=float), np.array(widths, dtype=float)


def plot_zzfloating_panel(ax, bins_plot, bins_c, bin_w, cfg, var_label):
    plt.sca(ax)

    zz_exp = np.array(cfg["zznorm_exp"], dtype=float)
    zz_exp_up = np.array(cfg["zznorm_up_exp"], dtype=float)
    zz_exp_dn = np.array(cfg["zznorm_down_exp"], dtype=float)
    zz_exp_stat_up = np.array(cfg["zznorm_stat_up_exp"], dtype=float)
    zz_exp_stat_dn = np.array(cfg["zznorm_stat_down_exp"], dtype=float)

    zz_obs = np.array(cfg["zznorm_obs"], dtype=float)
    zz_obs_up = np.array(cfg["zznorm_up_obs"], dtype=float)
    zz_obs_dn = np.array(cfg["zznorm_down_obs"], dtype=float)
    zz_obs_stat_up = np.array(cfg["zznorm_stat_up_obs"], dtype=float)
    zz_obs_stat_dn = np.array(cfg["zznorm_stat_down_obs"], dtype=float)

    panel_centers, panel_widths = get_zzfloating_panel_positions(cfg["variable"], bins_c, bin_w)

    if len(zz_exp) != len(panel_centers):
        if len(zz_exp) == len(bins_c):
            panel_centers = np.array(bins_c, dtype=float)
            panel_widths = np.array(bin_w, dtype=float)
        else:
            raise ValueError(
                f"ZZ floating panel shape mismatch for {cfg['variable']}: "
                f"{len(zz_exp)} ZZ points for {len(panel_centers)} merged positions "
                f"and {len(bins_c)} raw bins."
            )

    common_len = min(
        len(panel_centers),
        len(panel_widths),
        len(zz_exp),
        len(zz_exp_up),
        len(zz_exp_dn),
        len(zz_exp_stat_up),
        len(zz_exp_stat_dn),
        len(zz_obs),
        len(zz_obs_up),
        len(zz_obs_dn),
        len(zz_obs_stat_up),
        len(zz_obs_stat_dn),
    )
    if common_len == 0:
        print(f"Warning: no ZZ floating ratio points available for {cfg['variable']}")
        plt.hlines(1.0, bins_plot[0], bins_plot[-1], color="gray", linewidth=1.5)
        ax.set_xlim(bins_plot[0], bins_plot[-1])
        ax.set_ylim(0, 5)
        ax.set_ylabel(r"$ZZ/ZZ_{MC}$", fontsize=12, rotation=90, va="center", ha="center", multialignment="center", labelpad=18)
        plt.xlabel(r"$" + var_label + r"$" + cfg["x_unit"], fontsize=20)
        plt.xticks(fontsize=16)
        return

    if common_len < len(panel_centers):
        print(
            f"Warning: plotting {common_len} ZZ floating ratio points for {cfg['variable']} "
            f"out of {len(panel_centers)} panel bins."
        )

    panel_centers = panel_centers[:common_len]
    panel_widths = panel_widths[:common_len]
    zz_exp = zz_exp[:common_len]
    zz_exp_up = zz_exp_up[:common_len]
    zz_exp_dn = zz_exp_dn[:common_len]
    zz_exp_stat_up = zz_exp_stat_up[:common_len]
    zz_exp_stat_dn = zz_exp_stat_dn[:common_len]
    zz_obs = zz_obs[:common_len]
    zz_obs_up = zz_obs_up[:common_len]
    zz_obs_dn = zz_obs_dn[:common_len]
    zz_obs_stat_up = zz_obs_stat_up[:common_len]
    zz_obs_stat_dn = zz_obs_stat_dn[:common_len]

    ratio_zz = zz_obs / zz_exp
    ratio_zz_up = ratio_zz * np.sqrt((zz_obs_up / zz_obs) ** 2 + (zz_exp_up / zz_exp) ** 2)
    ratio_zz_dn = ratio_zz * np.sqrt((zz_obs_dn / zz_obs) ** 2 + (zz_exp_dn / zz_exp) ** 2)
    ratio_zz_stat_up = ratio_zz * np.sqrt((zz_obs_stat_up / zz_obs) ** 2 + (zz_exp_stat_up / zz_exp) ** 2)
    ratio_zz_stat_dn = ratio_zz * np.sqrt((zz_obs_stat_dn / zz_obs) ** 2 + (zz_exp_stat_dn / zz_exp) ** 2)
    ratio_zz_sys_up = np.sqrt(np.maximum(0, ratio_zz_up**2 - ratio_zz_stat_up**2))
    ratio_zz_sys_dn = np.sqrt(np.maximum(0, ratio_zz_dn**2 - ratio_zz_stat_dn**2))

    for i, (center, value, err_low, err_high, width) in enumerate(zip(
        panel_centers, ratio_zz, ratio_zz_sys_dn, ratio_zz_sys_up, panel_widths
    )):
        plt.hlines(
            value,
            center - 0.45 * width,
            center + 0.45 * width,
            color="brown",
            linewidth=2,
            label="Fitted ZZ / Predicted ZZ" if i == 0 else None,
        )
        plt.vlines(
            center,
            value - err_low,
            value + err_high,
            color="brown",
            linewidth=2,
        )

    plt.hlines(1.0, bins_plot[0], bins_plot[-1], color="gray", linewidth=1.5)

    for b in bins_plot:
        ax.axvline(x=b, color="gray", ls="dashed", lw=1, alpha=0.5)

    ax.tick_params(axis="y", labelsize=15)
    ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ax.set_yticklabels(["0", "1", "2", "3", "4", "5"], fontsize=15)
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.set_xlim(bins_plot[0], bins_plot[-1])
    ax.set_ylim(0, 5)
    ax.set_ylabel(
    r"$ZZ/ZZ_{MC}$",
    fontsize=12,
    rotation=90,
    va="center",
    ha="center",
    multialignment="center",
    labelpad=18,
    )
    plt.xlabel(r"$" + var_label + r"$" + cfg["x_unit"], fontsize=20)
    plt.xticks(fontsize=16)


def finalize_bottom_axis(axis, variable, cfg):
    var_label = get_var_label(cfg["variable"])
    fs_label = ""
    plt.xlabel(r"$" + var_label + r"$" + cfg["x_unit"] + fs_label, fontsize=20)

    fontsize_x = 16 if variable in ["Nj_pT4l", "rapidity4l_pT4l"] else 20
    plt.xticks(fontsize=fontsize_x)
    plt.yticks(fontsize=20)

    plt.xlim(cfg["x_lim"][0], cfg["x_lim"][-1] if "x_lim" in cfg else axis.get_xlim()[1])
    plt.ylim(0, 2)


def save_plot(fig, cfg, out_suffix):
    out_base = f"{path['plots_path']}PLOTS/{cfg['output_name']}{out_suffix}"
    print(f"Saving plot to: {out_base}.pdf")
    bottom_padding = 2.8
    fixed_bbox = Bbox.from_bounds(
        0,
        -bottom_padding,
        fig.get_figwidth(),
        fig.get_figheight() + bottom_padding,
    )
    fig.savefig(f"{out_base}.pdf", bbox_inches=fixed_bbox, dpi=600)
    fig.savefig(f"{out_base}.png", bbox_inches=fixed_bbox, dpi=600)
    plt.close(fig)


def main():
    args = get_options()
    config = load_json(args.config_json)

    variable_list = resolve_plot_variable_list(args.variables, config, args.ZZ, args.YEAR)

    for variable in variable_list:
        plot_jobs = create_plot_jobs(variable, config, args)

        for ch_label, current_config, out_suffix, bins_plot, bins_c, bin_w in plot_jobs:
            double_diff = is_double_diff(current_config)
            _, _, _, bin_number = get_bins(variable, current_config) if variable not in {"mass4l", "mass4l_zzfloating"} else (None, None, None, None)

            if variable in {"mass4l", "mass4l_zzfloating"}:
                theory = build_theory_mass4l(current_config, bin_w)
            else:
                theory = build_theory_standard(variable, current_config, bin_w)

            measurement = build_measurement(variable, current_config, bin_w)

            fig, frame1, frame2, frame3 = setup_figure(variable)

            plt.sca(frame1)
            apply_cms_label(current_config, args)
            plot_data_main(frame1, bins_c, measurement)
            plot_theory_main(frame1, bins_plot, bins_c, bin_w, theory)
            if variable == "absdetajj_mjj":
                plot_vbf_scans(frame1, bins_c, bin_w, current_config)
            style_main_panel(frame1, variable, current_config, bins_plot, args)

            pval = PVALUE_MAP.get(variable)

            if pval is not None:
                plt.text(
                    0.03,
                    0.95,
                    rf"$p$-value = {pval:.2f}",
                    transform=frame1.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    horizontalalignment="left",
                )

            plt.sca(frame2)
            plot_ratio_panel(frame2, bins_plot, bins_c, bin_w, theory, measurement, variable)

            axis_for_xticks = frame3 if "zzfloating" in variable else frame2
            plt.sca(axis_for_xticks)

            if variable in {"mass4l", "mass4l_zzfloating"}:
                apply_custom_xticks(axis_for_xticks, variable, bins_c, bins_plot)
            elif variable == "Nj":
                apply_custom_xticks(axis_for_xticks, variable, bins_c, bins_plot)
            elif variable in SPECIAL_BIN_WIDTH_VARS:
                apply_custom_xticks(axis_for_xticks, variable, bins_c, bins_plot)
            elif double_diff:
                #apply_double_diff_xticks(axis_for_xticks, bin_number, bins_c)
                apply_custom_xticks(axis_for_xticks, variable, bins_c, bins_plot)

            if "zzfloating" in variable:
                var_label = get_var_label(current_config["variable"])
                plot_zzfloating_panel(frame3, bins_plot, bins_c, bin_w, current_config, var_label)

                handles1, labels1 = frame1.get_legend_handles_labels()
                handles3, labels3 = frame3.get_legend_handles_labels()

                handles1, labels1 = reorder_legend(handles1, labels1)
                handles3, labels3 = reorder_legend(handles3, labels3)

                legend = frame1.legend(
                    list(handles1) + list(handles3),
                    list(labels1) + list(labels3),
                    fontsize=16,
                    loc=current_config.get("legend_location", "upper right"),
                    title_fontsize=14,
                )
            else:
                handles, labels = frame1.get_legend_handles_labels()
                handles, labels = reorder_legend(handles, labels)

                legend = frame1.legend(
                    list(handles),
                    list(labels),
                    fontsize=16,
                    loc=current_config.get("legend_location", "upper right"),
                    title_fontsize=14,
                )

            legend._legend_box.align = "left"

            plt.sca(axis_for_xticks)
            var_label = get_var_label(current_config["variable"])
            if "zzfloating" not in variable:
                plt.xlabel(r"$" + var_label + r"$" + current_config["x_unit"], fontsize=20)
                plt.xticks(fontsize=16 if variable in ["Nj_pT4l", "rapidity4l_pT4l"] else 20)
                plt.yticks(fontsize=20)
                plt.xlim(bins_plot[0], bins_plot[-1])
                plt.ylim(0, 2)

            save_plot(fig, current_config, out_suffix)


if __name__ == "__main__":
    main()
