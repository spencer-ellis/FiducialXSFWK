from ROOT import *
import numpy as np
import math
from functools import partial
import plotting as plot
import json
import argparse, optparse
import os.path, sys
import os

sys.path.append('../helperstuff/')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fit.createDatacard import get_zzfloating_merged_bin_indices, ZZFLOATING_BIN_MERGES
from paths import path

from ROOT import *
gROOT.SetBatch(True)

def get_merged_bins_for_zz(obsName_base, zz_bin_index):
    """Get list of original bins merged into a given ZZ normalization bin.
    
    Args:
        obsName_base: Observable name without '_zzfloating' suffix
        zz_bin_index: Index of ZZ normalization bin (0, 1, 2, ...)
    
    Returns:
        List of original bin indices that map to this ZZ bin, or None if not merged
    """
    if obsName_base not in ZZFLOATING_BIN_MERGES:
        return None
    
    mapping = ZZFLOATING_BIN_MERGES[obsName_base]
    if zz_bin_index >= len(mapping):
        return None
    
    group = mapping[zz_bin_index]
    if isinstance(group, int):
        return [group]
    elif isinstance(group, (list, tuple)):
        return list(group)
    return None

def format_double_diff_bin_line(obs_bin, label, label_2nd):
    return f"{obs_bin[0]} < {label} < {obs_bin[1]}, {obs_bin[2]} < {label_2nd} < {obs_bin[3]}"

NAMECOUNTER = 0

grootargs = []
def callback_rootargs(option, opt, value, parser):
    grootargs.append(opt)

def tfile_is_good(filename):
    try:
        return plot.TFileIsGood(filename)
    except OSError:
        return False

### Define function for parsing options
def parseOptions():

    global opt, args, runAllSteps

    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)

    # input options
    parser.add_option('',   '--obsName',dest='OBSNAME',    type='string',default='',   help='Name of the observalbe, supported: "mass4l", "pT4l", "massZ2", "rapidity4l", "cosThetaStar", "nets_reco_pt30_eta4p7"')
    parser.add_option('',   '--obsBins',  dest='OBSBINS',  type='string',default='',   help='Bin boundaries for the diff. measurement separated by "|", e.g. as "|0|50|100|", use the defalut if empty string')
    parser.add_option('',   '--year',  dest='YEAR',  type='string',default='',   help='Year -> 2016 or 2017 or 2018 or Full')
    parser.add_option('',   '--unblind', action='store_true', dest='UNBLIND', default=False, help='Use real data')
    parser.add_option('',   '--v4', action='store_true', dest='V4', default= False, help='Print NLL scans for v4 physics model')
    parser.add_option('',   '--interpolation', action='store_true', dest='INTER', default=False, help='Calculate acceptances at 124 and 126 GeV')
    parser.add_option('',   '--ZZfloating',action='store_true', dest='ZZ',default=False, help='Let ZZ normalisation to float')
    parser.add_option('',   '--doVBF', action='store_true', dest='DO_VBF', default=False, help='Also plot the floating VBF POIs for each absdetajj vs mjj bin')

    # store options and arguments as global variables
    global opt, args
    (opt, args) = parser.parse_args()

    if opt.DO_VBF and opt.OBSNAME.strip() != 'absdetajj vs mjj':
        parser.error('--doVBF may only be used with --obsName "absdetajj vs mjj"')

# parse the arguments and options
global opt, args, runAllSteps
parseOptions()
sys.argv = grootargs

def read(scan, param, files, ycut):
    goodfiles = [f for f in files if tfile_is_good(f)]
    if not goodfiles:
        print('Skipping missing scan input(s):', ', '.join(files))
        return None
    limit = plot.MakeTChain(goodfiles, 'limit')
    graph = plot.TGraphFromTree(limit, param, '2*deltaNLL', 'quantileExpected > -1.5')
    if graph is None or graph.GetN() < 2:
        print('Skipping empty scan:', scan, 'from', ', '.join(goodfiles))
        return None
    graph.SetName(scan)
    graph.Sort()
    plot.RemoveGraphXDuplicates(graph)
    plot.RemoveGraphYAbove(graph, ycut)
    # graph.Print()
    return graph


def Eval(obj, x, params):
    return obj.Eval(x[0])

def BuildScan(scan, param, files, color, yvals, ycut):
    graph = read(scan, param, files, ycut)
    if graph is None:
        return None
    bestfit = None

    graph.SetMarkerColor(color)
    spline = TSpline3("spline3", graph)
    global NAMECOUNTER
    pyfunc = partial(Eval, spline)
    func = TF1('splinefn'+str(NAMECOUNTER), pyfunc, graph.GetX()[0], graph.GetX()[graph.GetN() - 1], 1)
    bestfit = func.GetMinimumX() #AT
    NAMECOUNTER += 1
    func.SetLineColor(color)
    func.SetLineWidth(3)
    if bestfit is None:
        print('Skipping scan with no best fit:', scan, 'from', ', '.join(files))
        return None
    crossings = {}
    cross_1sig = None
    cross_2sig = None
    other_1sig = []
    other_2sig = []
    val = None
    val_2sig = None
    for yval in yvals:
        crossings[yval] = plot.FindCrossingsWithSpline(graph, func, yval)
        for cr in crossings[yval]:
            cr["contains_bf"] = cr["lo"] <= bestfit and cr["hi"] >= bestfit
    for cr in crossings[yvals[0]]:
        if cr['contains_bf']:
            val = (bestfit, cr['hi'] - bestfit, cr['lo'] - bestfit)
            cross_1sig = cr
        else:
            other_1sig.append(cr)
    if len(yvals) > 1:
        for cr in crossings[yvals[1]]:
            if cr['contains_bf']:
                val_2sig = (bestfit, cr['hi'] - bestfit, cr['lo'] - bestfit)
                cross_2sig = cr
            else:
                other_2sig.append(cr)
    else:
        val_2sig = (0., 0., 0.)
        cross_2sig = cross_1sig
    return {
        "graph"     : graph,
        "spline"    : spline,
        "func"      : func,
        "crossings" : crossings,
        "val"       : val,
        "val_2sig": val_2sig,
        "cross_1sig" : cross_1sig,
        "cross_2sig" : cross_2sig,
        "other_1sig" : other_1sig,
        "other_2sig" : other_2sig
    }

def quadrature_subtract(total_unc, stat_unc):
    diff = total_unc**2 - stat_unc**2
    if diff < 0:
        print('Warning: stat-only uncertainty is larger than stat+sys; setting syst component to 0.')
        return 0.0
    return math.sqrt(diff)

def build_2d_scan_graph(fname, poi_x, poi_y):
    if not tfile_is_good(fname):
        return None, None
    in_file = TFile.Open(fname, "READ")
    tree = in_file.Get("limit")
    graph = TGraph2D()
    graph.SetName("scan2d_"+str(abs(hash(fname)) % 1000000))
    best = None
    ipoint = 0
    for entry in tree:
        if hasattr(entry, 'quantileExpected') and entry.quantileExpected <= -1.5:
            continue
        xval = getattr(entry, poi_x)
        yval = getattr(entry, poi_y)
        zval = 2.0 * entry.deltaNLL
        if zval < 0 and abs(zval) < 1e-6:
            zval = 0.0
        graph.SetPoint(ipoint, xval, yval, zval)
        if best is None or zval < best[2]:
            best = (xval, yval, zval)
        ipoint += 1
    in_file.Close()
    if graph.GetN() < 3:
        return None, None
    return graph, best

def plot_vbf_2d_scans(obsName, raw_nBins, obs_bins, label, label_2nd, year, inputPath):
    outdir = os.path.join(path['plots_path'], "SCANS", obsName)
    os.makedirs(outdir, exist_ok=True)
    scan_specs = [
        ("Expected", "expected", "", ".123456.root"),
        ("Expected - stat-only", "expected_statOnly", "_NoSys", ".123456.root"),
    ]
    if opt.UNBLIND:
        scan_specs += [
            ("Observed", "observed", "", ".root"),
            ("Observed - stat-only", "observed_statOnly", "_NoSys", ".root"),
        ]

    for physical_bin in range(raw_nBins):
        poi_x = 'r_VBFH_'+obsName+'_'+str(physical_bin)
        poi_y = 'r_otherProd_'+obsName+'_'+str(physical_bin)
        scan_name = 'r_VBFH_vs_otherProd_'+str(physical_bin)
        for title, suffix, scan_suffix, file_suffix in scan_specs:
            fname = os.path.join(
                inputPath,
                'higgsCombine_'+obsName+'_'+scan_name+scan_suffix+'.MultiDimFit.mH125.38'+file_suffix
            )
            graph, best = build_2d_scan_graph(fname, poi_x, poi_y)
            if graph is None:
                print('Skipping missing or empty 2D VBF scan:', fname)
                continue

            c2 = TCanvas('c_'+scan_name+'_'+suffix, '', 800, 700)
            c2.SetRightMargin(0.16)
            c2.SetLeftMargin(0.12)
            c2.SetBottomMargin(0.12)
            graph.SetTitle('')
            graph.SetNpx(80)
            graph.SetNpy(80)
            graph.SetMinimum(0.0)
            graph.SetMaximum(10.0)
            graph.Draw('COLZ')
            graph.GetXaxis().SetTitle('r_{VBF}')
            graph.GetYaxis().SetTitle('r_{other prod.}')
            graph.GetZaxis().SetTitle('2 #Delta NLL')
            graph.GetXaxis().SetTitleOffset(1.15)
            graph.GetYaxis().SetTitleOffset(1.25)
            graph.GetZaxis().SetTitleOffset(1.25)

            if best is not None:
                marker = TMarker(best[0], best[1], 34)
                marker.SetMarkerColor(kBlack)
                marker.SetMarkerSize(1.8)
                marker.Draw('SAME')

            latex = TLatex()
            latex.SetNDC()
            latex.SetTextFont(42)
            latex.SetTextSize(0.035)
            latex.DrawLatex(0.16, 0.92, title)
            latex.DrawLatex(0.16, 0.86, format_double_diff_bin_line(obs_bins[physical_bin], label, label_2nd))
            latex.DrawLatex(0.60, 0.92, _lumi+' fb^{-1} (13.6 TeV)')

            c2.Update()
            c2.SaveAs(os.path.join(
                outdir,
                year+'_lhscan_2d_'+obsName+'_'+scan_name+'_'+suffix+'.png'
            ))

def get_vbf_scan_specs(raw_nBins):
    vbf_scan_kinds = ['vbf', 'other_prod', 'total_minus_vbf', 'ggh_extrap', 'ggh_fixed']
    return [
        {
            'kind': kind,
            'physical_bin': physical_bin,
        }
        for physical_bin in range(raw_nBins)
        for kind in vbf_scan_kinds
    ]

yvals = [1., 3.84]

obsName = opt.OBSNAME
print(obsName)

channel = ["Expected","Expected - no syst.","Observed","Observed - no syst."]

inputPath = path['eos_path']+'combine_files/'

if 'kL' in obsName:
    fileList = [ "higgsCombine_BIN_grid.MultiDimFit.mH125.38.123456.root",
                 "higgsCombine_BIN_NoSys_grid.MultiDimFit.mH125.38.123456.root"]
else:
    fileList = [ "higgsCombine_BIN_OBS.MultiDimFit.mH125.38.123456.root",
                 "higgsCombine_BIN_OBS_NoSys.MultiDimFit.mH125.38.123456.root"]

if(opt.UNBLIND and 'kL' in obsName):
    fileList = [ "higgsCombine_BIN_grid.MultiDimFit.mH125.38.123456.root",
                 "higgsCombine_BIN_NoSys_grid.MultiDimFit.mH125.38.123456.root",
                 "higgsCombine_BIN_grid.MultiDimFit.mH125.38.root",
                 "higgsCombine_BIN_NoSys_grid.MultiDimFit.mH125.38.root"]
elif(opt.UNBLIND):
    fileList = [ "higgsCombine_BIN_OBS.MultiDimFit.mH125.38.123456.root",
                 "higgsCombine_BIN_OBS_NoSys.MultiDimFit.mH125.38.123456.root",
                 "higgsCombine_BIN_OBS.MultiDimFit.mH125.38.root",
                 "higgsCombine_BIN_OBS_NoSys.MultiDimFit.mH125.38.root"]

titles = ["Expected","Expected - stat-only"]
idx_max = 1

if(opt.UNBLIND):
    print('Using real data')
    idx_max = 3
    titles = ["Expected","Expected - stat-only","Observed","Observed - stat-only"]
colors = [kRed, kRed, kBlack, kBlack]

resultsXS_data = {}
resultsXS_asimov = {}
if opt.OBSNAME.startswith('mass4l'):
    resultsXS_data_v2 = {}
    resultsXS_asimov_v2 = {}
if opt.V4:
    resultsXS_data_v4 = {}
    resultsXS_asimov_v4 = {}

year = opt.YEAR

if year == '2016':
    _lumi = '36.33'
elif year == '2017':
    _lumi = '41.48'
elif year == '2018':
    _lumi = '59.83'

elif year == 'Run3':
    _lumi = '171' #62

elif year == '2022':
    _lumi = '7.98'
elif year == '2022EE':
    _lumi = '26.67'
elif year == '2023preBPix':
    _lumi = '18.06'
elif year == '2023postBPix':
    _lumi = '9.69'
elif year == '2024':
    _lumi = '108.82'

elif year == '2022full':
    _lumi = '34.7'
elif year == '2023full':
    _lumi = '27.7'
    
else:
    _lumi = '138'

_poi    = 'r_smH_'
v4_flag = opt.V4

doubleDiff = False
if(obsName == 'mass4l'): label = 'm_{4l}'
elif(obsName == 'mass4l_zzfloating'): label = 'm_{4l}'
elif(obsName == 'Nj'): label = 'N_{jet}, pT>30 GeV, |#eta|<4.7'
elif(obsName == 'pT4l'): label = 'p_{T}^{H} (GeV)'
elif(obsName == 'pT4l_kL'): label = '#kappa_{#lambda}'
elif(obsName == 'rapidity4l'): label = '|y_{H}|'
elif(obsName == 'costhetaZ1'): label = 'cos(#theta_{1})'
elif(obsName == 'costhetaZ2'): label = 'cos(#theta_{2})'
elif(obsName == 'phi'): label = '#Phi'
elif(obsName == 'phi1'): label = '#Phi1'
elif(obsName == 'phistar'): label = '#Phi^{#star}'
elif(obsName == 'costhetastar'): label = 'cos(#theta^{*})'
elif(obsName == 'massZ1'): label = 'm_{Z1} (GeV)'
elif(obsName == 'massZ2'): label = 'm_{Z2} (GeV)'
elif(obsName == 'pTj1'): label = 'p_{T}^{(j1, 4.7)} (GeV)'
elif(obsName == 'pTHj'): label = 'p_{T}^{Hj} (GeV)'
elif(obsName == 'mHj'): label = 'm_{Hj} (GeV)'
elif(obsName == 'pTj2'): label = 'p_{T}^{(j2, 4.7)} (GeV)'
elif(obsName == 'mjj'): label = 'm_{jj} (GeV)'
elif(obsName == 'absdetajj'): label = '|#Delta#Eta_{jj}|'
elif(obsName == 'dphijj'): label = '|#Delta#Phi_{jj}|'
elif(obsName == 'pTHjj'): label = 'p_{T}^{Hjj} (GeV)'
elif(obsName == 'TCjmax'): label = 'TCjmax'
elif(obsName == 'TBjmax'): label = 'TBjmax'
elif(obsName == 'D0m'): label = 'D_{0m}'
elif(obsName == 'Dcp'): label = 'D_{cp}'
elif(obsName == 'D0hp'): label = 'D_{0h^{+}}'
elif(obsName == 'Dint'): label = 'D_{int}'
elif(obsName == 'DL1'): label = 'D_{#Lambda1}'
elif(obsName == 'DL1Zg'): label = 'D_{#Lambda1}_{Zg}'
elif(obsName == 'rapidity4l vs pT4l'):
    obsName_tmp = obsName.split(' vs ')
    obsName = obsName_tmp[0]+"_"+obsName_tmp[1]
    label = '|y_{H}|'
    label_2nd = 'p_{T}^{H} (GeV)'
    doubleDiff = True
elif(obsName == 'Nj vs pT4l'):
    obsName_tmp = obsName.split(' vs ')
    obsName = obsName_tmp[0]+"_"+obsName_tmp[1]
    label = 'N_{jet}'
    label_2nd = 'p_{T}^{H} (GeV)'
    doubleDiff = True
elif(obsName == 'pTj1 vs pTj2'):
    obsName_tmp = obsName.split(' vs ')
    obsName = obsName_tmp[0]+"_"+obsName_tmp[1]
    label = 'p_{T}^{j,1} (GeV)'
    label_2nd = 'p_{T}^{j,2} (GeV)'
    doubleDiff = True
elif(obsName == 'pT4l vs pTHj'):
    obsName_tmp = obsName.split(' vs ')
    obsName = obsName_tmp[0]+"_"+obsName_tmp[1]
    label = 'p_{T}^{H} (GeV)'
    label_2nd = 'p_{T}^{Hj} (GeV)'
    doubleDiff = True
elif(obsName == 'massZ1 vs massZ2'):
    obsName_tmp = obsName.split(' vs ')
    obsName = obsName_tmp[0]+"_"+obsName_tmp[1]
    label = 'm_{Z1} (GeV)'
    label_2nd = 'm_{Z2} (GeV)'
    doubleDiff = True
elif(obsName == 'TCjmax vs pT4l'):
    obsName_tmp = obsName.split(' vs ')
    obsName = obsName_tmp[0]+"_"+obsName_tmp[1]
    label = '#mathscr{T}_{#mathscr{C},{j}} (GeV)'
    label_2nd = 'p_{T}^{H} (GeV)'
    doubleDiff = True
elif(obsName == 'absdetajj vs mjj'):
    obsName_tmp = obsName.split(' vs ')
    obsName = obsName_tmp[0]+"_"+obsName_tmp[1]
    label = '|#Delta#Eta_{jj}|'
    label_2nd =  'm_{jj} (GeV)'
    doubleDiff = True
elif(obsName == 'pT4l vs pTj1'):
    obsName_tmp = obsName.split(' vs ')
    obsName = obsName_tmp[0]+"_"+obsName_tmp[1]
    label = 'p_{T}^{H} (GeV)'
    label_2nd = 'p_{T}^{(j1, 4.7)} (GeV)'
    doubleDiff = True
    

sys.path.append(path['eos_path']+'inputs')
# When using zzfloating, load inputs for the base observable
obsName_for_inputs = obsName.replace('_zzfloating', '') if opt.ZZ else obsName

if opt.INTER:
    _temp = __import__('inputs_sig_extrap_'+obsName_for_inputs+'_'+opt.YEAR, globals(), locals(), ['observableBins']) #, -1)
else:
    _temp = __import__('inputs_sig_'+obsName_for_inputs+'_'+opt.YEAR, globals(), locals(), ['observableBins']) #, -1)

obs_bins = _temp.observableBins
print(obs_bins)
_temp = __import__('xsec_'+obsName_for_inputs+'_'+opt.YEAR, globals(), locals(), ['xsec']) # , -1)
xsec = _temp.xsec

def load_fidxs(process):
    module_obs = obsName_for_inputs + '_zzfloating' if opt.ZZ else obsName_for_inputs
    module_name = 'fidXS_'+module_obs+'_'+process+'_'+opt.YEAR
    module = __import__(module_name, globals(), locals(), ['fidXS'])
    return module.fidXS

fidXS_components = {}
if opt.DO_VBF:
    fidXS_components['VBFH'] = load_fidxs('VBFH')
    fidXS_components['ggH'] = load_fidxs('ggH')
    fidXS_components['VH'] = load_fidxs('VH')
    fidXS_components['ttH'] = load_fidxs('ttH')
sys.path.remove(path['eos_path']+'inputs')

# _poi    = 'SigmaBin'
_obsName = {'pT4l': 'PTH', 'rapidity4l': 'YH', 'pTj1': 'pTj1', 'Nj': 'Nj'}
if obsName not in _obsName:
    _obsName[obsName] = obsName

if opt.ZZ:
    old_obsName = obsName
    obsName += '_zzfloating'
    _obsName[obsName] = _obsName[old_obsName] + '_zzfloating'

raw_nBins = len(obs_bins)
if not doubleDiff: raw_nBins = raw_nBins-1 #in case of 1D measurement the number of bins is -1 the length of the list of bin boundaries
if obsName.startswith("mass4l"): raw_nBins = raw_nBins + 3 #in case of mass4l len(obs_bins)=1, we need to add +3 for cross section in the three different final states

zznorm_indices = None
if opt.ZZ and 'zzfloating' in obsName:
    zznorm_indices = get_zzfloating_merged_bin_indices(obsName, raw_nBins)
    nBins = raw_nBins + len(zznorm_indices)
else:
    nBins = raw_nBins
vbf_special_scan_specs = {}
if opt.DO_VBF:
    if obsName.replace('_zzfloating', '') != 'absdetajj_mjj' or raw_nBins != 4:
        raise RuntimeError('--doVBF expects "absdetajj vs mjj" to have bins 0-3, but found '+str(raw_nBins)+' bins')
    for scan_spec in get_vbf_scan_specs(raw_nBins):
        vbf_special_scan_specs[nBins] = scan_spec
        nBins += 1

if v4_flag: nBins = (len(obs_bins)-1)*2
if v4_flag and doubleDiff: nBins = len(obs_bins)*2
if 'kL' in obsName: nBins = 1

for i in range(nBins):

    print("BIN: ", i)
    _bin = i
    vbf_special_scan = vbf_special_scan_specs.get(_bin)
    vbf_scan_kind = vbf_special_scan['kind'] if vbf_special_scan else None
    is_vbf = vbf_scan_kind == 'vbf'
    is_other_prod = vbf_scan_kind == 'other_prod'
    is_total_minus_vbf = vbf_scan_kind == 'total_minus_vbf'
    is_ggh_extrap = vbf_scan_kind == 'ggh_extrap'
    is_ggh_fixed = vbf_scan_kind == 'ggh_fixed'
    is_vbf_special = is_vbf or is_other_prod or is_total_minus_vbf or is_ggh_extrap or is_ggh_fixed
    vbf_physical_bin = vbf_special_scan['physical_bin'] if vbf_special_scan else None
    vbf_component_xs = None
    vbf_result_label = None
    vbf_plot_label = None

    if is_vbf:
        _obs_bin = 'r_VBFH_'+str(vbf_physical_bin)
        vbf_component_xs = fidXS_components['VBFH'][vbf_physical_bin]
        vbf_result_label = 'VBFH'
        vbf_plot_label = 'VBF'
    elif is_other_prod:
        _obs_bin = 'r_otherProd_'+str(vbf_physical_bin)
        vbf_component_xs = xsec['SigmaBin'+str(vbf_physical_bin)] - fidXS_components['VBFH'][vbf_physical_bin]
        vbf_result_label = 'otherProd'
        vbf_plot_label = 'other prod.'
    elif is_total_minus_vbf:
        _obs_bin = 'r_totalMinusVBF_'+str(vbf_physical_bin)
        vbf_component_xs = xsec['SigmaBin'+str(vbf_physical_bin)] - fidXS_components['VBFH'][vbf_physical_bin]
        vbf_result_label = 'totalMinusVBF'
        vbf_plot_label = 'total-VBF'
    elif is_ggh_extrap:
        _obs_bin = 'r_VBFH_ggHExtrap_'+str(vbf_physical_bin)
        vbf_component_xs = fidXS_components['VBFH'][vbf_physical_bin]
        vbf_result_label = 'VBFH_ggHExtrap'
        vbf_plot_label = 'VBF, ggH extrap.'
    elif is_ggh_fixed:
        _obs_bin = 'r_VBFH_ggHFixed_'+str(vbf_physical_bin)
        vbf_component_xs = fidXS_components['VBFH'][vbf_physical_bin]
        vbf_result_label = 'VBFH_ggHFixed'
        vbf_plot_label = 'VBF, ggH fixed'
    elif opt.ZZ and zznorm_indices is not None:
        if _bin < raw_nBins:
            _obs_bin = _poi+str(i)
        else:
            zz_bin = zznorm_indices[_bin - raw_nBins]
            _obs_bin = 'zz_norm_'+str(zz_bin)
    else:
        _obs_bin = _poi+str(i)

    if obsName.startswith("mass4l"):
        if _bin == 1:
            _obs_bin = 'r2e2muBin0'
        if _bin == 2:
            _obs_bin = 'r4muBin0'
        if _bin == 3:
            _obs_bin = 'r4eBin0'
        if _bin == 4:
            _obs_bin = 'zz_norm_0'
        if _bin == 5:
            _obs_bin = 'zz_norm_0_4e'
        if _bin == 6:
            _obs_bin = 'zz_norm_0_4mu'
        if _bin == 7:
            _obs_bin = 'zz_norm_0_2e2mu'

    if v4_flag:
        if (_bin % 2) == 0:
            _obs_bin = 'r2e2muBin'+str(i//2)
            print("v4_flag - _obs_bin",_obs_bin,str(i//2), str(i))
        else:
            _obs_bin = 'r4lBin'+str((i-1)//2)

    if 'kL' in obsName:
            _obs_bin = 'kappa_lambda'


    graphs = []
    grapherrs = []
    grapherrslow = []

    scan_fileList = fileList
    scan_titles = titles
    scan_colors = colors
    scan_idx_max = idx_max
    has_observed_scan = opt.UNBLIND

    if is_vbf_special:
        scan_fileList = []
        scan_titles = []
        scan_colors = []
        for ifile, file_template in enumerate(fileList):
            rfile = file_template.replace('OBS', _obs_bin)
            rfile = rfile.replace('BIN', obsName)
            fname = inputPath+rfile
            if tfile_is_good(fname):
                scan_fileList.append(file_template)
                scan_titles.append(titles[ifile])
                scan_colors.append(colors[ifile])
            else:
                print('Skipping missing optional scan file:', fname)
        if len(scan_fileList) == 0:
            print('No optional scan files found for bin '+str(vbf_physical_bin)+'. Skipping plot.')
            continue
        scan_idx_max = min(idx_max, len(scan_fileList)-1)
        has_observed_scan = any(('.123456.root' not in f) for f in scan_fileList)

    for ifile in range(len(scan_fileList)):
        rfile = scan_fileList[ifile].replace('OBS', _obs_bin)
        rfile = rfile.replace('BIN', obsName)
        fname = inputPath+rfile
        if not tfile_is_good(fname):
            print('Skipping missing scan file:', fname)
            continue
        inF = TFile.Open(fname,"READ")
        if not inF:
            print('Skipping unreadable scan file:', fname)
            continue
        tree = inF.Get("limit")
        if not tree:
            print('Skipping scan file without limit tree:', fname)
            inF.Close()
            continue
        graph = TGraph()

        if tree.GetBranch('r_smH_'+_obsName[obsName]+'_'+str(_bin)):
            tree.GetBranch('r_smH_'+_obsName[obsName]+'_'+str(_bin)).SetTitle('r_smH_'+str(_bin)+'/F');
            tree.GetBranch('r_smH_'+_obsName[obsName]+'_'+str(_bin)).SetName('r_smH_'+str(_bin)+'');

        
        ipoint = 0
        for entry in tree :
            if((2*entry.deltaNLL<5)):
                yval = 2.0 * entry.deltaNLL
                field = None

                base_nbins = raw_nBins if opt.ZZ else nBins

                if obsName.startswith("mass4l"):
                    mass4l_fields = {
                        0: "r_smH_0",
                        1: "r2e2muBin0",
                        2: "r4muBin0",
                        3: "r4eBin0",
                        4: "zz_norm_0",
                        5: "zz_norm_0_4e",
                        6: "zz_norm_0_4mu",
                        7: "zz_norm_0_2e2mu",
                    }
                    field = mass4l_fields.get(_bin)

                elif v4_flag:
                    prefix = "r2e2muBin" if _bin % 2 == 0 else "r4lBin"
                    index = _bin // 2
                    field = f"{prefix}{index}"

                elif "kL" in obsName and _bin == 0:
                    field = "kappa_lambda"

                elif is_vbf_special:
                    field = _obs_bin.replace(str(vbf_physical_bin), _obsName[obsName]+'_'+str(vbf_physical_bin), 1)

                else:
                    if opt.ZZ and _bin >= base_nbins and zznorm_indices is not None:
                        zz_bin = zznorm_indices[_bin - base_nbins]
                        field = f"zz_norm_{zz_bin}"
                    else:
                        field = f"r_smH_{_bin}"

                if field is not None:
                    if not hasattr(entry, field):
                        continue
                    graph.SetPoint(ipoint, getattr(entry, field), yval)
                    ipoint += 1
        inF.Close()
        if graph.GetN() < 2:
            print('Skipping scan with too few points:', fname)
            continue
        graphs.append(graph)
        scan_titles[len(graphs)-1] = scan_titles[ifile]
        scan_colors[len(graphs)-1] = scan_colors[ifile]

    scan_titles = scan_titles[:len(graphs)]
    scan_colors = scan_colors[:len(graphs)]
    if len(graphs) == 0:
        print('No usable scan files found for bin '+str(_bin)+'. Skipping plot.')
        continue
    scan_idx_max = min(scan_idx_max, len(graphs)-1)

    c=TCanvas("c", "c", 1000, 800)
    c.SetLeftMargin(0.14)
    c.SetRightMargin(0.08)
    c.cd()
    gStyle.SetOptTitle(0)

    if obsName == 'mass4l':
        x = np.array(graphs[0].GetX())
        y = np.array(graphs[0].GetY())
        for entry in range(graphs[0].GetN()):
            graphs[0].SetPoint(entry, x[entry], y[entry])

    graphs[0].SetLineColor(scan_colors[0])
    graphs[0].SetLineWidth(3)
    graphs[0].Sort()
    graphs[0].SetTitle(scan_titles[0])
    graphs[0].SetFillColor(scan_colors[0])
    graphs[0].SetFillStyle(3000)

    mini = graphs[scan_idx_max].GetXaxis().GetXmin()
    maxi = graphs[scan_idx_max].GetXaxis().GetXmax()
    if mini > graphs[0].GetXaxis().GetXmin():
        mini = graphs[0].GetXaxis().GetXmin()
    if maxi < graphs[0].GetXaxis().GetXmax():
        maxi = graphs[0].GetXaxis().GetXmax()
    maxY = 8.

    # if obsName == 'mass4l':
    #     mini = 2.1
    #     maxi = 3.7

    graphs[0].Draw("AC")
    if v4_flag:
        if _bin == 0: xtitle = "#sigma_{bin 2e2mu 0}"
        elif _bin == 1: xtitle = "#sigma_{bin 4l 0}"
        elif _bin == 2: xtitle = "#sigma_{bin 2e2mu 1}"
        elif _bin == 3: xtitle = "#sigma_{bin 4l 1}"
        elif _bin == 4: xtitle = "#sigma_{bin 2e2mu 2}"
        elif _bin == 5: xtitle = "#sigma_{bin 4l 2}"
        elif _bin == 6: xtitle = "#sigma_{bin 2e2mu 3}"
        elif _bin == 7: xtitle = "#sigma_{bin 4l 3}"
        elif _bin == 8: xtitle = "#sigma_{bin 2e2mu 4}"
        elif _bin == 9: xtitle = "#sigma_{bin 4l 4}"
        elif _bin == 10: xtitle = "#sigma_{bin 2e2mu 5}"
        elif _bin == 11: xtitle = "#sigma_{bin 4l 5}"
        elif _bin == 12: xtitle = "#sigma_{bin 2e2mu 6}"
        elif _bin == 13: xtitle = "#sigma_{bin 4l 6}"
        elif _bin == 14: xtitle = "#sigma_{bin 2e2mu 7}"
        elif _bin == 15: xtitle = "#sigma_{bin 4l 7}"
        elif _bin == 16: xtitle = "#sigma_{bin 2e2mu 8}"
        elif _bin == 17: xtitle = "#sigma_{bin 4l 8}"
        elif _bin == 18: xtitle = "#sigma_{bin 2e2mu 9}"
        elif _bin == 19: xtitle = "#sigma_{bin 4l 9}"
        elif _bin == 20: xtitle = "#sigma_{bin 2e2mu 10}"
        elif _bin == 21: xtitle = "#sigma_{bin 4l 10}"
    elif is_vbf_special:
        xtitle = "#sigma_{"+vbf_plot_label+"} (fb)"
    elif 'kL' in obsName:
        xtitle = "#kappa_{#lambda}"
    elif obsName == 'mass4l' or obsName == 'mass4l_zzfloating':
        if _bin == 0: xtitle = "#sigma_{incl} (fb)"
        elif _bin == 1: xtitle = "#sigma_{2e2mu}"
        elif _bin == 2: xtitle = "#sigma_{4mu}"
        elif _bin == 3: xtitle = "#sigma_{4e}"
        elif _bin == 4: xtitle = "ZZ_{norm}"
        elif _bin == 5: xtitle = "ZZ_{norm}^{4e}"
        elif _bin == 6: xtitle = "ZZ_{norm}^{4mu}"
        elif _bin == 7: xtitle = "ZZ_{norm}^{2e2mu}"
    elif 'zzfloating' in obsName:
        if _bin < raw_nBins:
            xtitle = "r_{" + str(_bin) + "}"
        else:
            zz_bin = _bin - raw_nBins
            xtitle = "ZZ_{norm}^{" + str(zz_bin) + "}"
    else:
        xtitle = "r_{" + str(_bin) + "}"
    graphs[0].GetXaxis().SetTitle(xtitle)
    graphs[0].GetYaxis().SetTitle("-2#Delta ln L")
    graphs[0].GetYaxis().SetTitleOffset(0.9)
    graphs[0].GetXaxis().SetTitleOffset(0.72)
    graphs[0].GetXaxis().SetTitleSize(0.059)
    graphs[0].GetYaxis().SetTitleSize(0.059)

    graphs[0].GetYaxis().SetRangeUser(0,maxY)
    graphs[0].GetXaxis().SetRangeUser(mini,maxi)
    c.Modified()

    graphs[0].GetYaxis().SetLimits(0,maxY)
    graphs[0].GetXaxis().SetLimits(mini,maxi)
    c.Modified()
    c.Update()

    for ig in range(1,len(graphs)) :
        graphs[ig].SetLineColor(scan_colors[ig])
        graphs[ig].SetFillColor(scan_colors[ig])
        graphs[ig].SetFillStyle(3000)
        graphs[ig].SetLineWidth(3)
        if 'stat-only' in scan_titles[ig]:
            graphs[ig].SetLineWidth(2)
            graphs[ig].SetLineStyle(2)
        graphs[ig].SetTitle(scan_titles[ig])
        graphs[ig].Sort()

        if obsName == 'mass4l':
            x = np.array(graphs[ig].GetX())
            y = np.array(graphs[ig].GetY())
            for entry in range(graphs[ig].GetN()):
                graphs[ig].SetPoint(entry, x[entry], y[entry])

        graphs[ig].Draw("CSAME")

    lineone = TLine(mini,1,maxi,1)
    linetwo = TLine(mini,3.85,maxi,3.85)
    lineone.SetLineColor(15)#kGray+1)
    linetwo.SetLineColor(15)#kGray+1)
    lineone.Draw("SAME")
    linetwo.Draw("SAME")

    leg = TLegend(0.62,0.88,0.9,0.72)
    leg.SetLineColor(0)
    leg.SetLineStyle(0)
    leg.SetLineWidth(0)
    #leg.SetFillStyle(0)
    leg.SetShadowColor(10)
    leg.SetTextSize(0.034)
    leg.SetTextFont(42)
    for ip in range(0, len(graphs)):
        leg.AddEntry(graphs[ip], scan_titles[ip], "l")

    leg.Draw("SAME")

    if is_vbf:
        poi = 'r_VBFH_'+_obsName[obsName]+'_'+str(vbf_physical_bin)
        poi_fn = 'r_VBFH_'+str(vbf_physical_bin)
    elif is_other_prod:
        poi = 'r_otherProd_'+_obsName[obsName]+'_'+str(vbf_physical_bin)
        poi_fn = 'r_otherProd_'+str(vbf_physical_bin)
    elif is_total_minus_vbf:
        poi = 'r_totalMinusVBF_'+_obsName[obsName]+'_'+str(vbf_physical_bin)
        poi_fn = 'r_totalMinusVBF_'+str(vbf_physical_bin)
    elif is_ggh_extrap:
        poi = 'r_VBFH_ggHExtrap_'+_obsName[obsName]+'_'+str(vbf_physical_bin)
        poi_fn = 'r_VBFH_ggHExtrap_'+str(vbf_physical_bin)
    elif is_ggh_fixed:
        poi = 'r_VBFH_ggHFixed_'+_obsName[obsName]+'_'+str(vbf_physical_bin)
        poi_fn = 'r_VBFH_ggHFixed_'+str(vbf_physical_bin)
    elif 'smH' in _obs_bin:
        poi = 'r_smH_'+_obsName[obsName]+'_'+str(i)
        poi_fn = 'r_smH_'+str(i)
    else:
        poi = _obs_bin
        poi_fn = poi
    if 'kL' in obsName:
        fname = inputPath + "higgsCombine_"+obsName+".MultiDimFit.mH125.38.123456.root"
        if tfile_is_good(fname):
            goodFile = TFile(fname)
        else:
            print('Skipping missing expected kL scan:', fname)
            continue
        limit = goodFile.Get('limit')
        kappa_lambda = []
        for entry in limit:
            kappa_lambda.append(entry.kappa_lambda)
        print(kappa_lambda)
        exp_nom = []
        exp_nom.append(kappa_lambda[0])
        exp_nom.append(kappa_lambda[2]-kappa_lambda[0])
        exp_nom.append(kappa_lambda[0]-kappa_lambda[1])

        fname = inputPath + "higgsCombine_pT4l_kL_grid.MultiDimFit.mH125.38.123456.root"
        obs_scan = BuildScan('scan', poi, [fname], 2, yvals, 7.)
        if obs_scan is None:
            print('Skipping expected kL 95% CL extraction for missing scan:', fname)
            continue
        obs_2sig = obs_scan['val_2sig']
        print('------------------------------------------------------')
        print('EXPECTED 95% CL exclusion:', obs_2sig[0]+obs_2sig[1], obs_2sig[0]+obs_2sig[2])
        print('------------------------------------------------------')
    else:
        fname = inputPath + "higgsCombine_"+obsName+"_"+poi_fn+".MultiDimFit.mH125.38.123456.root"
        print('STAT+SYST')
        exp_scan = BuildScan('scan', poi, [fname], 2, yvals, 7.)
        if exp_scan is None or exp_scan['val'] is None:
            print('Skipping bin '+str(_bin)+' because expected stat+syst scan is missing or unusable:', fname)
            continue
        exp_nom = exp_scan['val']
        # exp_2sig = exp_scan['val_2sig']

    if 'kL' in obsName:
        fname = inputPath + "higgsCombine_"+obsName+"_NoSys.MultiDimFit.mH125.38.123456.root"
        if tfile_is_good(fname):
            goodFile = TFile(fname)
        else:
            print('Skipping bin '+str(_bin)+' because expected kL stat-only scan is missing:', fname)
            continue
        limit = goodFile.Get('limit')
        kappa_lambda = []
        for entry in limit:
            kappa_lambda.append(entry.kappa_lambda)
        exp_nom_stat = []
        exp_nom_stat.append(kappa_lambda[0])
        exp_nom_stat.append(kappa_lambda[2]-kappa_lambda[0])
        exp_nom_stat.append(kappa_lambda[0]-kappa_lambda[1])
    else:
        fname = inputPath + "higgsCombine_"+obsName+"_"+poi_fn+"_NoSys.MultiDimFit.mH125.38.123456.root"
        print('STAT-ONLY')
        exp_scan_stat = BuildScan('scan', poi, [fname], 2, yvals, 7.)
        if exp_scan_stat is None or exp_scan_stat['val'] is None:
            print('Skipping bin '+str(_bin)+' because expected stat-only scan is missing or unusable:', fname)
            continue
        exp_nom_stat = exp_scan_stat['val']
        # exp_2sig_stat = exp_scan_stat['val_2sig']

    exp_up_sys = quadrature_subtract(exp_nom[1], exp_nom_stat[1])
    exp_do_sys = quadrature_subtract(exp_nom[2], exp_nom_stat[2])

    if opt.UNBLIND and has_observed_scan:
        if 'kL' in obsName:
            observed_required_files = [
                inputPath + "higgsCombine_"+obsName+".MultiDimFit.mH125.38.root",
                inputPath + "higgsCombine_"+obsName+"_NoSys.MultiDimFit.mH125.38.root",
            ]
        else:
            observed_required_files = [
                inputPath + "higgsCombine_"+obsName+"_"+poi_fn+".MultiDimFit.mH125.38.root",
                inputPath + "higgsCombine_"+obsName+"_"+poi_fn+"_NoSys.MultiDimFit.mH125.38.root",
            ]
        missing_observed_files = [fname for fname in observed_required_files if not tfile_is_good(fname)]
        if missing_observed_files:
            print('Skipping observed scan for bin '+str(_bin)+' because file(s) are missing:', ', '.join(missing_observed_files))
            has_observed_scan = False

    if (opt.UNBLIND and has_observed_scan):
        if 'kL' in obsName:
            fname = inputPath + "higgsCombine_"+obsName+".MultiDimFit.mH125.38.root"
            if tfile_is_good(fname):
                goodFile = TFile(fname)
            else:
                print('Skipping observed scan for missing file:', fname)
                has_observed_scan = False
                obs_nom = None
                obs_nom_stat = None
                obs_up_sys = None
                obs_do_sys = None
                continue
            limit = goodFile.Get('limit')
            kappa_lambda = []
            for entry in limit:
                kappa_lambda.append(entry.kappa_lambda)
            print(kappa_lambda)
            obs_nom = []
            obs_nom.append(kappa_lambda[0])
            obs_nom.append(kappa_lambda[2]-kappa_lambda[0])
            obs_nom.append(kappa_lambda[0]-kappa_lambda[1])

            fname = inputPath + "higgsCombine_pT4l_kL_grid.MultiDimFit.mH125.38.root"
            obs_scan = BuildScan('scan', poi, [fname], 2, yvals, 7.)
            if obs_scan is None:
                print('Skipping observed kL 95% CL extraction for missing scan:', fname)
                has_observed_scan = False
                continue
            obs_2sig = obs_scan['val_2sig']
            print('------------------------------------------------------')
            print('OBSERVED 95% CL exclusion:', obs_2sig[0]+obs_2sig[1], obs_2sig[0]+obs_2sig[2])
            print('------------------------------------------------------')
        else:
            fname = inputPath + "higgsCombine_"+obsName+"_"+poi_fn+".MultiDimFit.mH125.38.root"
            print('STAT+SYST')
            print(fname)
            obs_scan = BuildScan('scan', poi, [fname], 2, yvals, 7.)
            if obs_scan is None or obs_scan['val'] is None:
                print('Skipping observed scan for bin '+str(_bin)+' because file is missing or unusable:', fname)
                has_observed_scan = False
                continue
            # print obs_scan
            obs_nom = obs_scan['val']
            obs_2sig = obs_scan['val_2sig']

        if 'kL' in obsName:
            fname = inputPath + "higgsCombine_"+obsName+"_NoSys.MultiDimFit.mH125.38.root"
            if tfile_is_good(fname):
                goodFile = TFile(fname)
            else:
                print('Skipping observed scan for missing kL stat-only file:', fname)
                has_observed_scan = False
                continue
            limit = goodFile.Get('limit')
            kappa_lambda = []
            for entry in limit:
                kappa_lambda.append(entry.kappa_lambda)
            print(kappa_lambda)
            obs_nom_stat = []
            obs_nom_stat.append(kappa_lambda[0])
            obs_nom_stat.append(kappa_lambda[2]-kappa_lambda[0])
            obs_nom_stat.append(kappa_lambda[0]-kappa_lambda[1])
        else:
            fname = inputPath + "higgsCombine_"+obsName+"_"+poi_fn+"_NoSys.MultiDimFit.mH125.38.root"
            print('STAT-ONLY')
            obs_scan_stat = BuildScan('scan', poi, [fname], 2, yvals, 7.)
            if obs_scan_stat is None or obs_scan_stat['val'] is None:
                print('Skipping observed stat-only scan for bin '+str(_bin)+' because file is missing or unusable:', fname)
                has_observed_scan = False
                continue
            obs_nom_stat = obs_scan_stat['val']
            obs_2sig_stat = obs_scan_stat['val_2sig']

        obs_up_sys = quadrature_subtract(obs_nom[1], obs_nom_stat[1])
        obs_do_sys = quadrature_subtract(obs_nom[2], obs_nom_stat[2])

    #For v3 model we multiply by the expected th xs
    if 'smH' in _obs_bin:
        exp_nom = list(exp_nom)
        exp_nom_stat = list(exp_nom_stat)
        exp_nom[0] *= xsec['SigmaBin'+str(i)]
        exp_nom[1] *= xsec['SigmaBin'+str(i)]
        exp_nom[2] *= xsec['SigmaBin'+str(i)]
        exp_nom_stat[1] *= xsec['SigmaBin'+str(i)]
        exp_nom_stat[2] *= xsec['SigmaBin'+str(i)]
        exp_up_sys *= xsec['SigmaBin'+str(i)]
        exp_do_sys *= xsec['SigmaBin'+str(i)]

        if opt.UNBLIND and has_observed_scan:
            obs_nom = list(obs_nom)
            obs_nom_stat = list(obs_nom_stat)
            obs_nom[0] *= xsec['SigmaBin'+str(i)]
            obs_nom[1] *= xsec['SigmaBin'+str(i)]
            obs_nom[2] *= xsec['SigmaBin'+str(i)]
            obs_nom_stat[1] *= xsec['SigmaBin'+str(i)]
            obs_nom_stat[2] *= xsec['SigmaBin'+str(i)]
            obs_up_sys *= xsec['SigmaBin'+str(i)]
            obs_do_sys *= xsec['SigmaBin'+str(i)]

    if is_vbf_special:
        exp_nom = list(exp_nom)
        exp_nom_stat = list(exp_nom_stat)
        exp_nom[0] *= vbf_component_xs
        exp_nom[1] *= vbf_component_xs
        exp_nom[2] *= vbf_component_xs
        exp_nom_stat[1] *= vbf_component_xs
        exp_nom_stat[2] *= vbf_component_xs
        exp_up_sys *= vbf_component_xs
        exp_do_sys *= vbf_component_xs

        if opt.UNBLIND and has_observed_scan:
            obs_nom = list(obs_nom)
            obs_nom_stat = list(obs_nom_stat)
            obs_nom[0] *= vbf_component_xs
            obs_nom[1] *= vbf_component_xs
            obs_nom[2] *= vbf_component_xs
            obs_nom_stat[1] *= vbf_component_xs
            obs_nom_stat[2] *= vbf_component_xs
            obs_up_sys *= vbf_component_xs
            obs_do_sys *= vbf_component_xs

    if(opt.UNBLIND and has_observed_scan):
        Text3 = TPaveText(0.15, 0.81,0.4,0.9,'brNDC')
    else:
    	Text3 = TPaveText(0.15, 0.76,0.4,0.84,'bfNDC')

    plot_bin = _bin
    is_zz = False
    if is_vbf_special:
        plot_bin = vbf_physical_bin
    if opt.ZZ and not is_vbf_special:
        base_nbins = raw_nBins
        if _bin >= base_nbins:
            plot_bin = _bin - base_nbins
            is_zz = True

    if v4_flag:
        phys_bin = plot_bin // 2
        channel = "2e2mu" if plot_bin % 2 == 0 else "4l"
        if is_zz:
            exp_fit = 'Exp. ZZ_{norm, %s, %d} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (channel, phys_bin, exp_nom[0], exp_nom_stat[1], abs(exp_nom_stat[2]), exp_up_sys, exp_do_sys)
        else:
            exp_fit = 'Exp. #sigma_{%s, %d} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (channel, phys_bin, exp_nom[0], exp_nom_stat[1], abs(exp_nom_stat[2]), exp_up_sys, exp_do_sys)

    elif 'mass4l' in obsName:
        mass4l_labels = {0:'incl',1:'2e2mu',2:'4mu',3:'4e',4:'ZZ_{norm}',5:'ZZ_{norm}^{4e}',6:'ZZ_{norm}^{4mu}',7:'ZZ_{norm}^{2e2mu}'}
        label_name = mass4l_labels.get(_bin, str(_bin))
        if label_name.startswith('ZZ'):
            exp_fit = 'Exp. %s = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (label_name, exp_nom[0], exp_nom_stat[1], abs(exp_nom_stat[2]), exp_up_sys, exp_do_sys)
        else:
            exp_fit = 'Exp. #sigma_{%s} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (label_name, exp_nom[0], exp_nom_stat[1], abs(exp_nom_stat[2]), exp_up_sys, exp_do_sys)

    elif 'kL' in obsName:
        exp_fit = 'Exp. #kappa_{#lambda} = %.1f^{#plus %.1f}_{#minus %.1f} (stat)^{#plus %.1f}_{#minus %.1f} (syst)' % (exp_nom[0], exp_nom_stat[1], abs(exp_nom_stat[2]), exp_up_sys, exp_do_sys)

    else:
        if is_zz:
            exp_fit = 'Exp. ZZ_{norm, %d} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (plot_bin, exp_nom[0], exp_nom_stat[1], abs(exp_nom_stat[2]), exp_up_sys, exp_do_sys)
        elif is_vbf_special:
            exp_fit = 'Exp. #sigma_{%s} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (vbf_plot_label, exp_nom[0], exp_nom_stat[1], abs(exp_nom_stat[2]), exp_up_sys, exp_do_sys)
        else:
            exp_fit = 'Exp. #sigma_{%d} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (plot_bin, exp_nom[0], exp_nom_stat[1], abs(exp_nom_stat[2]), exp_up_sys, exp_do_sys)
            
    Text3.SetTextAlign(12);
    Text3.SetTextSize(0.038)
    Text3.AddText(exp_fit)
    Text3.SetFillStyle(0)
    Text3.SetLineStyle(0)
    Text3.SetBorderSize(0)
    Text3.Draw()

    if(opt.UNBLIND and has_observed_scan):
        Text4 = TPaveText(0.15, 0.71,0.4,0.8,'brNDC')
        if 'kL' in obsName:
            obs_fit = 'Obs. #kappa_{#lambda} = %.1f^{#plus %.1f}_{#minus %.1f} (stat)^{#plus %.1f}_{#minus %.1f} (syst)' % (obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
        elif 'mass4l' in obsName:
            if _bin == 0: obs_fit = 'Obs. #sigma_{incl} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
            if _bin == 1: obs_fit = 'Obs. #sigma_{2e2mu} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
            if _bin == 2: obs_fit = 'Obs. #sigma_{4mu} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
            if _bin == 3: obs_fit = 'Obs. #sigma_{4e} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
            if _bin == 4: obs_fit = 'Obs. ZZ_{norm} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
            if _bin == 5: obs_fit = 'Obs. ZZ_{norm}^{4e} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
            if _bin == 6: obs_fit = 'Obs. ZZ_{norm}^{4mu} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
            if _bin == 7: obs_fit = 'Obs. ZZ_{norm}^{2e2mu} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
        elif is_zz:
            obs_fit = 'Obs. ZZ_{norm, %d} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (plot_bin, obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
        elif is_vbf_special:
            obs_fit = 'Obs. #sigma_{%s} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (vbf_plot_label, obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
        else:
            obs_fit = 'Obs. #sigma_{%d} = %.2f^{#plus %.2f}_{#minus %.2f} (stat)^{#plus %.2f}_{#minus %.2f} (syst)' % (plot_bin, obs_nom[0], obs_nom_stat[1], abs(obs_nom_stat[2]), obs_up_sys, obs_do_sys)
        Text4.SetTextAlign(12);
        Text4.SetTextSize(0.038)
        Text4.AddText(obs_fit)
        Text4.SetFillStyle(0)
        Text4.SetLineStyle(0)
        Text4.SetBorderSize(0)
        Text4.Draw()

    Text = TPaveText(0.58, 0.88,0.93,0.95,'brNDC')
    #Text.SetNDC()
    Text.SetTextAlign(31); #31
    Text.SetTextSize(0.5*c.GetTopMargin())
    leftText = "CMS"
    re = "#bf{%s fb^{-1} (13.6 TeV)}" %(_lumi)
    Text.AddText(re)
    Text.SetFillStyle(0)
    Text.SetLineStyle(0)
    Text.SetBorderSize(0)
    Text.Draw()

    latex2 = TLatex()
    latex2.SetNDC()
    latex2.SetTextSize(0.6*c.GetTopMargin())
    latex2.SetTextFont(52)
    latex2.SetTextAlign(11)
    #latex2.DrawLatex(0.28, 0.945, "Unpublished")
    # latex2.DrawLatex(0.25, 0.915, "Preliminary")


    Text2 = TPaveText(0.25, 0.88,0.15,0.95,'brNDC')
    Text2.SetTextAlign(31);
    Text2.SetTextSize(0.065)
    Text2.AddText(leftText)
    Text2.SetFillStyle(0)
    Text2.SetLineStyle(0)
    Text2.SetBorderSize(0)
    Text2.Draw()

    latex2 = TLatex()
    latex2.SetNDC()
    latex2.SetTextSize(0.04)
    latex2.SetTextFont(42)
    latex2.SetTextAlign(31) # align right

    # Map plotted bin back to the physical observable bin
    plot_bin = _bin
    is_zz = False
    if is_vbf_special:
        plot_bin = vbf_physical_bin

    if opt.ZZ and not is_vbf_special:
        base_nbins = raw_nBins
        if plot_bin >= base_nbins:
            is_zz = True
            plot_bin -= base_nbins

    if 'jet' in obsName and not doubleDiff:
        latex2.DrawLatex(0.45, 0.65, f"{plot_bin} jet(s)")

    else:
        line1 = None
        line2 = None
        plot_lines = []

        if ('pTj1' in obsName) and not doubleDiff:
            if is_zz:
                # For merged ZZ bins, show all original bins that were merged
                obsName_base = obsName.replace('_zzfloating', '')
                merged_bins = get_merged_bins_for_zz(obsName_base, plot_bin)
                if merged_bins:
                    first_bin = min(merged_bins)
                    last_bin = max(merged_bins)
                    line1 = f"{obs_bins[first_bin]} < {label} < {obs_bins[last_bin+1]}"
                else:
                    line1 = f"{obs_bins[plot_bin]} < {label} < {obs_bins[plot_bin+1]}"
            else:
                line1 = f"{obs_bins[plot_bin]} < {label} < {obs_bins[plot_bin+1]}"
            x = 0.5

        elif obsName.startswith("mass4l"):
            line1 = ""
            x = 0.5

        elif doubleDiff and not v4_flag:
            if is_zz:
                obsName_base = obsName.replace('_zzfloating', '')
                merged_bins = get_merged_bins_for_zz(obsName_base, plot_bin)
                if merged_bins:
                    plot_lines = [format_double_diff_bin_line(obs_bins[merged_bin], label, label_2nd) for merged_bin in merged_bins]
                else:
                    plot_lines = [format_double_diff_bin_line(obs_bins[plot_bin], label, label_2nd)]
            else:
                plot_lines = [format_double_diff_bin_line(obs_bins[plot_bin], label, label_2nd)]
            x = 0.5

        elif doubleDiff and v4_flag:
            phys_bin = plot_bin // 2
            plot_lines = [format_double_diff_bin_line(obs_bins[phys_bin], label, label_2nd)]
            x = 0.5

        elif 'kL' in obsName:
            line1 = ""
            x = 0.5

        elif v4_flag:
            phys_bin = plot_bin // 2
            line1 = f"{obs_bins[phys_bin]} < {label} < {obs_bins[phys_bin+1]}"
            x = 0.5

        else:
            if is_zz:
                # For merged ZZ bins, show all original bins that were merged
                obsName_base = obsName.replace('_zzfloating', '')
                merged_bins = get_merged_bins_for_zz(obsName_base, plot_bin)
                if merged_bins:
                    first_bin = min(merged_bins)
                    last_bin = max(merged_bins)
                    line1 = f"{obs_bins[first_bin]} < {label} < {obs_bins[last_bin+1]}"
                else:
                    line1 = f"{obs_bins[plot_bin]} < {label} < {obs_bins[plot_bin+1]}"
            else:
                line1 = f"{obs_bins[plot_bin]} < {label} < {obs_bins[plot_bin+1]}"
            x = 0.5

        if plot_lines:
            for iline, line in enumerate(plot_lines):
                latex2.DrawLatex(x, 0.65 - 0.05 * iline, line)
        else:
            if line1 is not None:
                latex2.DrawLatex(x, 0.65, line1)
            if line2 is not None:
                latex2.DrawLatex(x, 0.60, line2)

    latex2.DrawLatex(0.995,0.21, "#scale[0.7]{#color[12]{68% CL}}")
    latex2.DrawLatex(0.995,0.49, "#scale[0.7]{#color[12]{95% CL}}")

    graphs[0].GetYaxis().SetRangeUser(0,maxY)
    graphs[0].GetXaxis().SetRangeUser(mini,maxi)
    c.Modified()
    graphs[0].GetYaxis().SetLimits(0,maxY)
    graphs[0].GetXaxis().SetLimits(mini,maxi)
    c.Modified()
    c.Update()

    if(opt.UNBLIND and has_observed_scan):
        if v4_flag:
            if _bin==0:
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==1:
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==2:
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin1'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin1_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==3:
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin1'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin1_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==4:
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin2'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin2_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==5:
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin2'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin2_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==6:
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin3'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin3_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==7:
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin3'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin3_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==8:
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin4'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin4_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==9:
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin4'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin4_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==10:
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin5'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin5_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==11:
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin5'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin5_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==12:
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin6'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin6_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==13:
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin6'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin6_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==14:
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin7'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin7_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==15:
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin7'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin7_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==16:
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin8'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin8_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==17:
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin8'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin8_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==18:
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin9'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_2e2mu_genbin5_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==19:
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin9'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v4['SM_125_'+obsName+'_4l_genbin5_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
        elif not obsName.startswith("mass4l"):
            if is_vbf_special:
                resultsXS_data['SM_125_'+obsName+'_'+vbf_result_label+'_genbin'+str(vbf_physical_bin)] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data['SM_125_'+obsName+'_'+vbf_result_label+'_genbin'+str(vbf_physical_bin)+'_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            else:
                resultsXS_data['SM_125_'+obsName+'_genbin'+str(i)] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data['SM_125_'+obsName+'_genbin'+str(i)+'_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
                if 'zzfloating' in obsName and i >= raw_nBins:
                    resultsXS_data['SM_125_'+obsName+'_zznorm_genbin'+str(zz_bin)] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                    resultsXS_data['SM_125_'+obsName+'_zznorm_genbin'+str(zz_bin)+'_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
        elif obsName == "mass4l_zzfloating":
            if _bin==0:
                resultsXS_data['SM_125_'+obsName+'_genbin'+str(i)] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data['SM_125_'+obsName+'_genbin'+str(i)+'_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==1:
                resultsXS_data_v2['SM_125_'+obsName+'_2e2mu_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v2['SM_125_'+obsName+'_2e2mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==2:
                resultsXS_data_v2['SM_125_'+obsName+'_4mu_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v2['SM_125_'+obsName+'_4mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==3:
                resultsXS_data_v2['SM_125_'+obsName+'_4e_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v2['SM_125_'+obsName+'_4e_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==4:
                resultsXS_data_v2['SM_125_'+obsName+'_zznorm_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v2['SM_125_'+obsName+'_zznorm_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==5:
                resultsXS_data_v2['SM_125_'+obsName+'_zznorm4e_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v2['SM_125_'+obsName+'_zznorm4e_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==6:
                resultsXS_data_v2['SM_125_'+obsName+'_zznorm4mu_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v2['SM_125_'+obsName+'_zznorm4mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==7:
                resultsXS_data_v2['SM_125_'+obsName+'_zznorm2e2mu_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v2['SM_125_'+obsName+'_zznorm2e2mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
        else:
            if _bin==0:
                resultsXS_data['SM_125_'+obsName+'_genbin'+str(i)] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data['SM_125_'+obsName+'_genbin'+str(i)+'_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==1:
                resultsXS_data_v2['SM_125_'+obsName+'_2e2mu_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v2['SM_125_'+obsName+'_2e2mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==2:
                resultsXS_data_v2['SM_125_'+obsName+'_4mu_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v2['SM_125_'+obsName+'_4mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            elif _bin==3:
                resultsXS_data_v2['SM_125_'+obsName+'_4e_genbin0'] = {"uncerDn": -1.0*abs(obs_nom[2]), "uncerUp": obs_nom[1], "central": obs_nom[0]}
                resultsXS_data_v2['SM_125_'+obsName+'_4e_genbin0_statOnly'] = {"uncerDn": -1.0*abs(obs_nom_stat[2]), "uncerUp": obs_nom_stat[1], "central": obs_nom[0]}
            

    if obsName.startswith("mass4l"):
        if _bin==0:
            resultsXS_asimov['SM_125_'+obsName+'_genbin'+str(i)] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov['SM_125_'+obsName+'_genbin'+str(i)+'_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==1:
            resultsXS_asimov_v2['SM_125_'+obsName+'_2e2mu_genbin0'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v2['SM_125_'+obsName+'_2e2mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==2:
            resultsXS_asimov_v2['SM_125_'+obsName+'_4mu_genbin0'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v2['SM_125_'+obsName+'_4mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==3:
            resultsXS_asimov_v2['SM_125_'+obsName+'_4e_genbin0'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v2['SM_125_'+obsName+'_4e_genbin0_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        if "zzfloating" in obsName:
            if _bin==4:
                resultsXS_asimov_v2['SM_125_'+obsName+'_zznorm_genbin0'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
                resultsXS_asimov_v2['SM_125_'+obsName+'_zznorm_genbin0_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
            elif _bin==5:
                resultsXS_asimov_v2['SM_125_'+obsName+'_zznorm4e_genbin0'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
                resultsXS_asimov_v2['SM_125_'+obsName+'_zznorm4e_genbin0_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
            elif _bin==6:
                resultsXS_asimov_v2['SM_125_'+obsName+'_zznorm4mu_genbin0'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
                resultsXS_asimov_v2['SM_125_'+obsName+'_zznorm4mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
            elif _bin==7:
                resultsXS_asimov_v2['SM_125_'+obsName+'_zznorm2e2mu_genbin0'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
                resultsXS_asimov_v2['SM_125_'+obsName+'_zznorm2e2mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
    elif v4_flag:
        if _bin==0:
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin0'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin0_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==1:
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin0'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin0_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==2:
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin1'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin1_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==3:
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin1'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin1_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==4:
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin2'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin2_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==5:
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin2'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin2_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==6:
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin3'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin3_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==7:
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin3'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin3_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==8:
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin4'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin4_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==9:
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin4'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin4_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==10:
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin5'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin5_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==11:
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin5'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin5_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==12:
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin6'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin6_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==13:
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin6'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin6_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==14:
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin7'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin7_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==15:
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin7'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin7_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==16:
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin8'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin8_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==17:
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin8'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin8_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==18:
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin9'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_2e2mu_genbin5_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        elif _bin==19:
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin9'] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov_v4['SM_125_'+obsName+'_4l_genbin5_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
    else:
        if is_vbf_special:
            resultsXS_asimov['SM_125_'+obsName+'_'+vbf_result_label+'_genbin'+str(vbf_physical_bin)] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov['SM_125_'+obsName+'_'+vbf_result_label+'_genbin'+str(vbf_physical_bin)+'_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
        else:
            resultsXS_asimov['SM_125_'+obsName+'_genbin'+str(i)] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
            resultsXS_asimov['SM_125_'+obsName+'_genbin'+str(i)+'_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}
            if 'zzfloating' in obsName and i >= raw_nBins:
                resultsXS_asimov['SM_125_'+obsName+'_zznorm_genbin'+str(zz_bin)] = {"uncerDn": -1.0*abs(exp_nom[2]), "uncerUp": exp_nom[1], "central": exp_nom[0]}
                resultsXS_asimov['SM_125_'+obsName+'_zznorm_genbin'+str(zz_bin)+'_statOnly'] = {"uncerDn": -1.0*abs(exp_nom_stat[2]), "uncerUp": exp_nom_stat[1], "central": exp_nom[0]}

    c.Update()
    #c.SaveAs("plots/lhscan_compare_"+obsName+"_"+poi+".pdf")
    #c.SaveAs("plots/"+year+"_lhscan_compare_"+obsName+"_"+poi+".png")

    outdir = os.path.join(path['plots_path'], "SCANS", obsName)
    os.makedirs(outdir, exist_ok=True)

    c.SaveAs(os.path.join(outdir,
            year + "_lhscan_compare_" + obsName + "_" + poi + ".png"))


if opt.DO_VBF:
    plot_vbf_2d_scans(obsName, raw_nBins, obs_bins, label, label_2nd, year, inputPath)

if v4_flag:
    if opt.UNBLIND:
        with open('resultsXS_LHScan_observed_'+obsName+'_v4.py', 'w') as f:
            f.write('resultsXS = '+str(resultsXS_data_v4)+' \n')
    else:
        with open('resultsXS_LHScan_expected_'+obsName+'_v4.py', 'w') as f:
            f.write('resultsXS = '+str(resultsXS_asimov_v4)+' \n')
else:
    with open('resultsXS_LHScan_expected_'+obsName+'_v3.py', 'w') as f:
        f.write('resultsXS = '+str(resultsXS_asimov)+' \n')
    if obsName.startswith("mass4l"):
        with open('resultsXS_LHScan_expected_'+obsName+'_v2.py', 'w') as f:
            f.write('resultsXS = '+str(resultsXS_asimov_v2)+' \n')

if(opt.UNBLIND) and not v4_flag:
    with open('resultsXS_LHScan_observed_'+obsName+'_v3.py', 'w') as f:
        f.write('resultsXS = '+str(resultsXS_data)+' \n')
    if obsName.startswith("mass4l"):
        with open('resultsXS_LHScan_observed_'+obsName+'_v2.py', 'w') as f:
            f.write('resultsXS = '+str(resultsXS_data_v2)+' \n')
# raw_input()
