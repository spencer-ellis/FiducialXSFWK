"""
Pileup uncertainty coefficients (PUWEIGHTNP_<obs>_<year>.py, PUWEIGHTNP_evts_<obs>_<year>.py) per process
(signal, qqzz, ggzz), final state and reco bin: "down/up" ratios of the yields with PUWeightDown/PUWeight and
PUWeightUp/PUWeight applied, as read by fit/createDatacard.py (pileupNP()).

Signal: by default the samples of all five Higgs masses (--signalMasses 124,124p5,125,125p5,126) are used together.
The pileup variation does not depend on a ~1 GeV change of m_H, and with only the 125 GeV samples single events with
very large PUWeightUp (up to ~200 in 2024) dominate the +1 sigma variation of a reco bin. Each sample is weighted
1/(number of masses used for its production mode), so the mix of production modes and the yields are unchanged.
Samples whose file or "Counters" histogram is missing are skipped (with a warning). --signalMasses 125 gives the
previous behaviour (125 GeV samples only).
"""
import optparse
import os
import re
import sys

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../helperstuff')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../inputs')))

import uproot

from binning import binning
from createdf_jes import skim_df
from observables import observables
from paths import path

print('Welcome in RunPileup!')


def parseOptions():
    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)
    parser.add_option('', '--obsName', dest='OBSNAME', type='string', default='',
                      help='Name of the observable')
    parser.add_option('', '--obsBins', dest='OBSBINS', type='string', default='',
                      help='Bin boundaries for the diff. measurement separated by "|"')
    parser.add_option('', '--year', dest='YEAR', type='string', default='',
                      help='Period -> 2022, 2022EE, 2022full, 2023preBPix, '
                           '2023postBPix, 2023full, 2024, Run3')
    parser.add_option('', '--verbose', action='store_true', dest='VERBOSE',
                      default=False, help='print values')
    parser.add_option('', '--m4lLower', dest='LOWER_BOUND', type='int',
                      default=105.0, help='Lower bound for m4l')
    parser.add_option('', '--m4lUpper', dest='UPPER_BOUND', type='int',
                      default=160.0, help='Upper bound for m4l')
    parser.add_option('', '--signalMasses', dest='SIGNALMASSES', type='string',
                      default='124,124p5,125,125p5,126',
                      help='Higgs mass samples used together for the signal, separated by "," (125: 125 GeV only)')
    return parser.parse_args()


opt, args = parseOptions()

SIGNAL_PROCS = ['ggH', 'VBFH', 'ttH', 'WminusH', 'WplusH', 'ZH']
FAMILY_ORDER = ['ggH', 'VBFH', 'WH', 'ZH', 'ttH']  # order of the signal concatenation (as before)


def sampleOK(year, sample):
    fname = path['eos_path_sig'] + year + "_MC/" + sample + "/ZZ4lAnalysis_SKIMMED.root"
    if not os.path.isfile(fname):
        return False, 'file not found'
    try:
        keys = [k.split(';')[0] for k in uproot.open(fname).keys()]
    except Exception as exc:
        return False, 'cannot be opened (%s)' % exc
    if 'Counters' not in keys:
        return False, 'no Counters histogram (number of generated events unknown)'
    if 'ZZTree/candTree' not in keys:
        return False, 'no ZZTree/candTree'
    return True, ''


def signalSamples(year, masses):
    """Signal samples of all requested masses that can be used, and the number of masses per production mode.
    WplusH and WminusH of one mass are merged into WH<mass> by skim_df, so a mass is used for WH only if both exist."""
    good = {}
    for m in masses:
        for p in SIGNAL_PROCS:
            ok, why = sampleOK(year, p + m)
            if not ok:
                print('WARNING: %s %s%s not used for the pileup coefficients: %s' % (year, p, m, why))
            good[(p, m)] = ok
    samples, nmass = [], dict.fromkeys(FAMILY_ORDER, 0)
    for m in masses:
        use_w = good[('WminusH', m)] and good[('WplusH', m)]
        for p in SIGNAL_PROCS:
            fam = 'WH' if p.startswith('W') else p
            if (good[(p, m)] and fam != 'WH') or (fam == 'WH' and use_w):
                samples.append(p + m)
        for fam in FAMILY_ORDER:
            if fam == 'WH':
                nmass[fam] += int(use_w)
            else:
                nmass[fam] += int(good[(fam, m)])
    for fam, n in nmass.items():
        if n == 0:
            raise RuntimeError('%s: no usable %s sample for the masses %s' % (year, fam, masses))
    print(year, 'signal samples for the pileup coefficients:', samples)
    print(year, 'masses per production mode:', nmass)
    return samples, nmass


def computeRatio(nominal, upVar, dnVar):
    if nominal == 0:
        return '-'
    dn_ratio = round(dnVar / nominal, 3)
    up_ratio = round(upVar / nominal, 3)
    if up_ratio == dn_ratio and up_ratio == 1.000:
        return '-'
    if up_ratio == dn_ratio:
        return str(dn_ratio)
    return str(dn_ratio) + '/' + str(up_ratio)


def pileup_weight(df, variation):
    ratio = np.where(df['PUWeight'] != 0.0, df[variation] / df['PUWeight'], 1.0)
    return df['weight_histo_reco'] * ratio


def getPileup(channel, m4l_low, m4l_high, obs_reco, obs_bins, recobin, year,
              obs_reco_2nd='None'):
    if not doubleDiff:
        obs_reco_low = obs_bins[recobin]
        obs_reco_high = obs_bins[recobin + 1]
    else:
        obs_reco_low = obs_bins[recobin][0]
        obs_reco_high = obs_bins[recobin][1]
        obs_reco_2nd_low = obs_bins[recobin][2]
        obs_reco_2nd_high = obs_bins[recobin][3]

    datafr = d_sig[year]
    datafr_qqzz = d_bkg[year]['qqzz']
    datafr_ggzz = d_bkg[year]['ggzz']

    if doubleDiff:
        processBin = '_' + channel + '_' + str(year) + '_' + obs_reco + '_' + obs_reco_2nd + '_recobin' + str(recobin)
    else:
        processBin = '_' + channel + '_' + str(year) + '_' + obs_reco + '_recobin' + str(recobin)

    cutobs_reco = (datafr[obs_reco] >= obs_reco_low) & (datafr[obs_reco] < obs_reco_high)
    cutobs_reco_qqzz = (datafr_qqzz[obs_reco] >= obs_reco_low) & (datafr_qqzz[obs_reco] < obs_reco_high)
    cutobs_reco_ggzz = (datafr_ggzz[obs_reco] >= obs_reco_low) & (datafr_ggzz[obs_reco] < obs_reco_high)

    if doubleDiff:
        cutobs_reco &= (datafr[obs_reco_2nd] >= obs_reco_2nd_low) & (datafr[obs_reco_2nd] < obs_reco_2nd_high)
        cutobs_reco_qqzz &= (datafr_qqzz[obs_reco_2nd] >= obs_reco_2nd_low) & (datafr_qqzz[obs_reco_2nd] < obs_reco_2nd_high)
        cutobs_reco_ggzz &= (datafr_ggzz[obs_reco_2nd] >= obs_reco_2nd_low) & (datafr_ggzz[obs_reco_2nd] < obs_reco_2nd_high)

    cutm4l_reco = (datafr['ZZMass'] > m4l_low) & (datafr['ZZMass'] < m4l_high) & (datafr['FinState_reco'] == channel)
    cutm4l_reco_qqzz = (datafr_qqzz['ZZMass'] > m4l_low) & (datafr_qqzz['ZZMass'] < m4l_high) & (datafr_qqzz['FinState_reco'] == channel)
    cutm4l_reco_ggzz = (datafr_ggzz['ZZMass'] > m4l_low) & (datafr_ggzz['ZZMass'] < m4l_high) & (datafr_ggzz['FinState_reco'] == channel)

    masks = {
        'signal': (datafr, cutm4l_reco & cutobs_reco),
        'qqzz': (datafr_qqzz, cutm4l_reco_qqzz & cutobs_reco_qqzz),
        'ggzz': (datafr_ggzz, cutm4l_reco_ggzz & cutobs_reco_ggzz),
    }

    for process, (df, mask) in masks.items():
        nominal = df[mask]['weight_histo_reco'].sum()
        up = pileup_weight(df[mask], 'PUWeightUp').sum()
        dn = pileup_weight(df[mask], 'PUWeightDown').sum()

        evts[process + processBin] = nominal
        evts[process + '_pileupup' + processBin] = up
        evts[process + '_pileupdn' + processBin] = dn
        evts_noWeight[process + processBin] = len(df[mask])
        evts_noWeight[process + '_pileupup' + processBin] = len(df[mask])
        evts_noWeight[process + '_pileupdn' + processBin] = len(df[mask])
        ratio[process + processBin] = computeRatio(nominal, up, dn)


def doGetPileup(obs_reco, obs_bins, obs_reco_2nd='None'):
    chans = ['4e', '4mu', '2e2mu']

    nBins = len(obs_bins)
    if not doubleDiff:
        nBins = len(obs_bins) - 1

    for year in years:
        for chan in chans:
            for recobin in range(nBins):
                getPileup(chan, m4l_low, m4l_high, obs_reco, obs_bins, recobin, year, obs_reco_2nd)

        with open('PUWEIGHTNP_' + obsname_out + '_' + str(year) + '.py', 'w') as f:
            f.write('obsbins = ' + str(obs_bins) + '\n')
            year_ratio = {key: value for key, value in ratio.items()
                          if '_' + str(year) + '_' in key}
            f.write('PUWEIGHTNP = ' + str(year_ratio) + '\n')

        with open('PUWEIGHTNP_evts_' + obsname_out + '_' + str(year) + '.py', 'w') as f:
            f.write('obsbins = ' + str(obs_bins) + '\n')
            year_evts = {key: value for key, value in evts.items()
                         if '_' + str(year) + '_' in key}
            year_evts_noWeight = {
                key: value for key, value in evts_noWeight.items()
                if '_' + str(year) + '_' in key
            }
            f.write('evts = ' + str(year_evts) + '\n')
            f.write('evts_noWeight = ' + str(year_evts_noWeight) + '\n')


obsname = opt.OBSNAME
obsname_out = obsname
doubleDiff = False

if ' vs ' in obsname:
    doubleDiff = True
    obsname_1st = opt.OBSNAME.split(' vs ')[0]
    obsname_2nd = opt.OBSNAME.split(' vs ')[1]
    obsname_out = obsname_1st + '_' + obsname_2nd

obs_reco = observables[obsname]['obs_reco']
if doubleDiff:
    obs_reco_2nd = observables[obsname]['obs_reco_2nd']

if opt.YEAR == '2022':
    years = ['2022']
    years_MC = ['2022']
elif opt.YEAR == '2022EE':
    years = ['2022EE']
    years_MC = ['2022EE']
elif opt.YEAR == '2023preBPix':
    years = ['2023preBPix']
    years_MC = ['2023preBPix']
elif opt.YEAR == '2023postBPix':
    years = ['2023postBPix']
    years_MC = ['2023postBPix']
elif opt.YEAR == '2024':
    years = ['2024']
    years_MC = ['2024']
elif opt.YEAR == '2022full':
    years = ['2022', '2022EE']
    years_MC = ['2022', '2022EE']
elif opt.YEAR == '2023full':
    years = ['2023preBPix', '2023postBPix']
    years_MC = ['2023preBPix', '2023postBPix']
elif opt.YEAR == 'Run3':
    years = ['2022', '2022EE', '2023preBPix', '2023postBPix', '2024']
    years_MC = ['2022', '2022EE', '2023preBPix', '2023postBPix', '2024']
else:
    print(str(opt.YEAR) + ' NOT DEFINED')
    sys.exit(1)

m4l_low = opt.LOWER_BOUND
m4l_high = opt.UPPER_BOUND

print(obsname, years, m4l_low, m4l_high, obsname_out)

obs_bins, doubleDiff = binning(opt.OBSNAME)

masses = opt.SIGNALMASSES.split(',')
d_sig_tmp = {}
d_bkg_tmp = {}
for year in years_MC:
    signals, nmass = signalSamples(year, masses)
    if doubleDiff:
        sig, bkg = skim_df([], year, doubleDiff, obs_reco, obs_reco_2nd, include_pileup=True, signals=signals)
    else:
        sig, bkg = skim_df([], year, doubleDiff, obs_reco, include_pileup=True, signals=signals)
    d_bkg_tmp[year] = bkg
    frames = []
    for m in masses:
        for fam in FAMILY_ORDER:
            if fam + m not in sig:
                continue
            df = sig[fam + m]
            if nmass[fam] != 1:
                df = df.copy()
                df['weight_histo_reco'] = df['weight_histo_reco'] / nmass[fam]
            frames.append(df)
    d_sig_tmp[year] = pd.concat(frames)

d_bkg = {}
d_sig = {}
for year in years:
    d_bkg[year] = d_bkg_tmp[year]
    d_sig[year] = d_sig_tmp[year]

ratio = {}
evts = {}
evts_noWeight = {}

if not doubleDiff:
    doGetPileup(obs_reco, obs_bins)
else:
    doGetPileup(obs_reco, obs_bins, obs_reco_2nd)

with open('PUWEIGHTNP_' + obsname_out + '.py', 'w') as f:
    f.write('obsbins = ' + str(obs_bins) + '\n')
    f.write('PUWEIGHTNP = ' + str(ratio) + '\n')

with open('PUWEIGHTNP_evts_' + obsname_out + '.py', 'w') as f:
    f.write('obsbins = ' + str(obs_bins) + '\n')
    f.write('evts = ' + str(evts) + '\n')
    f.write('evts_noWeight = ' + str(evts_noWeight) + '\n')
