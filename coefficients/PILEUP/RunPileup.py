import optparse
import os
import sys

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../helperstuff')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../inputs')))

from binning import binning
from createdf_jes import skim_df
from observables import observables

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
    return parser.parse_args()


opt, args = parseOptions()


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

d_sig_tmp = {}
d_bkg_tmp = {}
for year in years_MC:
    if doubleDiff:
        sig, bkg = skim_df([], year, doubleDiff, obs_reco, obs_reco_2nd, include_pileup=True)
    else:
        sig, bkg = skim_df([], year, doubleDiff, obs_reco, include_pileup=True)
    d_bkg_tmp[year] = bkg
    d_sig_tmp[year] = pd.concat([sig['ggH125'], sig['VBFH125'], sig['WH125'], sig['ZH125'], sig['ttH125']])

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
