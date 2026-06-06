import ROOT
import sys, os, pwd
from subprocess import *
import optparse, shlex, re
import math
import time
from decimal import *
import json
from collections import OrderedDict as od
from collections import defaultdict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import importlib.util

from paths import path
from higgs_xsbr_13TeV import *

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

def parseOptions():

    global opt, args, runAllSteps

    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)

    # input options
    parser.add_option('-d', '--dir',      dest='SOURCEDIR',type='string',default='./', help='run from the SOURCEDIR as working area, skip if SOURCEDIR is an empty string')
    parser.add_option('',   '--asimovModelName',dest='ASIMOVMODEL',type='string',default='SM_125', help='Name of the Asimov Model')
    parser.add_option('',   '--asimovMass',dest='ASIMOVMASS',type='string',default='125.0', help='Asimov Mass')
    parser.add_option('',   '--ModelNames',dest='MODELNAMES',type='string',default='SM_125',help='Names of models for unfolding, separated by | . Default is "SM_125"')
    parser.add_option('',   '--theoryMass',dest='THEORYMASS',    type='string',default='125.38',   help='Mass value for theory prediction')
    parser.add_option('',   '--fixMass',  dest='FIXMASS',  type='string',default='125.0',   help='Fix mass, default is a string "125.09" or can be changed to another string, e.g."125.6" or "False"')
    parser.add_option('',   '--obsName',  dest='OBSNAME',  type='string',default='pT4l',   help='Name of the observable, supported: "inclusive", "pT4l", "eta4l", "massZ2", "nJets"')
    parser.add_option('',   '--obsBins',  dest='OBSBINS',  type='string',default='|0|30|80|200|10000|',   help='Bin boundaries for the diff. measurement separated by "|", e.g. as "|0|50|100|", use the defalut if empty string')
    parser.add_option('',   '--year',  dest='YEAR',  type='string',default='2022',   help='Year -> 2016 or 2017 or 2018 or Full')
    parser.add_option('',   '--ZZfloating',action='store_true', dest='ZZ',default=False, help='Let ZZ normalisation to float')
    parser.add_option('',   '--interpolation', action='store_true', dest='INTER', default=False, help='Calculate acceptances at 124 and 126 GeV')
    parser.add_option('',   '--doVBF', action='store_true', dest='DO_VBF', default=False, help='Plot VBF scan correlation matrices for absdetajj vs mjj')

    # Unblind option
    parser.add_option('',   '--unblind', action='store_true', dest='UNBLIND', default=False, help='Use real data')
    # Calculate Systematic Uncertainties
    # parser.add_option('',   '--calcSys', action='store_true', dest='SYS', default=False, help='Calculate Systematic Uncertainties (in addition to stat+sys)')

    # store options and arguments as global variables
    global opt, args
    (opt, args) = parser.parse_args()

    if opt.DO_VBF and opt.OBSNAME.strip() not in ['absdetajj vs mjj', 'absdetajj_mjj']:
        parser.error('--doVBF may only be used with --obsName "absdetajj vs mjj"')


VAR_LABELS = {
    "mass4l": r"$m_{4\ell}$",
    "mass4l_zzfloating": r"$m_{4\ell}^{\text{ZZ floating}}$",
    "pT4l": r"$p^T_{4\ell}$",
    "rapidity4l": r"$|y_{4\ell}|$",
    "massZ1": r"$m_{Z_1}$",
    "massZ2": r"$m_{Z_2}$",
    "Nj": r"$N_{jets}$",
    "pTj1": r"$p^T_{j_1}$",
    "pTj2": r"$p^T_{j_2}$",
    "mjj": r"$m_{jj}$",
    "absdetajj": r"$|\Delta\eta_{jj}|$",
    "dphijj": r"$\Delta\Phi_{jj}$",
    "pTHj": r"$p^T_{Hj}$",
    "pTHjj": r"$p^T_{Hjj}$",
    "mHj": r"$m_{Hj}$",
    "TCjmax": r"$\mathcal{T}_C^{max}$",
    "TBjmax": r"$\mathcal{T}_B^{max}$",
    "rapidity4l_pT4l": r"$|y_{H}|$ vs. $p^T_{4\ell}$",
    "massZ1_massZ2": r"$m_{Z_1}$ vs. $m_{Z_2}$",
    "Nj_pT4l": r"$N_{jets}$ vs. $p^T_{4\ell}$",
    "pTj1_pTj2": r"$p^T_{j_1}$ vs. $p^T_{j_2}$",
    "pTHj_pT4l": r"$p^T_{Hj}$ vs. $p^T_{4\ell}$",
    "TCjmax_pT4l": r"$\mathcal{T}_C^{max}$ vs. $p^T_{4\ell}$",
    "pT4l_pTHj": r"$p^T_{4\ell}$ vs. $p^T_{Hj}$",
    "absdetajj_mjj": r"$|\Delta\eta_{jj}|$ vs. $m_{jj}$",
    "costhetaZ1": r"$\cos\theta_{Z_1}$",
    "costhetaZ2": r"$\cos\theta_{Z_2}$",
    "costhetastar": r"$\cos\theta^{*}_{ZZ}$",
    "phi": r"$\Phi$",
    "phi1": r"$\Phi_1$",
}

# parse the arguments and options
parseOptions()

# Define function for processing of os command
def processCmd(cmd, quiet = 0):
    output = '\n'
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT, bufsize=-1)
    for line in iter(p.stdout.readline, ''):
        output=output+str(line)
        # print line,
    p.stdout.close()
    if p.wait() != 0:
        raise RuntimeError("%r failed, exit status: %d" % (cmd, p.returncode))
    return output

def get_fit_name(obsName):
    obs_map = {'pT4l': 'PTH', 'rapidity4l': 'YH', 'pTj1': 'pTj1', 'Nj': 'Nj'}
    return obs_map.get(obsName, obsName)

def get_vbf_pois(label, fitName, nBins):
    poi_templates = {
        'doVBFH': 'r_VBFH_%s_%d',
        'doOtherProd': 'r_otherProd_%s_%d',
        'totalMinusVBF': 'r_totalMinusVBF_%s_%d',
        'ggHExtrap': 'r_VBFH_ggHExtrap_%s_%d',
        'ggHFixed': 'r_VBFH_ggHFixed_%s_%d',
    }
    if label == 'doVBFfix':
        pois = (
            ['r_VBFH_fix_%s_%d' %(fitName, i) for i in range(nBins)]
            + ['r_otherProd_fix_%s_%d' %(fitName, i) for i in range(nBins)]
        )
        pois_plot = (
            ['VBF_%d (fixed)' %i if i < nBins-1 else 'VBF_%d' %i for i in range(nBins)]
            + ['other_%d (fixed)' %i if i < nBins-1 else 'other_%d' %i for i in range(nBins)]
        )
        return pois, pois_plot
    if label == 'doVBFHotherProd':
        pois = (
            ['r_VBFH_otherProd_%s_%d' %(fitName, i) for i in range(nBins)]
            + ['r_otherProd_otherProd_%s_%d' %(fitName, i) for i in range(nBins)]
        )
        pois_plot = ['VBF_%d' %i for i in range(nBins)] + ['other_%d' %i for i in range(nBins)]
        return pois, pois_plot
    if label == 'totalMinusVBF':
        pois = (
            ['r_totalMinusVBF_%s_%d' %(fitName, i) for i in range(nBins-1)]
            + ['r_VBFH_totalMinusVBF_%s_%d' %(fitName, nBins-1)]
        )
        pois_plot = ['total-VBF_%d' %i for i in range(nBins-1)] + ['VBF_%d' %(nBins-1)]
        return pois, pois_plot
    if label == 'allExtrap':
        pois = (
            ['r_totalMinusVBF_allExtrap_%s_%d' %(fitName, i) for i in range(nBins-1)]
            + ['r_VBFH_allExtrap_%s_%d' %(fitName, nBins-1)]
        )
        pois_plot = ['total-VBF_%d' %i for i in range(nBins-1)] + ['VBF_%d' %(nBins-1)]
        return pois, pois_plot
    pois = [poi_templates[label] %(fitName, i) for i in range(nBins)]
    pois_plot = ['r_%d' %i for i in range(nBins)]
    return pois, pois_plot

def PlotCorrelation():
    for physicalModel in PhysicalModels:
        pois = []
        pois_plot = []
        if opt.DO_VBF:
            if obsName != 'absdetajj_mjj' or nBins != 4:
                raise RuntimeError('--doVBF expects "absdetajj vs mjj" to have bins 0-3, but found '+str(nBins)+' bins')
            pois, pois_plot = get_vbf_pois(physicalModel, get_fit_name(obsName), nBins)
        elif 'mass4l' in obsName and physicalModel == 'v2':
            pois = ['r4muBin0', 'r4eBin0', 'r2e2muBin0']
            pois_plot += ['$\sigma_{4\mu}$', '$\sigma_{4e}$', '$\sigma_{2e2\mu}$']
        # elif (obsName == 'massZ1' or obsName == 'massZ2' or obsName == 'costhetastar' or obsName == 'D0m' or obsName == 'Dint' or obsName == 'Dcp' or obsName == 'DL1' or obsName == 'DL1') and physicalModel == 'v4':
        elif physicalModel == 'v4':
            for obsBin in range(nBins):
                pois += ['r4lBin'+str(obsBin)]
                pois_plot += ['$\sigma_{4e+4\mu,'+str(obsBin)+'}$']
            for obsBin in range(nBins):
                pois += ['r2e2muBin'+str(obsBin)]
                pois_plot += ['$\sigma_{2e2\mu,'+str(obsBin)+'}$']
        else:
            # pois += ['CMS_eff_e']
            # pois_plot += ['eff_e']
            fitName = get_fit_name(obsName)
            for obsBin in range(nBins):
                pois += ['r_smH_'+fitName+'_'+str(obsBin)]
                pois_plot += ['r_'+str(obsBin)]
        if obsName == 'mass4l_zzfloating':
            if physicalModel == 'v3':
                pois += ['zz_norm_0']
                pois_plot += ['ZZ']
            else:
                pois = ['r4muBin0', 'zz_norm_0', 'r4eBin0', 'r2e2muBin0']
                pois_plot = ['$\sigma_{4\mu}$', 'ZZ', '$\sigma_{4e}$', '$\sigma_{2e2\mu}$']

        pars = od()
        modes = od()
        fixed_pois = set()
        if opt.DO_VBF and physicalModel == 'doVBFfix':
            fitName = get_fit_name(obsName)
            fixed_pois = set(
                ['r_VBFH_fix_%s_%d' %(fitName, i) for i in range(nBins-1)]
                + ['r_otherProd_fix_%s_%d' %(fitName, i) for i in range(nBins-1)]
            )

        fit_path = path['eos_path']+'combine_files/multidimfit_'+obsName+'_'+physicalModel+'.root'
        inFile = ROOT.TFile(fit_path, 'READ')
        fitResult = inFile.Get('fit_mdf')
        if not fitResult:
            raise RuntimeError('Missing fit_mdf in '+fit_path+'. Rerun RunCorrelation.py.')
        if fitResult.status() != 0:
            raise RuntimeError(
                'Correlation fit %s failed with status %d.'
                %(physicalModel, fitResult.status())
            )
        if fitResult.covQual() < 2:
            raise RuntimeError(
                'Correlation fit %s has invalid covariance quality %d.'
                %(physicalModel, fitResult.covQual())
            )
        theList = fitResult.floatParsFinal()
 
        for iPar in range(len(theList)):
            print( theList[iPar].GetName() )
            if not (theList[iPar].GetName() in pois): continue
            print(iPar, theList[iPar])
            pars[theList[iPar].GetName()] = iPar

        nPars = len(pars.keys())
        print ('Procesing the following %g parameters:'%nPars)
        for par in pars.keys(): print (par)
        theMap = {}
        for iPar in pars:
            for jPar in pars:
                theVal = fitResult.correlation(iPar, jPar)
                if not math.isfinite(theVal):
                    raise RuntimeError(
                        'Correlation fit %s contains non-finite values for %s and %s.'
                        %(physicalModel, iPar, jPar)
                    )
                theMap[(iPar,jPar)] = theVal

        rows = []
        for i in pois:
            row = []
            for j in pois:
                if i in fixed_pois or j in fixed_pois:
                    row.append(1.0 if i == j else 0.0)
                else:
                    row.append(theMap[(i,j)])
            rows.append(row)
        #for b in pois:
         #   rows.append([theMap[i] for i in theMap if i[0]==b])

        theMap = pd.DataFrame(rows, pois_plot, pois_plot)
        print(theMap)

        fig, ax = plt.subplots(figsize = (20, 10))
        ax.text(0., 1.01, r'$\bf{{CMS}}$', fontsize = 20, transform = ax.transAxes)

        ax.text(0.62, 1.01, r'171 fb$^{-1}$ (13.6 TeV)', fontsize = 20, transform = ax.transAxes)
        #ax.text(0.63, 0.9, r'H$\rightarrow$ ZZ', fontsize = 25, transform = ax.transAxes)
        #ax.text(0.55, 0.85, r'm$_{\mathrm{H}}$ = 125.38 GeV', fontsize = 25, transform = ax.transAxes)

        #ax.text(0.45, 0.95, VAR_LABELS[obsName]+r' - H$\rightarrow$ ZZ, m$_{\mathrm{H}}$ = 125.38 GeV', fontsize = 12, transform = ax.transAxes)
        ax.text(0.58, 0.8, VAR_LABELS[obsName], fontsize = 30, transform = ax.transAxes)
        if opt.DO_VBF:
            ax.text(0.62, 0.72, physicalModel, fontsize = 24, transform = ax.transAxes)

        mask = np.zeros_like(theMap)
        mask[np.triu_indices_from(mask, k = 1)] = True


        palette = sns.diverging_palette(240, 10, n=20, as_cmap = True)

        hmap = sns.heatmap(theMap,
                mask = mask,
                vmin=-1.0, vmax=1.0,
        #         xticklabels=ticks,
        #         yticklabels=ticks,
                annot = True,
                fmt = '.2f',
                square = True,
                annot_kws={'size': 12},
                cmap = palette,
                cbar_kws={'pad': .005, 'ticks':np.arange(-1.2, 1.2, 0.2)})
        hmap.figure.axes[-1].tick_params(axis = 'y', labelsize =24, direction='in', length = 10)

        for t in hmap.texts:
            if '-0.00' in t.get_text():
                t.set_text('0.00')

        plt.yticks(rotation=0, fontsize = 20)
        plt.xticks(fontsize = 20, rotation=45)

        plt.axhline(y=0, color='k',linewidth=2.5)
        plt.axhline(y=theMap.shape[1], color='k',linewidth=2.5)
        plt.axvline(x=0, color='k',linewidth=2.5)
        plt.axvline(x=theMap.shape[0], color='k',linewidth=2.5)

        #if opt.UNBLIND:
        #    plt.savefig('/home/llr/cms/bonanomi/fiducial/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/FidJJes/fit/corr_matrix/corr_'+obsName+'_'+physicalModel+'.pdf', bbox_inches='tight')
        #    #plt.savefig('/home/llr/cms/tarabini/CMSSW_10_2_13/src/HiggsAnalysis/FiducialXSFWK/plots/'+obsName+'/data/corr_'+obsName+'_'+physicalModel+'.png')
        #else:
        #    plt.savefig("corr_"+obsName+"_"+physicalModel+".pdf", bbox_inches="tight")
        outdir = os.path.join(path['plots_path'], 'CORRELATION', opt.YEAR)
        os.makedirs(outdir, exist_ok=True)
        path_png = os.path.join(outdir, f'corr_{opt.YEAR}_{obsName}_{physicalModel}.png')
        path_pdf = os.path.join(outdir, f'corr_{opt.YEAR}_{obsName}_{physicalModel}.pdf')
        plt.savefig(path_png, bbox_inches='tight', dpi=600)
        plt.savefig(path_pdf, bbox_inches='tight', dpi=600)
        plt.tight_layout()
        plt.close()

if 'vs' in opt.OBSNAME:
    obsName_tmp = opt.OBSNAME.split(' vs ')
    obsName = obsName_tmp[0]+'_'+obsName_tmp[1]
else:
    obsName = opt.OBSNAME

DataModelName = 'SM_125'
if opt.DO_VBF:
    PhysicalModels = ['doVBFH', 'doOtherProd', 'doVBFfix', 'doVBFHotherProd', 'totalMinusVBF', 'ggHExtrap', 'ggHFixed', 'allExtrap']
elif obsName.startswith("mass4l"):
    PhysicalModels = ['v2','v3']
elif obsName == 'D0m' or obsName == 'Dcp' or obsName == 'D0hp' or obsName == 'Dint' or obsName == 'DL1' or obsName == 'DL1Zg' or obsName == 'costhetaZ1' or obsName == 'costhetaZ2'or obsName == 'costhetastar' or obsName == 'phi' or obsName == 'phistar' or obsName == 'massZ1' or obsName == 'massZ2':
    PhysicalModels = ['v3','v4']
elif 'kL' in obsName:
    PhysicalModels = ['kLambda']
else:
    PhysicalModels = ['v3']

# prepare the set of bin boundaries to run over, it is retrieved from inputs file
# _temp = __import__('inputs_sig_'+obsName+'_'+opt.YEAR, globals(), locals(), ['observableBins', 'acc'], -1)
#_temp = __import__('inputs_sig_'+obsName+'_'+opt.YEAR, globals(), locals(), ['observableBins', 'acc'])
#observableBins = _temp.observableBins
#acc = _temp.acc

if opt.INTER:
    fname = path['eos_path']+'inputs/inputs_sig_extrap_'+obsName+'_'+opt.YEAR+".py"
else:
    fname = path['eos_path']+'inputs/inputs_sig_'+obsName+'_'+opt.YEAR+".py"

module_name = os.path.splitext(os.path.basename(fname))[0]
spec = importlib.util.spec_from_file_location(module_name, fname)
_temp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_temp)

observableBins = _temp.observableBins
acc = _temp.acc

# print 'Running Fiducial XS computation - '+obsName+' - bin boundaries: ', observableBins, '\n'
# print 'Theory xsec and BR at MH = '+_th_MH
# print 'Current directory: python'

doubleDiff = False
if type(observableBins) is dict: doubleDiff = True # If binning is a dictionary it is a double differential analysis

nBins = len(observableBins)
if not doubleDiff: nBins = nBins-1 #in case of 1D measurement the number of bins is -1 the length of the list of bin boundaries

os.chdir(path['eos_path']+'combine_files/')
PlotCorrelation()
