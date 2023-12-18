print("start import")
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import uproot as u
import awkward as ak
import mplhep as hep
from sklearn import metrics
from scipy import integrate
import var_dict
plt.style.use(hep.style.CMS)


listbranch = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_hflav', 'jet_jetId', 'pfParticleTransformerAK4JetTags_probb', 'pfParticleTransformerAK4JetTags_probbb', 'pfParticleTransformerAK4JetTags_problepb', 'pfParticleNetAK4JetTags_probb', 'pfParticleNetAK4JetTags_probbb', 'pfDeepFlavourJetTags_probb', 'pfDeepFlavourJetTags_probbb', 'pfDeepFlavourJetTags_problepb', 'pfParticleNetAK4DiscriminatorsJetTags_CvsB',  'pfParticleNetAK4DiscriminatorsJetTags_CvsL', 'pfParticleTransformerAK4JetTags_probc', 'pfParticleTransformerAK4JetTags_probuds', 'pfParticleTransformerAK4JetTags_probg', 'pfDeepFlavourJetTags_probc', 'pfDeepFlavourJetTags_probuds', 'pfDeepFlavourJetTags_probg', 'pfParticleNetAK4DiscriminatorsJetTags_BvsAll', 'pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_BvsAll', 'pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_CvsL', 'pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_CvsB']

#inputfile = "/eos/home-f/fheyen/ttbar_samples/Run_3/ntuple_ttbar_had_*"
inputfile = "/eos/home-f/fheyen/ttbar_samples/Run_2/ntuple_ttbar_had_*"
print(inputfile)
df1 = u.concatenate(inputfile,listbranch)

print("finish import")

lw = 2

jet_pt_cut = 15
jet_eta_cut = 2.5

discriminator = ["CvAll"]


for b in discriminator:
    
    hist = df1
    crit = (df1['jet_pt'] >= jet_pt_cut) * (abs(df1['jet_eta']) <= jet_eta_cut) *  (df1['jet_jetId']>4)

    isB = (df1['jet_hflav'] == 5)[crit]
    isC = (df1['jet_hflav'] == 4)[crit]
    isL = (df1['jet_hflav'] < 4)[crit]

    hist = hist[crit]
        
    fig, axs = plt.subplots(2,1,figsize=(15,15),gridspec_kw={'height_ratios': [3, 1]})
    axs[0].grid(True, which='both',linewidth=2)
    axs[1].grid(True, which='both',linewidth=2)

    x_range = np.arange(0, 1, 0.01)
    efficiency_ParT = np.arange(0, 1, 0.01)
    efficiency_PNet2 = np.arange(0, 1, 0.01)
    efficiency_PNet3 = np.arange(0, 1, 0.01)
    efficiency_DJ = np.arange(0, 1, 0.01)

    efficiency_mistag_ParT = np.arange(0, 1, 0.01)
    efficiency_mistag_PNet2 = np.arange(0, 1, 0.01)
    efficiency_mistag_PNet3 = np.arange(0, 1, 0.01)
    efficiency_mistag_DJ = np.arange(0, 1, 0.01)
    
    if(b == "BvAll"):
        # 'ParticleTransformer'
        disc_ParT = df1['pfParticleTransformerAK4JetTags_probb']+df1['pfParticleTransformerAK4JetTags_probbb']+df1['pfParticleTransformerAK4JetTags_problepb']
        disc_ParT = disc_ParT[crit].to_numpy()
        AUC_ParT = metrics.roc_auc_score(isB, disc_ParT)
        # 'ParticleNet'
        disc_PNet2 = df1["pfParticleNetAK4DiscriminatorsJetTags_BvsAll"]
        disc_PNet2 = disc_PNet2[crit].to_numpy()
        AUC_PNet2 = metrics.roc_auc_score(isB, disc_PNet2)
        # 'ParticleNet3'
        disc_PNet3 = df1["pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_BvsAll"]
        disc_PNet3 = disc_PNet3[crit].to_numpy()
        AUC_PNet3 = metrics.roc_auc_score(isB, disc_PNet3)
        # 'DeepJet'
        disc_DJ = df1['pfDeepFlavourJetTags_probb']+df1['pfDeepFlavourJetTags_probbb']+df1['pfDeepFlavourJetTags_problepb']
        disc_DJ = disc_DJ[crit].to_numpy()
        AUC_DJ = metrics.roc_auc_score(isB, disc_DJ)
    elif(b == "CvAll"):
        #'ParticleNet2'
        disc_CvB_PNet2 = df1["pfParticleNetAK4DiscriminatorsJetTags_CvsB"]
        disc_CvB_PNet2 = disc_CvB_PNet2[crit].to_numpy()

        #'ParticleNet3'
        disc_CvB_PNet3 = df1["pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_CvsB"]
        disc_CvB_PNet3 = disc_CvB_PNet3[crit].to_numpy()
        #'ParticleTransformer'
        disc_CvB_ParT = df1['pfParticleTransformerAK4JetTags_probc']/(df1['pfParticleTransformerAK4JetTags_probc']+df1['pfParticleTransformerAK4JetTags_probb']+df1['pfParticleTransformerAK4JetTags_probbb']+df1['pfParticleTransformerAK4JetTags_problepb'])
        disc_CvB_ParT = disc_CvB_ParT[crit].to_numpy()
        #'DeepJet'
        disc_CvB_DJ = df1['pfDeepFlavourJetTags_probc']/(df1['pfDeepFlavourJetTags_probc']+df1['pfDeepFlavourJetTags_probb']+df1['pfDeepFlavourJetTags_probbb']+df1['pfDeepFlavourJetTags_problepb'])
        disc_CvB_DJ = disc_CvB_DJ[crit].to_numpy()
        

        #'ParticleNet2'
        disc_CvL_PNet2 = df1["pfParticleNetAK4DiscriminatorsJetTags_CvsL"]
        disc_CvL_PNet2 = disc_CvL_PNet2[crit].to_numpy()
        #'ParticleNet3'
        disc_CvL_PNet3 = df1["pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_CvsL"]
        disc_CvL_PNet3 = disc_CvL_PNet3[crit].to_numpy()
        #'ParticleTransformer'
        disc_CvL_ParT = df1['pfParticleTransformerAK4JetTags_probc']/(df1['pfParticleTransformerAK4JetTags_probc']+df1['pfParticleTransformerAK4JetTags_probuds']+df1['pfParticleTransformerAK4JetTags_probg'])
        disc_CvL_ParT = disc_CvL_ParT[crit].to_numpy()
        #'DeepJet'
        disc_CvL_DJ = df1['pfDeepFlavourJetTags_probc']/(df1['pfDeepFlavourJetTags_probc']+df1['pfDeepFlavourJetTags_probuds']+df1['pfDeepFlavourJetTags_probg'])
        disc_CvL_DJ = disc_CvL_DJ[crit].to_numpy()


    for i in range(len(x_range)):
        if(b == "BvAll"):
            disco_ParT = disc_ParT > x_range[i]
            disco_PNet2 = disc_PNet2 > x_range[i]
            disco_PNet3 = disc_PNet3 > x_range[i]
            disco_DJ = disc_DJ > x_range[i]
        if(b == "CvAll"):
            disco_CvB_PNet2 = disc_CvB_PNet2 > x_range[i]
            disco_CvL_PNet2 = disc_CvL_PNet2 > x_range[i]
            disco_PNet2 = np.logical_and(disco_CvB_PNet2, disco_CvL_PNet2)

            disco_CvB_PNet3 = disc_CvB_PNet3 > x_range[i]
            disco_CvL_PNet3 = disc_CvL_PNet3 > x_range[i]
            disco_PNet3 = np.logical_and(disco_CvB_PNet3, disco_CvL_PNet3)

            disco_CvB_ParT = disc_CvB_ParT > x_range[i]
            disco_CvL_ParT = disc_CvL_ParT > x_range[i]
            disco_ParT = np.logical_and(disco_CvB_ParT, disco_CvL_ParT)

            disco_CvB_DJ = disc_CvB_DJ > x_range[i]
            disco_CvL_DJ = disc_CvL_DJ > x_range[i]
            disco_DJ = np.logical_and(disco_CvB_DJ, disco_CvL_DJ)

        isB_pass_ParT = isB[disco_ParT]
        isC_pass_ParT = isC[disco_ParT]
        isL_pass_ParT = isL[disco_ParT]
        hist_pass_ParT = hist[disco_ParT]

        isB_pass_PNet2 = isB[disco_PNet2]
        isC_pass_PNet2 = isC[disco_PNet2]
        isL_pass_PNet2 = isL[disco_PNet2]
        hist_pass_PNet2 = hist[disco_PNet2]

        isB_pass_PNet3 = isB[disco_PNet3]
        isC_pass_PNet3 = isC[disco_PNet3]
        isL_pass_PNet3 = isL[disco_PNet3]
        hist_pass_PNet3 = hist[disco_PNet3]

        isB_pass_DJ = isB[disco_DJ]
        isC_pass_DJ = isC[disco_DJ]
        isL_pass_DJ = isL[disco_DJ]
        hist_pass_DJ = hist[disco_DJ]

        if(b == "BvAll"):
            efficiency_ParT[i] = len(hist_pass_ParT[isB_pass_ParT]) / len(hist[isB])
            efficiency_PNet2[i] = len(hist_pass_PNet2[isB_pass_PNet2]) / len(hist[isB])
            efficiency_PNet3[i] = len(hist_pass_PNet3[isB_pass_PNet3]) / len(hist[isB])
            efficiency_DJ[i] = len(hist_pass_DJ[isB_pass_DJ]) / len(hist[isB])

            efficiency_mistag_ParT[i] = len(hist_pass_ParT[np.logical_or(isL_pass_ParT, isC_pass_ParT)]) / len(hist[np.logical_or(isL, isC)])
            efficiency_mistag_PNet2[i] = len(hist_pass_PNet2[np.logical_or(isL_pass_PNet2, isC_pass_PNet2)]) / len(hist[np.logical_or(isL, isC)])
            efficiency_mistag_PNet3[i] = len(hist_pass_PNet3[np.logical_or(isL_pass_PNet3, isC_pass_PNet3)]) / len(hist[np.logical_or(isL, isC)])
            efficiency_mistag_DJ[i] = len(hist_pass_DJ[np.logical_or(isL_pass_DJ, isC_pass_DJ)]) / len(hist[np.logical_or(isL, isC)])

        if(b == "CvAll"):
            efficiency_ParT[i] = len(hist_pass_ParT[isC_pass_ParT]) / len(hist[isC])
            efficiency_PNet2[i] = len(hist_pass_PNet2[isC_pass_PNet2]) / len(hist[isC])
            efficiency_PNet3[i] = len(hist_pass_PNet3[isC_pass_PNet3]) / len(hist[isC])
            efficiency_DJ[i] = len(hist_pass_DJ[isC_pass_DJ]) / len(hist[isC])

            efficiency_mistag_ParT[i] = len(hist_pass_ParT[np.logical_or(isL_pass_ParT, isB_pass_ParT)]) / len(hist[np.logical_or(isL, isB)])
            efficiency_mistag_PNet2[i] = len(hist_pass_PNet2[np.logical_or(isL_pass_PNet2, isB_pass_PNet2)]) / len(hist[np.logical_or(isL, isB)])
            efficiency_mistag_PNet3[i] = len(hist_pass_PNet3[np.logical_or(isL_pass_PNet3, isB_pass_PNet3)]) / len(hist[np.logical_or(isL, isB)])
            efficiency_mistag_DJ[i] = len(hist_pass_DJ[np.logical_or(isL_pass_DJ, isB_pass_DJ)]) / len(hist[np.logical_or(isL, isB)])

        
    fig, ax = plt.subplots(figsize=(15,15))
    ax.grid(True, which='both',linewidth=2)

    AUC_ParT = abs(integrate.cumtrapz(efficiency_ParT, efficiency_mistag_ParT, initial = 0))
    AUC_PNet2 = abs(integrate.cumtrapz(efficiency_PNet2, efficiency_mistag_PNet2, initial = 0))
    AUC_PNet3 = abs(integrate.cumtrapz(efficiency_PNet3, efficiency_mistag_PNet3, initial = 0))
    AUC_DJ = abs(integrate.cumtrapz(efficiency_DJ, efficiency_mistag_DJ, initial = 0))

    ax.plot(efficiency_ParT, efficiency_mistag_ParT, color='k', markersize = 18, label = "ParT, AUC: "+str(np.round(max(AUC_ParT), 3)))
    ax.plot(efficiency_PNet2, efficiency_mistag_PNet2, color = "royalblue", markersize = 18, label = "PNet Run2, AUC: "+str(np.round(max(AUC_PNet2), 3)))
    ax.plot(efficiency_PNet3, efficiency_mistag_PNet3, color = "sienna", markersize = 18, label = "PNet Run3, AUC: "+str(np.round(max(AUC_PNet3), 3)))
    ax.plot(efficiency_DJ, efficiency_mistag_DJ, color = "red", markersize = 18, label = "DeepJet, AUC: "+str(np.round(max(AUC_DJ), 3)))

    hep.cms.label(rlabel='13 TeV')

    if(b == "BvAll"):
        ax.set_xlabel("B-tag efficiency")
        ax.set_ylabel("B-mistag efficiency")
    if(b == "CvAll"):
        ax.set_xlabel("C-tag efficiency")
        ax.set_ylabel("C-mistag efficiency")

    ax.set_ylim([0.001, 1.0])
    ax.set_xlim([0.1, 1.0])     
    ax.set_yscale('log')

    ax.legend(loc='best', fontsize = 32)
    plt.savefig('plots/ROC/'+b+'.png', bbox_inches='tight')

    plt.close()

