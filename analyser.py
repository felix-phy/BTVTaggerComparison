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
import var_dict
plt.style.use(hep.style.CMS)


listbranch = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_hflav', 'jet_jetId', 'pfParticleTransformerAK4JetTags_probb', 'pfParticleTransformerAK4JetTags_probbb', 'pfParticleTransformerAK4JetTags_problepb', 'pfParticleNetAK4JetTags_probb', 'pfParticleNetAK4JetTags_probbb', 'pfDeepFlavourJetTags_probb', 'pfDeepFlavourJetTags_probbb', 'pfDeepFlavourJetTags_problepb', 'pfParticleNetAK4DiscriminatorsJetTags_CvsB',  'pfParticleNetAK4DiscriminatorsJetTags_CvsL', 'pfParticleTransformerAK4JetTags_probc', 'pfParticleTransformerAK4JetTags_probuds', 'pfParticleTransformerAK4JetTags_probg', 'pfDeepFlavourJetTags_probc', 'pfDeepFlavourJetTags_probuds', 'pfDeepFlavourJetTags_probg', 'pfParticleNetAK4DiscriminatorsJetTags_BvsAll', 'pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_BvsAll', 'pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_CvsL', 'pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_CvsB']
plot = ['jet_pt', 'jet_eta', 'jet_phi']

#inputfile = "/eos/home-f/fheyen/ttbar_samples/Run_3/ntuple_ttbar_had_*"
inputfile = "/eos/home-f/fheyen/ttbar_samples/Run_2/ntuple_ttbar_had_*"
print(inputfile)
df1 = u.concatenate(inputfile,listbranch)

print("finish import")

lw = 2

jet_pt_cut = 15
jet_eta_cut = 2.5

nbins = 20

truth_flav = "B"    #desired truth flavour for Sip2D flavour split, leave empty for no split
discriminator = "CvAll"


for b in plot:
    
    hist = df1[b]
    crit = (df1['jet_pt'] >= jet_pt_cut) * (abs(df1['jet_eta']) <= jet_eta_cut) *  (df1['jet_jetId']>4)

    isB = (df1['jet_hflav'] == 5)[crit]
    isC = (df1['jet_hflav'] == 4)[crit]
    isL = (df1['jet_hflav'] < 4)[crit]

    hist = hist[crit]
    hist = hist[~np.isnan(hist)]

    x_range = np.arange(var_dict.vars[b][0][0], var_dict.vars[b][0][1], abs(var_dict.vars[b][0][1] - var_dict.vars[b][0][0])/nbins)

    print(x_range)

    if(discriminator == "BvAll"):
        # 'ParticleTransformer'
        wp_ParT = [0.0856,0.4319,0.8516]
        disc_ParT = df1['pfParticleTransformerAK4JetTags_probb']+df1['pfParticleTransformerAK4JetTags_probbb']+df1['pfParticleTransformerAK4JetTags_problepb']
        disc_ParT = disc_ParT[crit].to_numpy()

        # 'ParticleNet Run2'
        wp_PNet2 = [0.0458,0.2496,0.7061]
        disc_PNet2 = df1["pfParticleNetAK4DiscriminatorsJetTags_BvsAll"]
        disc_PNet2 = disc_PNet2[crit].to_numpy()

        # 'ParticleNet Run3'
        wp_PNet3 = [0.0458,0.2496,0.7061]
        disc_PNet3 = df1["pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_BvsAll"]
        disc_PNet3 = disc_PNet3[crit].to_numpy()

        # 'DeepJet'
        wp_DJ = [0.0624,0.323,0.7427]
        disc_DJ = df1['pfDeepFlavourJetTags_probb']+df1['pfDeepFlavourJetTags_probbb']+df1['pfDeepFlavourJetTags_problepb']
        disc_DJ = disc_DJ[crit].to_numpy()

    elif('C' in discriminator):
        #'ParticleNet'
        wp_PNet3 = [[0.182,0.054], [0.304,0.160], [0.258, 0.491]]
        disc_CvB_PNet3 = df1["pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_CvsB"]
        disc_CvL_PNet3 = df1["pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags_CvsL"]
        disc_CvB_PNet3 = disc_CvB_PNet3[crit].to_numpy()
        disc_CvL_PNet3 = disc_CvL_PNet3[crit].to_numpy()
        #'ParticleNet 2'
        wp_PNet2 = [[0.182,0.054], [0.304,0.160], [0.258, 0.491]]
        disc_CvB_PNet2 = df1["pfParticleNetAK4DiscriminatorsJetTags_CvsB"]
        disc_CvL_PNet2 = df1["pfParticleNetAK4DiscriminatorsJetTags_CvsL"]
        disc_CvB_PNet2 = disc_CvB_PNet2[crit].to_numpy()
        disc_CvL_PNet2 = disc_CvL_PNet2[crit].to_numpy()
        #'ParticleTransformer'
        wp_ParT = [[0.067, 0.039], [0.128, 0.117], [0.095,0.358]]
        disc_CvB_ParT = df1['pfParticleTransformerAK4JetTags_probc']/(df1['pfParticleTransformerAK4JetTags_probc']+df1['pfParticleTransformerAK4JetTags_probb']+df1['pfParticleTransformerAK4JetTags_probbb']+df1['pfParticleTransformerAK4JetTags_problepb'])
        disc_CvL_ParT = df1['pfParticleTransformerAK4JetTags_probc']/(df1['pfParticleTransformerAK4JetTags_probc']+df1['pfParticleTransformerAK4JetTags_probuds']+df1['pfParticleTransformerAK4JetTags_probg'])
        disc_CvB_ParT = disc_CvB_ParT[crit].to_numpy()
        disc_CvL_ParT = disc_CvL_ParT[crit].to_numpy()
        #'DeepJet'
        wp_DJ = [[0.206, 0.042], [0.298, 0.108], [0.241, 0.305]] #CvB, then CvL
        disc_CvB_DJ = df1['pfDeepFlavourJetTags_probc']/(df1['pfDeepFlavourJetTags_probc']+df1['pfDeepFlavourJetTags_probb']+df1['pfDeepFlavourJetTags_probbb']+df1['pfDeepFlavourJetTags_problepb'])
        disc_CvL_DJ = df1['pfDeepFlavourJetTags_probc']/(df1['pfDeepFlavourJetTags_probc']+df1['pfDeepFlavourJetTags_probuds']+df1['pfDeepFlavourJetTags_probg'])
        disc_CvB_DJ = disc_CvB_DJ[crit].to_numpy()
        disc_CvL_DJ = disc_CvL_DJ[crit].to_numpy()


    for i in range(3):
        if(discriminator == "BvAll"):
            disco_ParT = disc_ParT > wp_ParT[i]
            disco_PNet2 = disc_PNet2 > wp_PNet2[i]
            disco_PNet3 = disc_PNet3 > wp_PNet3[i]
            disco_DJ = disc_DJ > wp_DJ[i]
        
        elif(discriminator == "CvL"):
            disco_CvL_PNet2 = disc_CvL_PNet2 > wp_PNet2[i][1]
            disco_CvL_PNet3 = disc_CvL_PNet3 > wp_PNet3[i][1]
            disco_CvL_ParT = disc_CvL_ParT > wp_ParT[i][1]
            disco_CvL_DJ = disc_CvL_DJ > wp_DJ[i][1]

            disco_PNet2 = disco_CvL_PNet2
            disco_PNet3 = disco_CvL_PNet3
            disco_ParT = disco_CvL_ParT
            disco_DJ = disco_CvL_DJ
        elif(discriminator == "CvB"):
            disco_CvB_PNet2 = disc_CvB_PNet2 > wp_PNet2[i][0]
            disco_CvB_PNet3 = disc_CvB_PNet3 > wp_PNet3[i][0]
            disco_CvB_ParT = disc_CvB_ParT > wp_ParT[i][0]
            disco_CvB_DJ = disc_CvB_DJ > wp_DJ[i][0]

            disco_PNet2 = disco_CvB_PNet2
            disco_PNet3 = disco_CvB_PNet3
            disco_ParT = disco_CvB_ParT
            disco_DJ = disco_CvB_DJ
        elif(discriminator == "CvAll"):
            disco_CvB_PNet2 = disc_CvB_PNet2 > wp_PNet2[i][0]
            disco_CvL_PNet2 = disc_CvL_PNet2 > wp_PNet2[i][1]
            disco_PNet2 = np.logical_and(disco_CvB_PNet2, disco_CvL_PNet2)

            disco_CvB_PNet3 = disc_CvB_PNet3 > wp_PNet3[i][0]
            disco_CvL_PNet3 = disc_CvL_PNet3 > wp_PNet3[i][1]
            disco_PNet3 = np.logical_and(disco_CvB_PNet3, disco_CvL_PNet3)

            disco_CvB_ParT = disc_CvB_ParT > wp_ParT[i][0]
            disco_CvL_ParT = disc_CvL_ParT > wp_ParT[i][1]
            disco_ParT = np.logical_and(disco_CvB_ParT, disco_CvL_ParT)

            disco_CvB_DJ = disc_CvB_DJ > wp_DJ[i][0]
            disco_CvL_DJ = disc_CvL_DJ > wp_DJ[i][1]
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

        if(truth_flav == ''):
            no_cut_ParT, _ = np.histogram(hist, x_range)
            with_cut_ParT, _ = np.histogram(hist_pass_ParT, x_range)

            eff_PNet2 = len(hist_pass_PNet2) / len(hist)
            no_cut_PNet2, _ = np.histogram(hist, x_range)
            with_cut_PNet2, _ = np.histogram(hist_pass_PNet2, x_range)

            eff_PNet3 = len(hist_pass_PNet3) / len(hist)
            no_cut_PNet3, _ = np.histogram(hist, x_range)
            with_cut_PNet3, _ = np.histogram(hist_pass_PNet3, x_range)

            eff_DJ = len(hist_pass_DJ) / len(hist)
            no_cut_DJ, _ = np.histogram(hist, x_range)
            with_cut_DJ, _ = np.histogram(hist_pass_DJ, x_range)

            truth_label = 'All Flavours'

        if(truth_flav == 'B'):
            no_cut_ParT, _ = np.histogram(hist[isB], x_range)
            with_cut_ParT, _ = np.histogram(hist_pass_ParT[isB_pass_ParT], x_range)

            eff_PNet2 = len(hist_pass_PNet2[isB_pass_PNet2]) / len(hist[isB])
            no_cut_PNet2, _ = np.histogram(hist[isB], x_range)
            with_cut_PNet2, _ = np.histogram(hist_pass_PNet2[isB_pass_PNet2], x_range)

            eff_PNet3 = len(hist_pass_PNet3[isB_pass_PNet3]) / len(hist[isB])
            no_cut_PNet3, _ = np.histogram(hist[isB], x_range)
            with_cut_PNet3, _ = np.histogram(hist_pass_PNet3[isB_pass_PNet3], x_range)

            eff_DJ = len(hist_pass_DJ[isB_pass_DJ]) / len(hist[isB])
            no_cut_DJ, _ = np.histogram(hist[isB], x_range)
            with_cut_DJ, _ = np.histogram(hist_pass_DJ[isB_pass_DJ], x_range)

            truth_label = 'B_jets'
        if(truth_flav == 'C'):
            no_cut_ParT, _ = np.histogram(hist[isC], x_range)
            with_cut_ParT, _ = np.histogram(hist_pass_ParT[isC_pass_ParT], x_range)

            eff_PNet2 = len(hist_pass_PNet2[isC_pass_PNet2]) / len(hist[isC])
            no_cut_PNet2, _ = np.histogram(hist[isC], x_range)
            with_cut_PNet2, _ = np.histogram(hist_pass_PNet2[isC_pass_PNet2], x_range)

            eff_PNet3 = len(hist_pass_PNet3[isC_pass_PNet3]) / len(hist[isC])
            no_cut_PNet3, _ = np.histogram(hist[isC], x_range)
            with_cut_PNet3, _ = np.histogram(hist_pass_PNet3[isC_pass_PNet3], x_range)

            eff_DJ = len(hist_pass_DJ[isC_pass_DJ]) / len(hist[isC])
            no_cut_DJ, _ = np.histogram(hist[isC], x_range)
            with_cut_DJ, _ = np.histogram(hist_pass_DJ[isC_pass_DJ], x_range)

            truth_label = 'C_jets'
        if(truth_flav == 'L'):
            no_cut_ParT, _ = np.histogram(hist[isL], x_range)
            with_cut_ParT, _ = np.histogram(hist_pass_ParT[isL_pass_ParT], x_range)

            eff_PNet2 = len(hist_pass_PNet2[isL_pass_PNet2]) / len(hist[isL])
            no_cut_PNet2, _ = np.histogram(hist[isL], x_range)
            with_cut_PNet2, _ = np.histogram(hist_pass_PNet2[isL_pass_PNet2], x_range)

            eff_PNet3 = len(hist_pass_PNet3[isL_pass_PNet3]) / len(hist[isL])
            no_cut_PNet3, _ = np.histogram(hist[isL], x_range)
            with_cut_PNet3, _ = np.histogram(hist_pass_PNet3[isL_pass_PNet3], x_range)

            eff_DJ = len(hist_pass_DJ[isL_pass_DJ]) / len(hist[isL])
            no_cut_DJ, _ = np.histogram(hist[isL], x_range)
            with_cut_DJ, _ = np.histogram(hist_pass_DJ[isL_pass_DJ], x_range)

            truth_label = 'L_jets'

        if(truth_flav == 'C+L'):
            no_cut_ParT, _ = np.histogram(hist[np.logical_or(isC, isL)], x_range)
            with_cut_ParT, _ = np.histogram(hist_pass_ParT[np.logical_or(isC_pass_ParT, isL_pass_ParT)], x_range)

            no_cut_PNet2, _ = np.histogram(hist[np.logical_or(isC, isL)], x_range)
            with_cut_PNet2, _ = np.histogram(hist_pass_PNet2[np.logical_or(isC_pass_PNet2, isL_pass_PNet2)], x_range)

            no_cut_PNet3, _ = np.histogram(hist[np.logical_or(isC, isL)], x_range)
            with_cut_PNet3, _ = np.histogram(hist_pass_PNet3[np.logical_or(isC_pass_PNet3, isL_pass_PNet3)], x_range)

            no_cut_DJ, _ = np.histogram(hist[np.logical_or(isC, isL)], x_range)
            with_cut_DJ, _ = np.histogram(hist_pass_DJ[np.logical_or(isC_pass_DJ, isL_pass_DJ)], x_range)

            truth_label = 'UDSGC_jets'

        ratio_ParT = with_cut_ParT/no_cut_ParT
        err_ratio_ParT = ratio_ParT*np.sqrt((1/no_cut_ParT)+(1/with_cut_ParT))

        ratio_PNet2 = with_cut_PNet2/no_cut_PNet2
        err_ratio_PNet2 = ratio_PNet2*np.sqrt((1/no_cut_PNet2)+(1/with_cut_PNet2))

        ratio_PNet3 = with_cut_PNet3/no_cut_PNet3
        err_ratio_PNet3 = ratio_PNet3*np.sqrt((1/no_cut_PNet3)+(1/with_cut_PNet3))

        ratio_DJ = with_cut_DJ/no_cut_DJ
        err_ratio_DJ = ratio_PNet2*np.sqrt((1/no_cut_DJ)+(1/with_cut_DJ))

        fig, axs = plt.subplots(2,1,figsize=(15,15),gridspec_kw={'height_ratios': [3, 1]})
        axs[0].grid(True, which='both',linewidth=2)
        axs[1].grid(True, which='both',linewidth=2)

        axs[0].errorbar(x_range[1:], ratio_ParT, fmt='.k', yerr = err_ratio_ParT, capsize=10, markersize = 18, label = "ParT")
        axs[0].errorbar(x_range[1:], ratio_PNet2, fmt='.', color = "royalblue", yerr = err_ratio_PNet2, capsize=10, markersize = 18, label = "PNet Run2")
        axs[0].errorbar(x_range[1:], ratio_PNet3, fmt='.', color = "sienna", yerr = err_ratio_PNet3, capsize=10, markersize = 18, label = "PNet Run3")
        axs[0].errorbar(x_range[1:], ratio_DJ, fmt='.', color = "red", yerr = err_ratio_DJ, capsize=10, markersize = 18, label = "DeepJet")

        hep.cms.label(rlabel='13 TeV', ax=axs[0])

        r_PNet2 = np.array([])
        r_PNet3 = np.array([])
        r_ParT = np.array([])

        r_PNet2 = np.append(0, ratio_PNet2/ratio_DJ)
        axs[1].step(x_range, r_PNet2, color = 'royalblue', lw = lw)
        r_PNet3 = np.append(0, ratio_PNet3/ratio_DJ)
        axs[1].step(x_range, r_PNet3, color = 'sienna', lw = lw)
        r_ParT = np.append(0, ratio_ParT/ratio_DJ)
        axs[1].step(x_range, r_ParT, color = 'k', lw = lw)
        axs[1].step(x_range, np.ones(len(x_range)), color = 'black', lw = lw, ls = 'dashed')
    
        axs[1].set_ylabel('tagger/DeepJet')
        axs[1].set_ylim([0.7,1.3])
    
        axs[1].set_xlabel(var_dict.vars[b][1])
        wp_label = ''
        if(i == 0):
            wp_label = 'Loose'
        if(i == 1):
            wp_label = 'Medium'
        if(i == 2):
            wp_label = 'Tight'
        if(truth_flav == "B"):
            axs[0].set_ylabel(wp_label + ' ' + discriminator +' efficiency (B Jets)')
        elif(truth_flav == "L"):
            axs[0].set_ylabel(wp_label + ' ' + discriminator +' efficiency (UDSG Jets)')
        elif(truth_flav == "C"):
            axs[0].set_ylabel(wp_label + ' ' + discriminator +' efficiency (C Jets)')
        elif(truth_flav == "C+L"):
            axs[0].set_ylabel(wp_label + ' ' + discriminator +' efficiency (UDSGC Jets)')
        else:
            axs[0].set_ylabel(wp_label + ' ' + discriminator +' efficiency')

        if('L' in truth_flav):
            axs[0].set_ylim([0.0, max(ratio_DJ)*1.1])
        else:
            axs[0].set_ylim([0.0, 1.0])      

        axs[0].legend(loc='best', fontsize = 32)
        #plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=True)
        plt.savefig('plots/'+discriminator+'/'+b+'/'+truth_label+'_'+wp_label+'.png', bbox_inches='tight')

        plt.close()

