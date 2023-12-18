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
plt.style.use(hep.style.CMS)


listbranch = ['Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_hadronFlavour','Jet_jetId',
              'Jet_btagDeepFlavB', 'Jet_btagDeepFlavCvB', 'Jet_btagDeepFlavCvL', 'Jet_btagPNetB', 'Jet_btagPNetCvB', 'Jet_btagPNetCvL', 'Jet_btagRobustParTAK4B', 'Jet_btagRobustParTAK4CvB', 'Jet_btagRobustParTAK4CvL',
              'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dSig_0',
              'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dVal_0']

#listbranch = ['Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_hadronFlavour',
#              'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dSig_0',
#              'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dVal_0']


df1 = u.concatenate('/eos/cms/store/group/phys_btag/BPix_2023C/HS_BPix/NanoAOD/'+"ttbar"+"*.root:Events/",listbranch)
#df1 = u.concatenate('/eos/user/a/ademoor/BPixStudies/'+"*.root:Events/",listbranch)
#df2 = u.concatenate('2023D/'+"MuonEG"+"*.root:Events/",listbranch)

print("finish import")

lw = 2

label1 = 'TTbar out-hole'
label2 = 'TTBar in-hole'

truth_flav = 'B'    #desired truth flavour for Sip2D flavour split, leave empty for no split

for b in listbranch:
    print(b)
    hist1 = df1[b]

    if('phi' in b):
        crit1 = (df1['Jet_pt'] >= 200) * (abs(df1['Jet_eta']) <= 2.5) *  (df1['Jet_jetId']>4) #out-hole
        crit2 = (df1['Jet_pt'] >= 200) * (abs(df1['Jet_eta']) <= 2.5) *  (df1['Jet_jetId']>4) #in-hole
        #crit1 = (df1['Jet_pt'] >= 20) * (abs(df1['Jet_eta']) <= 2.5) * (df1['Jet_eta'] <= 0.2) * (df1['Jet_eta'] >= -1.5) * (df1['Jet_jetId']>4)  #out-hole
        #crit2 = (df1['Jet_pt'] >= 20) * (abs(df1['Jet_eta']) <= 2.5) * (df1['Jet_eta'] <= 0.2) * (df1['Jet_eta'] >= -1.5) * (df1['Jet_jetId']>4) #in-hole
    else:
        in_hole = ((df1['Jet_phi'] <= -0.75) & (df1['Jet_phi'] >= -1.3) & ((df1['Jet_eta'] <= 0.2) & (df1['Jet_eta'] >= -1.5)))
        crit1 = (df1['Jet_pt'] >= 20) * (abs(df1['Jet_eta']) <= 2.5) * ~in_hole * (df1['Jet_jetId']>4)   #out-hole
        crit2 = (df1['Jet_pt'] >= 20) * (abs(df1['Jet_eta']) <= 2.5) * in_hole * (df1['Jet_jetId']>4)    #in-hole

    isB1 = (df1['Jet_hadronFlavour'] == 5)[crit1]
    isC1 = (df1['Jet_hadronFlavour'] == 4)[crit1]
    isL1 = (df1['Jet_hadronFlavour'] < 4)[crit1]

    isB2 = (df1['Jet_hadronFlavour'] == 5)[crit2]
    isC2 = (df1['Jet_hadronFlavour'] == 4)[crit2]
    isL2 = (df1['Jet_hadronFlavour'] < 4)[crit2]

    hist2 = hist1[crit2]
    hist1 = hist1[crit1]

    if b != 'nSV':
        hist1 = ak.flatten(hist1).to_numpy()
        hist2 = ak.flatten(hist2).to_numpy()
        print(hist1.shape)

        isB1 = ak.flatten(isB1).to_numpy()
        isC1 = ak.flatten(isC1).to_numpy()
        isL1 = ak.flatten(isL1).to_numpy()
        
        isB2 = ak.flatten(isB2).to_numpy()
        isC2 = ak.flatten(isC2).to_numpy()
        isL2 = ak.flatten(isL2).to_numpy()
    else:
        hist1 = hist1.to_numpy()
        hist2 = hist2.to_numpy()
        print(hist1.shape)

    flav1 = [isB1, isC1, isL1]
    flav2 = [isB2, isC2, isL2]
        
    fig, axs = plt.subplots(2,1,figsize=(15,15),gridspec_kw={'height_ratios': [3, 1]})
    axs[0].grid(True, which='both',linewidth=2)
    axs[1].grid(True, which='both',linewidth=2)
    nbins = 50

    hist1 = hist1[~np.isnan(hist1)]
    hist2 = hist2[~np.isnan(hist2)]
    
    if 'btag' in b:
        nbins = np.arange(0.0,1.02,0.02)
    if 'SVs' in b:
        nbins = np.arange(np.percentile(hist1, 0.01), np.percentile(hist1,0.99), abs(np.percentile(hist1,0.99) - np.percentile(hist1, 0.01)) * 0.02)
    if 'pfcand' in b:
        nbins = np.arange(0, 30, 1)
    if 'nsv' in b:
        nbins = np.arange(0, 8, 1)
    if 'npv' in b:
        nbins = np.arange(0, 100, 1)
    if 'Jet_pt' in b:
#        nbins = np.arange(150, 400, 5)
        nbins = np.arange(0, 250, 5)
    if '2dSig' in b:
        nbins = np.arange(0, 50, 1)
        if(truth_flav == "B"):
            hist1 = hist1[isB1]
            hist2 = hist2[isB2]
            label1 = 'B jets, TTbar out-hole'
            label2 = 'B jets, TTBar in-hole'
        if(truth_flav == "UDSG"):
            hist1 = hist1[isL1]
            hist2 = hist2[isL2]
            label1 = 'UDSG jets, TTbar out-hole'
            label2 = 'UDSG jets, TTBar in-hole'
    if '2dVal' in b:
        nbins = np.arange(0, 0.1, 0.002)
        #axs[0].set_yscale("log")
        if(truth_flav == "B"):
            hist1 = hist1[isB1]
            hist2 = hist2[isB2]
            label1 = 'B jets, TTbar out-hole'
            label2 = 'B jets, TTBar in-hole'
        if(truth_flav == "UDSG"):
            hist1 = hist1[isL1]
            hist2 = hist2[isL2]
            label1 = 'UDSG jets, TTbar out-hole'
            label2 = 'UDSG jets, TTBar in-hole'
    if '3dSig' in b:
        nbins = np.arange(-50, 50, 1)
    if '3dVal' in b:
        nbins = np.arange(0, 40, 1)

    weights1 = np.ones_like(hist1)/float(len(hist1))
    weights2 = np.ones_like(hist2)/float(len(hist2))

    h1,b1,_ = axs[0].hist(hist1, nbins, color = 'black', lw = lw, label=label1, histtype = 'step', weights = weights1)
    h2,b2,_ = axs[0].hist(hist2, nbins, color = 'red', lw = lw, label=label2, histtype = 'step', weights = weights2)
    ratio = np.append(0, (h2-h1)/h1)
    axs[1].step(b1, ratio, color = 'red', lw = lw)
    axs[1].step(b1, np.zeros(len(b1)), color = 'black', lw = lw, ls = 'dashed')
    
    hep.cms.label(rlabel='13.6 TeV', ax=axs[0])
    
    axs[1].set_xlabel(b)
    axs[0].set_ylabel('Number of jets')
    axs[1].set_ylabel('(in-out)/out')
    axs[1].set_ylim([-0.2,0.2])

    fig.tight_layout()
    
    axs[0].legend(loc='best', fontsize = 32)

    if('trackSip2d' in b and (truth_flav == "B" or truth_flav == "UDSG")):
        plt.savefig("TTBar"+'_'+truth_flav +'_jets_'+b+'.png')
    else: 
        plt.savefig("TTBar"+'_'+b+'.png')

    if 'phi' in b:
        max_phi = -0.75
        min_phi = -1.30
        ### B discriminator with PostEE WP from Luca
        for idx, disc in enumerate([df1['Jet_btagDeepFlavB'], df1['Jet_btagPNetB'],df1['Jet_btagRobustParTAK4B']]):
            disc = ak.flatten(disc[crit1]).to_numpy()

            if idx == 0:
                tagger = 'DeepJet'
                wp = [0.0624,0.323,0.7427]
                disc_label = 'BvAll'     
            if idx == 1:
                tagger = 'ParticleNet'
                wp = [0.0458,0.2496,0.7061]
                disc_label = 'BvAll'
            if idx == 2:
                tagger = 'ParticleTransformer'
                wp = [0.0856,0.4319,0.8516]
                disc_label = 'BvAll'


            for wp_idx, wpi in enumerate(wp):
                disco = disc > wpi
                isB0 = isB2[disco]
                isC0 = isC2[disco]
                isL0 = isL2[disco]
                hist0 = hist2[disco]

                inside1 = (hist1 > min_phi) * (hist1 < max_phi)
                inside0 = (hist0 > min_phi) * (hist0 < max_phi)
                
                phi_range = np.arange(-3.14,3.14,2*3.14/30)

                for flav in ["B_jets", "L_jets"]:
                    if(flav == 'B_jets'):
                        eff_in1 = len(hist0[isB0*~inside0]) / len(hist1[isB1*~inside1])
                        eff_in0 = len(hist0[isB0*inside0]) / len(hist1[isB1*inside1])
                        no_cut, _ = np.histogram(hist1[isB1], phi_range)
                        with_cut, _ = np.histogram(hist0[isB0], phi_range)
                        truth_label = 'B_jets'
                    if(flav == 'L_jets'):
                        eff_in1 = len(hist0[isL0*~inside0]) / len(hist1[isL1*~inside1])
                        eff_in0 = len(hist0[isL0*inside0]) / len(hist1[isL1*inside1])
                        no_cut, _ = np.histogram(hist1[isL1], phi_range)
                        with_cut, _ = np.histogram(hist0[isL0], phi_range)
                        truth_label = 'L_jets'

                    ratio = with_cut/no_cut
                    err_ratio = ratio*np.sqrt((1/no_cut)+(1/with_cut))

                    fig, ax = plt.subplots(figsize=(15,15))
                    ax.grid(True, which='both',linewidth=2)

                    print(tagger)
                    print(wpi)
                    #print(ratio)
                    #print(phi_range)
                    print(eff_in1)
                    print(eff_in0)
                    print(eff_in0/eff_in1)
                    ax.errorbar(phi_range[1:] - 2*3.14/30, ratio, fmt='o', yerr = err_ratio, label = flav)
            
                    hep.cms.label(rlabel='13.6 TeV')
                
                    ax.set_xlabel('Jet_phi')
                    wp_label = ''
                    if(wp_idx == 0):
                        wp_label = 'Loose'
                    if(wp_idx == 1):
                        wp_label = 'Medium'
                    if(wp_idx == 2):
                        wp_label = 'Tight'
                    ax.set_ylabel(tagger + ' ' + wp_label + ' ' + disc_label +' efficiency')
                    ax.set_ylim([0.00, 1.00])          

                    ax.legend(loc='best', fontsize = 32)
                    plt.savefig('jet_phi_plots/'+b+'_'+tagger+'_'+disc_label+'_'+truth_label+'_'+wp_label+'.png')

                    plt.close()

        for idx, disc in enumerate([[df1['Jet_btagDeepFlavCvB'], df1['Jet_btagDeepFlavCvL']], [df1['Jet_btagPNetCvB'],df1['Jet_btagPNetCvL']],[df1['Jet_btagRobustParTAK4CvB'],df1['Jet_btagRobustParTAK4CvL']]]):
            disc1 = ak.flatten(disc[0][crit1]).to_numpy()
            disc2 = ak.flatten(disc[1][crit1]).to_numpy()

            print(len(disc1))
            print(len(disc2))

            if idx == 0:
                tagger = 'DeepJet'
                wp = [[0.206, 0.042], [0.298, 0.108], [0.241, 0.305]] #CvB, then CvL   
            if idx == 1:
                tagger = 'ParticleNet'
                wp = [[0.182,0.054], [0.304,0.160], [0.258, 0.491]]
            if idx == 2:
                tagger = 'ParticleTransformer'
                wp = [[0.067, 0.039], [0.128, 0.117], [0.095,0.358]]

            for wp_idx, wpi in enumerate(wp):
                disc_CvB = disc1 > wpi[0] 
                disc_CvL = disc2 > wpi[1]
                disc_CvAll = np.logical_and(disc_CvB, disc_CvL)
                isB0 = isB2[disc_CvAll]
                isC0 = isC2[disc_CvAll]
                isL0 = isL2[disc_CvAll]
                hist0 = hist2[disc_CvAll]
                
                phi_range = np.arange(-3.14,3.14,2*3.14/30)

                no_cut_B, _ = np.histogram(hist1[isB1], phi_range)
                with_cut_B, _ = np.histogram(hist0[isB0], phi_range)
                truth_label_B = 'B_jets'
        
                no_cut_C, _ = np.histogram(hist1[isC1], phi_range)
                with_cut_C, _ = np.histogram(hist0[isC0], phi_range)
                truth_label_C = 'C_jets'

                no_cut_L, _ = np.histogram(hist1[isL1], phi_range)
                with_cut_L, _ = np.histogram(hist0[isL0], phi_range)
                truth_label_L = 'L_jets'

                ratio_B = with_cut_B/no_cut_B
                ratio_C = with_cut_C/no_cut_C
                ratio_L = with_cut_L/no_cut_L
                err_ratio_B = ratio_B*np.sqrt((1/no_cut_B)+(1/with_cut_B))
                err_ratio_C = ratio_C*np.sqrt((1/no_cut_C)+(1/with_cut_C))
                err_ratio_L = ratio_L*np.sqrt((1/no_cut_L)+(1/with_cut_L))

                fig, ax = plt.subplots(figsize=(15,15))
                ax.grid(True, which='both',linewidth=2)
                ax.errorbar(phi_range[1:] - 2*3.14/30, ratio_B, fmt='o', color = 'red', yerr = err_ratio_B, label = truth_label_B)
                ax.errorbar(phi_range[1:] - 2*3.14/30, ratio_C, fmt='o', color = 'blue', yerr = err_ratio_C, label = truth_label_C)
                ax.errorbar(phi_range[1:] - 2*3.14/30, ratio_L, fmt='o', color = 'black', yerr = err_ratio_L, label = truth_label_L)
        
                hep.cms.label(rlabel='13.6 TeV')
            
                ax.set_xlabel('Jet_phi')
                wp_label = ''
                if(wp_idx == 0):
                    wp_label = 'Loose'
                if(wp_idx == 1):
                    wp_label = 'Medium'
                if(wp_idx == 2):
                    wp_label = 'Tight'
                ax.set_ylabel(tagger + ' ' + wp_label + '_CvAll_'+' efficiency')
                ax.set_ylim([0.00, 1.00])          

                ax.legend(loc='best', fontsize = 32)
                plt.savefig('jet_phi_plots/'+b+'_'+tagger+'_CvAll_'+wp_label+'.png')

                plt.close()
    
    if(('btagDeep' in b) or ('btagPNet' in b) or ('btagRobust' in b)):
        discr_wp = np.arange(0.0,1.01,0.01)
        per1, per2, per3 = [], [], []
        for w in discr_wp:
            per1.append(len(hist1[hist1 >w])/(len(hist1)+1e-7))
            per2.append(len(hist2[hist2 >w])/(len(hist2)+1e-7))

        fig, ax = plt.subplots(figsize=(15,15))
        ax.grid(True, which='both',linewidth=2)
        
        plt.plot(discr_wp, per1, 'o', label = label1)
        plt.plot(discr_wp, per2, '^', label = label2)

        hep.cms.label(rlabel='13.6 TeV')
        
        ax.set_xlabel(b)
        ax.set_ylabel('Efficiency')
        
        ax.legend(loc='best', fontsize = 32)
        plt.savefig(b+'_efficiency.png')
        plt.close()
 
