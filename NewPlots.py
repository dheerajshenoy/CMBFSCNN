import numpy as np
import matplotlib.pyplot as plt

# Load data files
cases = {
    's1d1': {
        'auto_rec': 'NEW_PLOTS/s1d1_recovered_auto.npy',
        'auto_target': 'NEW_PLOTS/s1d1_target_auto.npy',
        'cross_rec': 'NEW_PLOTS/s1d1_recovered_cross.npy',
        'cross_target': 'NEW_PLOTS/s1d1_target_cross.npy',
        'title': 'Standard s1d1'
    },
    's5d10': {
        'auto_rec': 'NEW_PLOTS/s5d10_recovered_auto.npy',
        'auto_target': 'NEW_PLOTS/s5d10_target_auto.npy',
        'cross_rec': 'NEW_PLOTS/s5d10_recovered_cross.npy',
        'cross_target': 'NEW_PLOTS/s5d10_target_cross.npy',
        'title': 's5d10 models'
    },
    'fixed_train': {
        'auto_rec': 'NEW_PLOTS/s1d1_fixed_train_recovered_auto.npy',
        'auto_target': 'NEW_PLOTS/s1d1_fixed_train_target_auto.npy',
        'cross_rec': 'NEW_PLOTS/s1d1_fixed_train_recovered_cross.npy',
        'cross_target': 'NEW_PLOTS/s1d1_fixed_train_target_cross.npy',
        'title': 's1d1 Fixed FG in training'
    },
    'fixed_all': {
        'auto_rec': 'NEW_PLOTS/s1d1_fixed_all_recovered_auto.npy',
        'auto_target': 'NEW_PLOTS/s1d1_fixed_all_target_auto.npy',
        'cross_rec': 'NEW_PLOTS/s1d1_fixed_all_recovered_cross.npy',
        'cross_target': 'NEW_PLOTS/s1d1_fixed_all_target_cross.npy',
        'title': 's1d1 Fixed FG in all'
    }
}

def create_power_spectrum_plot(mode_index, mode_name):
    # Create figure with 4x4 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 4, hspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    
    # Plot for each case
    for idx, (case_name, case_data) in enumerate(cases.items()):
        # Load data
        auto_rec = np.load(case_data['auto_rec'])
        auto_target = np.load(case_data['auto_target'])
        cross_rec = np.load(case_data['cross_rec'])
        cross_target = np.load(case_data['cross_target'])
        
        # Get ell values and limit to 300
        ell = auto_rec[0, 0, :]
        mask = ell <= 256
        ell = ell[mask]
        
        # Calculate mean and std over all simulations
        auto_rec_mean = np.mean(auto_rec[:, mode_index, :], axis=0)[mask]
        auto_target_mean = np.mean(auto_target[:, mode_index, :], axis=0)[mask]
        cross_rec_mean = np.mean(cross_rec[:, mode_index, :], axis=0)[mask]
        cross_target_mean = np.mean(cross_target[:, mode_index, :], axis=0)[mask]
        
        auto_rec_std = np.std(auto_rec[:, mode_index, :], axis=0)[mask]
        auto_target_std = np.zeros_like(auto_rec_std)
        #auto_target_std = np.std(auto_target[:, mode_index, :], axis=0)[mask]
        cross_rec_std = np.std(cross_rec[:, mode_index, :], axis=0)[mask]
        #cross_target_std = np.std(cross_target[:, mode_index, :], axis=0)[mask]
        cross_target_std = np.zeros_like(cross_rec_std)
        
        # Calculate residuals and their uncertainties
        auto_residual = auto_rec_mean - auto_target_mean
        cross_residual = cross_rec_mean - cross_target_mean
        delta_auto_residual = np.zeros((len(auto_rec[:,0,0]), len(ell))) 
        auto_residual_std = np.zeros(len(ell))
        for i in range(len(ell)):
            for j in range(len(auto_rec[:,0,0])):
                delta_auto_residual[j,i] = abs(auto_rec[j,mode_index,i] - auto_target[j,mode_index,i])
            auto_residual_std[i] = np.sqrt(np.sum(delta_auto_residual[:,i]**2)/len(auto_rec[:,0,0])) 
        delta_cross_residual = np.zeros((len(cross_rec[:,0,0]), len(ell))) 
        cross_residual_std = np.zeros(len(ell))
        for i in range(len(ell)):
            for j in range(len(cross_rec[:,0,0])):
                delta_cross_residual[j,i] = abs(cross_rec[j,mode_index,i] - cross_target[j,mode_index,i])
            cross_residual_std[i] = np.sqrt(np.sum(delta_cross_residual[:,i]**2)/len(cross_rec[:,0,0])) 
        #auto_residual_std = np.sqrt(auto_rec_std**2 + auto_target_std**2)
        #cross_residual_std = np.sqrt(cross_rec_std**2 + cross_target_std**2)
    
        # Plot auto spectra with errors
        axs[0, idx].semilogy(ell, auto_rec_mean, 'r-', label='Recovered')
        axs[0, idx].fill_between(ell, auto_rec_mean-auto_rec_std, auto_rec_mean+auto_rec_std, 
                                color='r', alpha=0.2)
        axs[0, idx].semilogy(ell, auto_target_mean, 'b--', label='Target')
        axs[0, idx].fill_between(ell, auto_target_mean-auto_target_std, auto_target_mean+auto_target_std, 
                                color='b', alpha=0.2)
        axs[0, idx].set_title(f"{case_data['title']} ({mode_name}-mode)")
        if idx == 0:
            axs[0, idx].set_ylabel(r'$D_{\ell}^{' + mode_name + mode_name + r'}$ [$\mu K^2$] (Auto)')
        axs[0, idx].grid(True)
        axs[0, idx].legend()
        
        # Plot auto residuals with errors
        axs[1, idx].plot(ell, auto_residual, 'k-')
        axs[1, idx].fill_between(ell, auto_residual-auto_residual_std, auto_residual+auto_residual_std, 
                                color='gray', alpha=0.3)
        if idx == 0:
            axs[1, idx].set_ylabel(r'$\Delta D_{\ell}^{' + mode_name + mode_name + r'}$ (Auto)')
        axs[1, idx].grid(True)
    
        # Plot cross spectra with errors
        axs[2, idx].semilogy(ell, cross_rec_mean, 'r-', label='Recovered')
        axs[2, idx].fill_between(ell, cross_rec_mean-cross_rec_std, cross_rec_mean+cross_rec_std, 
                                color='r', alpha=0.2)
        axs[2, idx].semilogy(ell, cross_target_mean, 'b--', label='Target')
        axs[2, idx].fill_between(ell, cross_target_mean-cross_target_std, cross_target_mean+cross_target_std, 
                                color='b', alpha=0.2)
        if idx == 0:
            axs[2, idx].set_ylabel(r'$D_{\ell}^{' + mode_name + mode_name + r'}$ [$\mu K^2$] (Cross)')
        axs[2, idx].grid(True)
        axs[2, idx].legend()
    
        # Plot cross residuals with errors
        axs[3, idx].plot(ell, cross_residual, 'k-')
        axs[3, idx].fill_between(ell, cross_residual-cross_residual_std, cross_residual+cross_residual_std, 
                                color='gray', alpha=0.3)
        axs[3, idx].set_xlabel('ℓ')
        if idx == 0:
            axs[3, idx].set_ylabel(r'$\Delta D_{\ell}^{' + mode_name + mode_name + r'}$ (Cross)')
        axs[3, idx].grid(True)
        
        # Set xlim for all panels in this column
        for row in range(4):
            axs[row, idx].set_xlim(0, 256)
        if mode_index==1:
            for row in range(4):
                axs[2,idx].set_ylim(0,20)
    
    plt.savefig(f'power_spectra_comparison_{mode_name}mode.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create separate plots for E and B modes
create_power_spectrum_plot(1, 'E')  # E-mode
create_power_spectrum_plot(2, 'B')  # B-mode
