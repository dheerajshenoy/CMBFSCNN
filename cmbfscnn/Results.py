# -*- coding: utf-8 -*-
import numpy as np
from . import plottor as pt

from . import CMBFS_mode


class Plot_results(CMBFS_mode.Calculate_power_spectra):
    def __init__(self, result_dir="DATA_results/", is_half_split_map=True):
        super(Plot_results, self).__init__(result_dir)
        self.result_dir = result_dir
        self.is_half_split_map = is_half_split_map
        self.save_PS_dir
        self._creat_ps_file

    def plot_predicted_sphere_map(self, n=0):
        if self.is_half_split_map:
            pre_cmbQ = np.load(getattr(self, "output_Q_dir_1"))
            target_cmbQ = np.load(getattr(self, "target_Q_dir_1"))
            pre_cmbU = np.load(getattr(self, "output_U_dir_1"))
            target_cmbU = np.load(getattr(self, "target_U_dir_1"))
        else:
            pre_cmbQ = np.load(getattr(self, "output_Q_dir"))
            target_cmbQ = np.load(getattr(self, "target_Q_dir"))
            pre_cmbU = np.load(getattr(self, "output_U_dir"))
            target_cmbU = np.load(getattr(self, "target_U_dir"))

        pt.plot_sphere_map(
            pre_cmbQ,
            target_cmbQ,
            title=["Simulated CMB Q map", "Recovered CMB Q map", "Residual"],
            range=[3, 3, 0.2],
            save_dir="recover_CMB_Q_map",
            N_sample=n,
        )
        pt.plot_sphere_map(
            pre_cmbU,
            target_cmbU,
            title=["Simulated CMB U map", "Recovered CMB U map", "Residual"],
            range=[3, 3, 0.2],
            save_dir="recover_CMB_U_map",
            N_sample=n,
        )

    def plot_predicted_flat_map(self, n=0):
        if self.is_half_split_map:
            pre_cmbQ = np.load(
                getattr(self, "output_Qmap_dir") + "predicted_CMB_Q" + "_map_half_1.npy"
            )
            target_cmbQ = np.load(
                getattr(self, "output_Qmap_dir") + "target_CMB_Q" + "_map_half_1.npy"
            )
            pre_cmbU = np.load(
                getattr(self, "output_Umap_dir") + "predicted_CMB_U" + "_map_half_1.npy"
            )
            target_cmbU = np.load(
                getattr(self, "output_Umap_dir") + "target_CMB_U" + "_map_half_1.npy"
            )
        else:
            pre_cmbQ = np.load(
                getattr(self, "output_Qmap_dir") + "predicted_CMB_Q" + "_map.npy"
            )
            target_cmbQ = np.load(
                getattr(self, "output_Qmap_dir") + "target_CMB_Q" + "_map.npy"
            )
            pre_cmbU = np.load(
                getattr(self, "output_Umap_dir") + "predicted_CMB_U" + "_map.npy"
            )
            target_cmbU = np.load(
                getattr(self, "output_Umap_dir") + "target_CMB_U" + "_map.npy"
            )
        title1, title2 = (
            ["Simulated CMB Q map", "Recovered CMB Q map", "Residual"],
            ["Simulated CMB U map", "Recovered CMB U map", "Residual"],
        )

        pt.plot_image(
            pre_cmbQ,
            target_cmbQ,
            title1,
            N_sample=n,
            save_dir="recovered_CMB_flat_Qmap",
            range=[3, 3, 0.2],
        )
        pt.plot_image(
            pre_cmbU,
            target_cmbU,
            title2,
            N_sample=n,
            save_dir="recovered_CMB_flat_Umap",
            range=[3, 3, 0.1],
        )

    def plot_recovered_CMB_QU_PS(self, nlb=5):
        pre_cmbQ_ps = np.load(getattr(self, "save_output_Q_dir_1").format(nlb))
        pre_cmbU_ps = np.load(getattr(self, "save_output_U_dir_1").format(nlb))
        tar_cmbQ_ps = np.load(getattr(self, "save_target_Q_dir_1").format(nlb))
        tar_cmbU_ps = np.load(getattr(self, "save_target_U_dir_1").format(nlb))
        true_cmb_Q_ps = np.load(getattr(self, "true_output_Q_dir").format(nlb))
        true_cmb_U_ps = np.load(getattr(self, "true_output_U_dir").format(nlb))
        pre_denoise_Q_ps = np.load(getattr(self, "save_output_cros_Q_dir").format(nlb))
        pre_denoise_U_ps = np.load(getattr(self, "save_output_cros_U_dir").format(nlb))
        pt.plot_QQUU_PS(
            ell=pre_cmbQ_ps[10, 0, :],
            out_QQ=pre_cmbQ_ps[10, 1, :],
            tar_QQ=tar_cmbQ_ps[10, 1, :],
            out_UU=pre_cmbU_ps[10, 1, :],
            tar_UU=tar_cmbU_ps[10, 1, :],
            out_denoise_QQ=pre_denoise_Q_ps[10, 1, :],
            true_QQ=true_cmb_Q_ps[10, 1, :],
            out_denoise_UU=pre_denoise_U_ps[10, 1, :],
            true_UU=true_cmb_U_ps[10, 1, :],
            filename="qu.png",
        )

    def plot_recovered_CMB_QU_PS_averaged(self, nlb=5):
        pre_cmbQ_ps = np.load(getattr(self, "save_output_Q_dir_1").format(nlb))
        pre_cmbU_ps = np.load(getattr(self, "save_output_U_dir_1").format(nlb))
        tar_cmbQ_ps = np.load(getattr(self, "save_target_Q_dir_1").format(nlb))
        tar_cmbU_ps = np.load(getattr(self, "save_target_U_dir_1").format(nlb))
        true_cmb_Q_ps = np.load(getattr(self, "true_output_Q_dir").format(nlb))
        true_cmb_U_ps = np.load(getattr(self, "true_output_U_dir").format(nlb))
        pre_denoise_Q_ps = np.load(getattr(self, "save_output_cros_Q_dir").format(nlb))
        pre_denoise_U_ps = np.load(getattr(self, "save_output_cros_U_dir").format(nlb))

        # average over all realisations
        pre_cmbQ_ps_mean = np.mean(pre_cmbQ_ps, axis=0)
        pre_cmbU_ps_mean = np.mean(pre_cmbU_ps, axis=0)
        tar_cmbQ_ps_mean = np.mean(tar_cmbQ_ps, axis=0)
        tar_cmbU_ps_mean = np.mean(tar_cmbU_ps, axis=0)
        true_cmb_Q_ps_mean = np.mean(true_cmb_Q_ps, axis=0)
        true_cmb_U_ps_mean = np.mean(true_cmb_U_ps, axis=0)
        pre_denoise_Q_ps_mean = np.mean(pre_denoise_Q_ps, axis=0)
        pre_denoise_U_ps_mean = np.mean(pre_denoise_U_ps, axis=0)

        error_Q = pre_cmbQ_ps_mean - tar_cmbQ_ps_mean
        error_U = pre_cmbU_ps_mean - tar_cmbU_ps_mean

        # plot
        pt.plot_QQUU_PS(
            ell=pre_cmbQ_ps_mean[0, :],
            out_QQ=pre_cmbQ_ps_mean[1, :],
            tar_QQ=tar_cmbQ_ps_mean[1, :],
            out_UU=pre_cmbU_ps_mean[1, :],
            tar_UU=tar_cmbU_ps_mean[1, :],
            out_denoise_QQ=pre_denoise_Q_ps_mean[1, :],
            true_QQ=true_cmb_Q_ps_mean[1, :],
            out_denoise_UU=pre_denoise_U_ps_mean[1, :],
            true_UU=true_cmb_U_ps_mean[1, :],
            filename="qu_averaged.png",
            errors=True,
        )

    def plot_recovered_CMB_EB_PS_averaged(self, nlb=5):
        pre_cmbEB_ps = np.load(getattr(self, "save_output_EB_dir").format(nlb))
        tar_cmbEB_ps = np.load(getattr(self, "save_target_EB_dir").format(nlb))
        true_cmbEB_ps = np.load(getattr(self, "save_true_EB_dir").format(nlb))
        pre_denoise_cmbEB_ps = np.load(
            getattr(self, "save_output_cros_EB_dir").format(nlb)
        )

        ell = pre_cmbEB_ps[0, 0, :]
        # average over all realisations
        pre_cmbEB_ps_mean = np.mean(pre_cmbEB_ps, axis=0)
        tar_cmbEB_ps_mean = np.mean(tar_cmbEB_ps, axis=0)
        true_cmbEB_ps_mean = np.mean(true_cmbEB_ps, axis=0)
        pre_denoise_cmbEB_ps_mean = np.mean(pre_denoise_cmbEB_ps, axis=0)

        pre_cmbEB_ps_std = np.std(pre_cmbEB_ps, axis=0)
        tar_cmbEB_ps_std = np.std(tar_cmbEB_ps, axis=0)
        true_cmbEB_ps_std = np.std(true_cmbEB_ps, axis=0)
        pre_denoise_cmbEB_ps_std = np.std(pre_denoise_cmbEB_ps, axis=0)

        error_E = pre_cmbEB_ps_mean - tar_cmbEB_ps_mean
        error_B = pre_cmbEB_ps_mean - tar_cmbEB_ps_mean

        pt.plot_EEBB_PS(
            ell=ell,
            out_EE=pre_cmbEB_ps_mean[1, :],
            tar_EE=tar_cmbEB_ps_mean[1, :],
            out_BB=pre_cmbEB_ps_mean[2, :],
            tar_BB=tar_cmbEB_ps_mean[2, :],
            out_denoise_EE=pre_denoise_cmbEB_ps_mean[1, :],
            true_EE=true_cmbEB_ps_mean[1, :],
            out_denoise_BB=pre_denoise_cmbEB_ps_mean[2, :],
            true_BB=true_cmbEB_ps_mean[2, :],
            filename="eb_averaged.png",
            # stds={
            #     "out_EE": pre_cmbEB_ps_std[1, :],
            #     "tar_EE": tar_cmbEB_ps_std[1, :],
            #     "out_BB": pre_cmbEB_ps_std[2, :],
            #     "tar_BB": tar_cmbEB_ps_std[2, :],
            #     "out_denoise_EE": pre_denoise_cmbEB_ps_std[1, :],
            #     "true_EE": true_cmbEB_ps_std[1, :],
            #     "out_denoise_BB": pre_denoise_cmbEB_ps_std[2, :],
            #     "true_BB": true_cmbEB_ps_std[2, :],
            # },
            errors = { "E": error_E, "B": error_B }
        )

    def plot_recovered_CMB_EB_PS(self, nlb=5):
        pre_cmbEB_ps = np.load(getattr(self, "save_output_EB_dir").format(nlb))
        tar_cmbEB_ps = np.load(getattr(self, "save_target_EB_dir").format(nlb))
        true_cmbEB_ps = np.load(getattr(self, "save_true_EB_dir").format(nlb))
        pre_denoise_cmbEB_ps = np.load(
            getattr(self, "save_output_cros_EB_dir").format(nlb)
        )

        ell = pre_cmbEB_ps[0, 0, :]

        pt.plot_EEBB_PS(
            ell=ell,
            out_EE=pre_cmbEB_ps[0, 1, :],
            tar_EE=tar_cmbEB_ps[0, 1, :],
            out_BB=pre_cmbEB_ps[0, 2, :],
            tar_BB=tar_cmbEB_ps[0, 2, :],
            out_denoise_EE=pre_denoise_cmbEB_ps[0, 1, :],
            true_EE=true_cmbEB_ps[0, 1, :],
            out_denoise_BB=pre_denoise_cmbEB_ps[0, 2, :],
            true_BB=true_cmbEB_ps[0, 2, :],
            filename="eb.png",
        )

    def plot_recovered_CMB_EB_PS_err(self, nlb=5):
        pre_cmbEB_ps = np.load(getattr(self, "save_output_EB_dir").format(nlb))
        tar_cmbEB_ps = np.load(getattr(self, "save_target_EB_dir").format(nlb))
        true_cmbEB_ps = np.load(getattr(self, "save_true_EB_dir").format(nlb))
        pre_denoise_cmbEB_ps = np.load(
            getattr(self, "save_output_cros_EB_dir").format(nlb)
        )

        ell = pre_cmbEB_ps[0, 0, :]

        delta_QQ_1 = np.zeros((len(pre_cmbEB_ps[:, 0, 0]), len(ell)))
        error_QQ_1 = np.zeros(len(ell))
        for i in range(len(ell)):
            for j in range(len(pre_cmbEB_ps[:, 0, 0])):
                delta_QQ_1[j, i] = abs(pre_cmbEB_ps[j, 1, i] - tar_cmbEB_ps[j, 1, i])
            error_QQ_1[i] = np.sqrt(
                np.sum(delta_QQ_1[:, i] ** 2) / len(pre_cmbEB_ps[:, 0, 0])
            )

        delta_UU_1 = np.zeros((len(pre_cmbEB_ps[:, 0, 0]), len(ell)))
        error_UU_1 = np.zeros(len(ell))
        for i in range(len(ell)):
            for j in range(len(pre_cmbEB_ps[:, 0, 0])):
                delta_UU_1[j, i] = abs(pre_cmbEB_ps[j, 2, i] - tar_cmbEB_ps[j, 2, i])
            error_UU_1[i] = np.sqrt(
                np.sum(delta_UU_1[:, i] ** 2) / len(pre_cmbEB_ps[:, 0, 0])
            )

        delta_QQ_2 = np.zeros((len(pre_denoise_cmbEB_ps[:, 0, 0]), len(ell)))
        error_QQ_2 = np.zeros(len(ell))
        for i in range(len(ell)):
            for j in range(len(pre_denoise_cmbEB_ps[:, 0, 0])):
                delta_QQ_2[j, i] = abs(
                    pre_denoise_cmbEB_ps[j, 1, i] - true_cmbEB_ps[j, 1, i]
                )
            error_QQ_2[i] = np.sqrt(
                np.sum(delta_QQ_2[:, i] ** 2) / len(pre_denoise_cmbEB_ps[:, 0, 0])
            )

        delta_UU_2 = np.zeros((len(pre_denoise_cmbEB_ps[:, 0, 0]), len(ell)))
        error_UU_2 = np.zeros(len(ell))
        for i in range(len(ell)):
            for j in range(len(pre_denoise_cmbEB_ps[:, 0, 0])):
                delta_UU_2[j, i] = abs(
                    pre_denoise_cmbEB_ps[j, 2, i] - true_cmbEB_ps[j, 2, i]
                )
            error_UU_2[i] = np.sqrt(
                np.sum(delta_UU_2[:, i] ** 2) / len(pre_denoise_cmbEB_ps[:, 0, 0])
            )

        pt.plot_EEBB_PS_err(
            ell=ell,
            out_EE=pre_cmbEB_ps[0, 1, :],
            tar_EE=tar_cmbEB_ps[0, 1, :],
            out_BB=pre_cmbEB_ps[0, 2, :],
            tar_BB=tar_cmbEB_ps[0, 2, :],
            out_denoise_EE=pre_denoise_cmbEB_ps[0, 1, :],
            true_EE=true_cmbEB_ps[0, 1, :],
            out_denoise_BB=pre_denoise_cmbEB_ps[0, 2, :],
            true_BB=true_cmbEB_ps[0, 2, :],
            error_QQ_1=error_QQ_1,
            error_UU_1=error_UU_1,
            error_QQ_2=error_QQ_2,
            error_UU_2=error_UU_2,
        )
