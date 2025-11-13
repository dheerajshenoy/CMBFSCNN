# -*- coding: utf-8 -*-

import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import matplotlib

ell_max = 400


def plot_sphere_map(
    denoise_map, target_map, title=[], range=[], save_dir="", N_sample=0
):
    residual_map = target_map - denoise_map
    fig = plt.figure(figsize=(12, 4))
    cmap = plt.cm.jet

    # Default titles and ranges if not provided
    if not title:
        title = ["Target", "Denoised", "Residual"]
    if not range:
        range = [
            np.max(np.abs(target_map[N_sample])),
            np.max(np.abs(denoise_map[N_sample])),
            np.max(np.abs(residual_map[N_sample])),
        ]

    fontsize = {"title": 10, "cbar_label": 8}

    # Plot: Target map
    hp.projview(
        target_map[N_sample, :],
        fig=fig.number,
        cmap=cmap,
        sub=(1, 3, 1),
        unit=r"$\mathrm{\mu K}$",
        min=-range[0],
        max=range[0],
        title=title[0],
        projection_type="mollweide",
        fontsize=fontsize,
    )

    # Plot: Denoised map
    hp.projview(
        denoise_map[N_sample, :],
        fig=fig.number,
        cmap=cmap,
        sub=(1, 3, 2),
        unit=r"$\mathrm{\mu K}$",
        min=-range[1],
        max=range[1],
        title=title[1],
        projection_type="mollweide",
        fontsize=fontsize,
    )

    # Plot: Residual map
    hp.projview(
        residual_map[N_sample, :],
        fig=fig.number,
        cmap=cmap,
        sub=(1, 3, 3),
        unit=r"$\mathrm{\mu K}$",
        min=-range[2],
        max=range[2],
        title=title[2],
        projection_type="mollweide",
        fontsize=fontsize,
        badcolor="white",
    )

    plt.subplots_adjust(
        top=0.85, bottom=0.05, left=0.03, right=0.97, hspace=0.1, wspace=0.05
    )
    # plt.suptitle("CMB Map Comparison", fontsize=14, fontweight="bold")

    if save_dir:
        plt.savefig(save_dir + ".png", dpi=200, bbox_inches="tight")
    plt.clf()


def plot_image(denoise_map, target_map, title, N_sample=0, save_dir="", range=[]):
    denoise_map, target_map = denoise_map[N_sample, :], target_map[N_sample, :]
    residual_map = denoise_map - target_map

    fig = plt.figure(figsize=(18, 6))
    gs0 = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Use perceptually uniform colormap for target & denoised
    cmap = plt.cm.jet

    # --- Target Map ---
    ax1 = fig.add_subplot(gs0[0])
    im1 = ax1.imshow(target_map, cmap=cmap, vmin=-range[0], vmax=range[0])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.ax.tick_params(which="major", length=8, direction="in", width=2, labelsize=14)
    ax1.set_title(title[0], fontsize=18)
    ax1.axis("off")

    # --- Denoised Map ---
    ax2 = fig.add_subplot(gs0[1], sharex=ax1, sharey=ax1)
    im2 = ax2.imshow(denoise_map, cmap=cmap, vmin=-range[1], vmax=range[1])
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.ax.tick_params(which="major", length=8, direction="in", width=2, labelsize=14)
    ax2.set_title(title[1], fontsize=18)
    ax2.axis("off")

    # --- Residual Map ---
    ax3 = fig.add_subplot(gs0[2], sharex=ax1, sharey=ax1)
    im3 = ax3.imshow(residual_map, cmap=cmap, vmin=-range[2], vmax=range[2])
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cb3 = fig.colorbar(im3, cax=cax3)
    cb3.ax.tick_params(which="major", length=8, direction="in", width=2, labelsize=14)
    ax3.set_title(title[2], fontsize=18)
    ax3.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.05, left=0.03, right=0.97)

    if save_dir:
        plt.savefig(save_dir + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_EEBB_PS(
    ell,
    out_EE,
    tar_EE,
    out_BB,
    tar_BB,
    out_denoise_EE,
    true_EE,
    out_denoise_BB,
    true_BB,
    filename="power_EEBB.png",
    dpi=300,
    errors: dict = None,
):
    # title = "s5 d10 recovery using model trained on s1 d1 a2 data"
    plt.style.use("seaborn-v0_8-whitegrid")

    # Consistent font settings for publication
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 1.2,
        }
    )

    def make_plot(axs):
        # Color palette: subtle but distinct
        color_true = "#1b9e77"
        color_noisy = "#7570b3"
        color_recovered = "#d95f02"

        lw = 1.5

        # --- EE (noisy) ---
        ax1 = axs[0, 0]
        ax1.plot(
            ell, tar_EE, label="Simulated noisy EE", color=color_noisy, lw=lw, alpha=0.8
        )
        ax1.plot(
            ell,
            out_EE,
            label="Recovered noisy EE",
            color=color_recovered,
            lw=lw,
            ls="--",
            alpha=0.9,
        )

        ax1.set_ylabel(r"$D_{\ell}^{EE}$")
        ax1.set_xlabel(r"$\ell$")
        ax1.set_xlim(0, ell_max)
        ax1.set_yscale("log")
        ax1.legend(frameon=False)

        err = out_EE - tar_EE

        ax1.fill_between(
            ell,
            out_EE - err,
            out_EE + err,
            color="#1f77b4",
            alpha=0.15,
        )

        # --- BB (noisy) ---
        ax3 = axs[1, 0]
        ax3.plot(
            ell, tar_BB, label="Simulated noisy BB", color=color_noisy, lw=lw, alpha=0.8
        )
        ax3.plot(
            ell,
            out_BB,
            label="Recovered noisy BB",
            color=color_recovered,
            lw=lw,
            ls="--",
            alpha=0.9,
        )
        ax3.set_ylabel(r"$D_{\ell}^{BB}$")
        ax3.set_xlabel(r"$\ell$")
        ax3.set_xlim(0, ell_max)
        ax3.set_yscale("log")
        ax3.legend(frameon=False)

        err = out_BB - tar_BB

        ax3.fill_between(
            ell,
            out_BB - err,
            out_BB + err,
            color="#1f77b4",
            alpha=0.15,
        )

        # --- EE (denoised) ---
        ax2 = axs[0, 1]
        ax2.plot(ell, true_EE, label="True EE", color=color_true, lw=lw, alpha=0.9)
        ax2.plot(
            ell,
            out_denoise_EE,
            label="Recovered EE",
            color=color_recovered,
            lw=lw,
            ls="--",
            alpha=0.9,
        )
        # ax2.set_ylabel(r"$D_{\ell}^{EE}$")
        ax2.set_xlabel(r"$\ell$")
        ax2.set_xlim(0, ell_max)
        ax2.set_yscale("log")
        ax2.legend(frameon=False)

        err = out_denoise_EE - true_EE

        ax2.fill_between(
            ell,
            out_denoise_EE - err,
            out_denoise_EE + err,
            color="#1f77b4",
            alpha=0.15,
        )

        # --- BB (denoised) ---
        ax4 = axs[1, 1]
        ax4.plot(ell, true_BB, label="True BB", color=color_true, lw=lw, alpha=0.9)
        ax4.plot(
            ell,
            out_denoise_BB,
            label="Recovered BB",
            color=color_recovered,
            lw=lw,
            ls="--",
            alpha=0.9,
        )
        # ax4.set_ylabel(r"$D_{\ell}^{BB}$")
        ax4.set_xlabel(r"$\ell$")
        ax4.set_xlim(0, ell_max)
        ax4.set_yscale("log")
        ax4.legend(frameon=False)

        err = out_denoise_BB - true_BB

        ax4.fill_between(
            ell,
            out_denoise_BB - err,
            out_denoise_BB + err,
            color="#1f77b4",
            alpha=0.15,
        )

        # Add light gridlines and consistent tick spacing
        for ax in axs.flat:
            ax.grid(True, ls=":", lw=0.6, alpha=0.6)
            ax.tick_params(direction="in", length=4, width=1.0)

    # Figure setup
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    fig.subplots_adjust(
        left=0.1, right=0.97, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3
    )
    # fig.suptitle(title, fontsize=16, weight='bold')
    make_plot(axs)

    fig.align_ylabels(axs[:, 1])
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)


def plot_EEBB_PS_err(
    ell,
    out_EE,
    tar_EE,
    out_BB,
    tar_BB,
    out_denoise_EE,
    true_EE,
    out_denoise_BB,
    true_BB,
    error_QQ_1,
    error_UU_1,
    error_QQ_2,
    error_UU_2,
    filename="power_EEBB.png",
    dpi=300,
):
    """Plot EE and BB power spectra with error bands and residuals."""
    plt.style.use("seaborn-v0_8-paper")
    fig = plt.figure(figsize=(12, 8))
    outer_gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.25)

    def make_plot(
        subspec, tar, out, y_label, y_res_label, label_tar, label_out, error, BB, lim
    ):
        delta = tar - out

        gs_inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=subspec, height_ratios=[3, 1], hspace=0.05
        )

        # --- Main Power Spectrum ---
        p1 = fig.add_subplot(gs_inner[0])
        p1.plot(ell, tar, label=label_tar, color="#2E8B57", lw=1.4)
        p1.plot(ell, out, label=label_out, color="#D62728", lw=1.4, ls="--")
        p1.set_yscale("log")
        p1.set_xlim(0, ell_max)
        p1.set_ylabel(y_label, fontsize=11)
        p1.legend(fontsize=8, loc="lower right", frameon=False)
        p1.tick_params(axis="x", which="both", labelbottom=False)
        p1.grid(True, which="both", linestyle=":", alpha=0.4)

        # --- Residual plot ---
        p2 = fig.add_subplot(gs_inner[1], sharex=p1)
        p2.fill_between(ell, delta - error, delta + error, color="#1f77b4", alpha=0.15)
        p2.plot(ell, delta, ".", color="#1f77b4", markersize=2, label="Residual")
        p2.axhline(0, color="gray", linestyle="--", lw=1)
        p2.set_ylabel(y_res_label, fontsize=11)
        p2.set_xlabel(r"$\ell$", fontsize=11)
        p2.tick_params(axis="both", which="major", labelsize=8)
        # max_resid = np.max(np.abs(delta))
        # p2.set_ylim(-1.5 * max_resid, 1.5 * max_resid)
        # p2.set_ylim(-1.2 * lim, 1.2 * lim)
        p2.grid(True, linestyle=":", alpha=0.4)

    make_plot(
        outer_gs[0, 0],
        tar_EE,
        out_EE,
        y_label=r"$D_{\ell}^{EE}$ [$\mu$K$^2$]",
        y_res_label=r"$\Delta D_{\ell}^{EE}$ [$\mu$K$^2$]",
        label_tar="Simulated noisy",
        label_out="Recovered noisy",
        error=error_QQ_1,
        BB=False,
        lim=2.1,
    )

    make_plot(
        outer_gs[1, 0],
        tar_BB,
        out_BB,
        y_label=r"$D_{\ell}^{BB}$ [$\mu$K$^2$]",
        y_res_label=r"$\Delta D_{\ell}^{BB}$ [$\mu$K$^2$]",
        label_tar="Simulated noisy",
        label_out="Recovered noisy",
        error=error_UU_1,
        BB=True,
        lim=6.1,
    )

    make_plot(
        outer_gs[0, 1],
        true_EE,
        out_denoise_EE,
        y_label=r"$D_{\ell}^{EE}$ [$\mu$K$^2$]",
        y_res_label=r"$\Delta D_{\ell}^{EE}$ [$\mu$K$^2$]",
        label_tar="Simulated denoised",
        label_out="Recovered denoised",
        error=error_QQ_2,
        BB=False,
        lim=19.5,
    )

    make_plot(
        outer_gs[1, 1],
        true_BB,
        out_denoise_BB,
        y_label=r"$D_{\ell}^{BB}$ [$\mu$K$^2$]",
        y_res_label=r"$\Delta D_{\ell}^{BB}$ [$\mu$K$^2$]",
        label_tar="Simulated denoised",
        label_out="Recovered denoised",
        error=error_UU_2,
        BB=True,
        lim=15.5,
    )

    plt.subplots_adjust(left=0.08, right=0.96, bottom=0.07, top=0.95)
    fig.suptitle("EE and BB Power Spectrum Comparison", fontsize=14, fontweight="bold")
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)


def plot_QQUU_PS(
    ell,
    out_QQ,
    tar_QQ,
    out_UU,
    tar_UU,
    out_denoise_QQ,
    true_QQ,
    out_denoise_UU,
    true_UU,
    filename="power_QU.png",
    dpi=300,
    errors=False,
):
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 1.2,
        }
    )

    color_true, color_noisy, color_recovered = "#1b9e77", "#7570b3", "#d95f02"
    lw = 1.5

    def make_plot(axs):
        # --- QQ noisy ---
        ax1 = axs[0, 0]
        ax1.plot(ell, tar_QQ, label="Simulated noisy QQ", color=color_noisy, lw=lw)
        ax1.plot(
            ell,
            out_QQ,
            label="Recovered noisy QQ",
            color=color_recovered,
            lw=lw,
            ls="--",
        )
        ax1.set_ylabel(r"$D_{\ell}^{QQ}$")
        ax1.set_xlabel(r"$\ell$")
        ax1.set_xlim(0, ell_max)
        ax1.set_yscale("log")
        ax1.legend(frameon=False)

        # --- UU noisy ---
        ax3 = axs[1, 0]
        ax3.plot(ell, tar_UU, label="Simulated noisy UU", color=color_noisy, lw=lw)
        ax3.plot(
            ell,
            out_UU,
            label="Recovered noisy UU",
            color=color_recovered,
            lw=lw,
            ls="--",
        )

        ax3.set_ylabel(r"$D_{\ell}^{UU}$")
        ax3.set_xlabel(r"$\ell$")
        ax3.set_xlim(0, ell_max)
        ax3.set_yscale("log")
        ax3.legend(frameon=False)

        # --- QQ denoised ---
        ax2 = axs[0, 1]
        ax2.plot(ell, true_QQ, label="True QQ", color=color_true, lw=lw)
        ax2.plot(
            ell,
            out_denoise_QQ,
            label="Recovered QQ",
            color=color_recovered,
            lw=lw,
            ls="--",
        )
        ax2.set_xlabel(r"$\ell$")
        ax2.set_xlim(0, ell_max)
        ax2.set_yscale("log")
        ax2.legend(frameon=False)

        # --- UU denoised ---
        ax4 = axs[1, 1]
        ax4.plot(ell, true_UU, label="True UU", color=color_true, lw=lw)
        ax4.plot(
            ell,
            out_denoise_UU,
            label="Recovered UU",
            color=color_recovered,
            lw=lw,
            ls="--",
        )
        ax4.set_xlabel(r"$\ell$")
        ax4.set_xlim(0, ell_max)
        ax4.set_yscale("log")
        ax4.legend(frameon=False)

        if errors:
            err = out_QQ - tar_QQ

            ax1.fill_between(
                ell, out_QQ - err, out_QQ + err, color="#1f77b4", alpha=0.15
            )

            err = (out_denoise_QQ - true_QQ)

            ax2.fill_between(
                ell, out_denoise_QQ - err, out_denoise_QQ + err, color="#1f77b4", alpha=0.15
            )

            err = out_UU - tar_UU

            ax3.fill_between(
                ell, out_UU - err, out_UU + err, color="#1f77b4", alpha=0.15
            )

            err = out_denoise_UU - true_UU

            ax4.fill_between(
                ell, out_denoise_UU - err, out_denoise_UU + err, color="#1f77b4", alpha=0.15
            )

        for ax in axs.flat:
            ax.grid(True, ls=":", lw=0.6, alpha=0.6)
            ax.tick_params(direction="in", length=4, width=1.0)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    fig.subplots_adjust(
        left=0.1, right=0.97, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3
    )
    make_plot(axs)
    fig.align_ylabels(axs[:, 1])
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)
