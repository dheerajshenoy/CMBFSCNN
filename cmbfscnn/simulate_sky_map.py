# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import healpy as hp
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# import pysm
import astropy.units as u
import pysm3
from pysm3 import units as pysm_units, utils
# from pysm.nominal import models
from pysm.common import convert_units

from . import get_power_sperctra as ps

# def randomize_synchrotron(sync_sky, std_A=0.1, std_beta=0.05):
#     sync_sky.pl_index *= np.random.normal(1.0, std_beta)
#     sync_sky.I_ref *= np.random.normal(1.0, std_A)
#
# def randomize_dust(dust_sky, std_A=0.1, std_beta=0.05):
#     dust_sky.mbb_index *= np.random.normal(1.0, std_beta)
#     dust_sky.I_ref *= np.random.normal(1.0, std_A)
#
# def randomize_ame(ame_sky, std_A=0.1):
#     comp1 = ame_sky.components[0]
#     comp2 = ame_sky.components[1]
#
#     comp1.I_ref *= np.random.normal(1.0, std_A)
#     comp2.I_ref *= np.random.normal(1.0, std_A)
#

def randomize_synchrotron(sync_sky, config_random):
    s1_seed = config_random['syn_seed']
    np.random.seed(s1_seed)

    if 'syn_spectralindex_random' not in config_random:
        s1_index_std = 0
    else:
        s1_index_std, syn_index_ran_class = config_random['syn_spectralindex_random']

    if s1_index_std == 0:
        print("NOTE: No randomization of synchrotron spectral index")
    else:
        spectral_index = sync_sky.pl_index
        if syn_index_ran_class == 'one' :
            sync_sky.pl_index = spectral_index + \
                    get_specificRandn(50, 0, s1_index_std, -2 * s1_index_std, 2 * s1_index_std)[0] * (spectral_index)
        elif syn_index_ran_class == 'multi' :
            pixel_n = len(spectral_index)
            sync_sky.pl_index = spectral_index + \
                    get_specificRandn(pixel_n * 2, 0, s1_index_std,
                                      -2 * s1_index_std,
                                      2 * s1_index_std)[:pixel_n] * (spectral_index)
        else:
            print("NOTE: syn_spectralindex config error!")

    if 'syn_amplitude_random' not in config_random:
        s1_A_std = 0
    else:
        s1_A_std, syn_A_ran_class = config_random['syn_amplitude_random']

    if s1_A_std == 0:
        print("NOTE: No randomization of synchrotron amplitude")
    else:
        A_I = sync_sky.I_ref
        A_Q = sync_sky.Q_ref
        A_U = sync_sky.U_ref

        if syn_A_ran_class == 'one' :
           sync_sky.I_ref = A_I + get_specificRandn(50, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[0] * A_I
           sync_sky.Q_ref = A_Q + get_specificRandn(50, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[0] * A_Q
           sync_sky.U_ref = A_U + get_specificRandn(50, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[0] * A_U
        elif syn_A_ran_class == 'multi':
            pixel_n = len(A_I)
            sync_sky.I_ref = A_I + get_specificRandn(pixel_n*2, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[:pixel_n] * A_I
            sync_sky.Q_ref = A_Q + get_specificRandn(pixel_n*2, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[:pixel_n] * A_Q
            sync_sky.U_ref = A_U + get_specificRandn(pixel_n*2, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[:pixel_n] * A_U
        else:
            print('Note! syn_amplitude_random config error')

def randomize_dust(dust_sky, config_random):
    d1_seed = config_random['syn_seed']
    np.random.seed(d1_seed)

    if 'dust_amplitude_random' not in config_random:
        d1_A_std = 0
    else:
        d1_A_std, dust_A_ran_class = config_random['dust_amplitude_random']

    if d1_A_std == 0:
        print("NOTE: No randomization of dust amplitude")
    else:
        A_I = dust_sky.I_ref
        A_Q = dust_sky.Q_ref
        A_U = dust_sky.U_ref

        if dust_A_ran_class == 'one' :
           dust_sky.I_ref = A_I + get_specificRandn(50, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[0] * A_I
           dust_sky.Q_ref = A_Q + get_specificRandn(50, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[0] * A_Q
           dust_sky.U_ref = A_U + get_specificRandn(50, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[0] * A_U
        elif dust_A_ran_class == 'multi':
            pixel_n = len(A_I)
            dust_sky.I_ref = A_I + get_specificRandn(pixel_n*2, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[:pixel_n] * A_I
            dust_sky.Q_ref = A_Q + get_specificRandn(pixel_n*2, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[:pixel_n] * A_Q
            dust_sky.U_ref = A_U + get_specificRandn(pixel_n*2, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[:pixel_n] * A_U
        else:
            print('Note! dust_amplitude config error')

    if 'dust_spectralindex_random' not in config_random:
        d1_index_std = 0
    else:
        d1_index_std, dust_index_ran_class = config_random['dust_spectralindex_random']
    if d1_index_std == 0:
        print('Note! dust_spectralindex are not random')
    else:
        spectral_index = dust_sky.mbb_index
        if dust_index_ran_class == 'one' :
            dust_sky.mbb_index = spectral_index + get_specificRandn(50, 0, d1_index_std, -2 * d1_index_std, 2 * d1_index_std)[0] *\
                    (spectral_index - 2.)
        elif dust_index_ran_class == 'multi' :
            pixel_n = len(spectral_index)
            dust_sky.mbb_index = spectral_index + get_specificRandn(pixel_n*2, 0, d1_index_std, -2 * d1_index_std, 2 * d1_index_std)[:pixel_n] *\
                    (spectral_index - 2.)
        else:
            print('Note! syn_spectralindex config error')

    if 'dust_temp_random' not in config_random:
        d1_temp_std = 0
    else:
        d1_temp_std, dust_temp_ran_class = config_random['dust_temp_random']
    if d1_temp_std == 0:
        print('Note! dust_temp are not random')
    else:
        temp = dust_sky.mbb_temperature
        if dust_temp_ran_class == 'one':
            dust_sky.mbb_temperature = temp + get_specificRandn(50, 0, d1_temp_std, -2 * d1_temp_std, 2 * d1_temp_std)[0] * temp
        elif dust_temp_ran_class == 'multi':
            pixel_n = len(temp)
            dust_sky.mbb_temperature = temp + get_specificRandn(pixel_n*2, 0, d1_temp_std, -2 * d1_temp_std, 2 * d1_temp_std)[:pixel_n] * temp
        else:
            print('Note! syn_temp config error')



def randomize_ame(ame_sky, config_random):
        a2_seed = config_random['ame_seed']
        np.random.seed(int(a2_seed))
        if 'ame_amplitude_random' not in config_random:
            a2_A_std = 0
        else:
            a2_A_std, ame_A_ran_class = config_random['ame_amplitude_random']
        if a2_A_std == 0:
            print('Note! ame_amplitude are not random')
        else:
            A_I1 = ame_sky.components[0].I_ref
            A_I2 = ame_sky.components[1].I_ref
            if ame_A_ran_class == 'one':
                ame_sky.components[0].I_ref = A_I1 + get_specificRandn(50, 0, a2_A_std, -2 * a2_A_std, 2 * a2_A_std)[0] * A_I1
                ame_sky.components[1].I_ref = A_I2 + get_specificRandn(50, 0, a2_A_std, -2 * a2_A_std, 2 * a2_A_std)[0] * A_I2
            elif ame_A_ran_class == 'multi':
                pixel_n = len(A_I1)
                ame_sky.components[0].I_ref = A_I1 + get_specificRandn(pixel_n*2, 0, a2_A_std, -2 * a2_A_std, 2 * a2_A_std)[:pixel_n] * A_I1
                ame_sky.components[1].I_ref = A_I2 + get_specificRandn(pixel_n*2, 0, a2_A_std, -2 * a2_A_std, 2 * a2_A_std)[:pixel_n] * A_I2
            else:
                print('Note! ame_A config error')


def get_specificRandn(n, mu, sigma, range_min, range_max):
    randn = np.random.randn(n) * sigma + mu
    choose_1 = np.where(randn>=range_min)
    randn = randn[choose_1]
    choose_2 = np.where(randn<=range_max)
    randn = randn[choose_2]
    return randn


class Get_data(object):
    def __init__(self, Nside,  config_random = {}, freqs=None,  using_beam = False,
                  beam = None, out_unit = None):
        self.Nside = Nside
        self.Nside_exp = 512
        self.Nside_fg = None
        self.freqs = freqs
        self.using_beam = using_beam
        self.beam = beam
        self.out_unit = out_unit # the unit of output, 'K_CMB', 'Jysr', 'uK_RJ';
        # Default unit of signals is 'uK_RJ'; Default unit of noise may be 'K_CMB'
        self.config_random = config_random

    def data(self):
        # random can be 'fixed', fix cosmological paramaters
        self.Nside_fg = 512
        cmb_specs = ps.ParametersSampling(random=self.config_random['Random_types_of_cosmological_parameters'], spectra_type='unlensed_scalar')
        sky_config_fg = ['s1', 'd1', 'a2']
        sky_config_cmb = ["c2"]

        # c2_seed = self.config_random['cmb_seed']
        # c2 = pysm3.CMBLensed(nside = self.Nside_exp, cmb_seed = c2_seed)
        # c2.cmb_spectra = cmb_specs

        # c2_unlens = c2_unlens_mode(self.Nside)
        # cmb_specs_copy = cmb_specs.copy()
        # c2_unlens = pysm3.CMBLensed(nside = self.Nside_exp, cmb_spectra = cmb_specs_copy, cmb_seed = c2_seed)

        sky_fg = pysm3.Sky(nside = self.Nside_fg, preset_strings = sky_config_fg)

        s1 = sky_fg.components[0]
        d1 = sky_fg.components[1]
        a2 = sky_fg.components[2]

        randomize_synchrotron(s1, self.config_random)
        randomize_dust(d1, self.config_random)
        randomize_ame(a2, self.config_random)


        sky_cmb = pysm3.Sky(nside = self.Nside_exp, preset_strings = sky_config_cmb)
        # sky_cmb.components = [c2]
        sky_fg.components = [s1, d1, a2]



        foreground = np.zeros((len(self.freqs), 3, 12*self.Nside_fg**2))
        for i in range(len(self.freqs)):
            foreground[i, :, :] = sky_fg.get_emission(self.freqs[i]*u.GHz).value
        foreground = foreground.astype(np.float32)


        foreground_1 = []
        for i in range(foreground.shape[0]):
            foreground_i = foreground[i, :, :]
            if self.Nside_fg != self.Nside_exp:
                alm_foregroundi = hp.map2alm(foreground_i)
                nside_in = self.Nside_fg
                nside_out = self.Nside_exp

                # Verificar si alm tiene varias componentes
                if alm_foregroundi.ndim > 1:
                    alm_T_total = alm_foregroundi[0]  # Tomar la componente T
                    alm_E_total = alm_foregroundi[1]  # Tomar la componente E
                    alm_B_total = alm_foregroundi[2]  # Tomar la componente B
                else:
                    raise ValueError("alm no tiene suficientes componentes para T, E y B")

                # Obtener lmax de la serie alm
                lmax_in_total = hp.Alm.getlmax(len(alm_T_total))
                lmax_out_total = min(lmax_in_total, 3 * nside_out - 1)  # Limitado por nside_out

                # Obtener funciones de ventana de píxeles
                pixwin_2048 = hp.pixwin(nside_in, pol=True)  # Incluye T, E y B
                pixwin_512 = hp.pixwin(nside_out, pol=True)  # Incluye T, E y B

                # Degradar alm para T, E y B
                alm_degraded_T_total = hp.almxfl(alm_T_total, 1.0 / pixwin_2048[0][: lmax_in_total + 1])  # Remueve efecto de nside=2048
                alm_out_T_total = hp.almxfl(alm_degraded_T_total, pixwin_512[0][: lmax_out_total + 1])    # Aplica efecto de nside=512

                alm_degraded_E_total = hp.almxfl(alm_E_total, 1.0 / pixwin_2048[1][: lmax_in_total + 1])  # Remueve efecto de nside=2048
                alm_out_E_total = hp.almxfl(alm_degraded_E_total, pixwin_512[1][: lmax_out_total + 1])    # Aplica efecto de nside=512

                alm_degraded_B_total = hp.almxfl(alm_B_total, 1.0 / pixwin_2048[1][: lmax_in_total + 1])  # Remueve efecto de nside=2048
                alm_out_B_total = hp.almxfl(alm_degraded_B_total, pixwin_512[1][: lmax_out_total + 1])    # Aplica efecto de nside=512

                # Convertir alm a un único mapa combinando T, E y B en nside=512
                foreground_1.append(hp.alm2map([alm_out_T_total, alm_out_E_total, alm_out_B_total], nside=nside_out, pol=True))
            else:
                foreground_1 = foreground

        cmb = np.zeros((len(self.freqs), 3, 12*self.Nside_exp**2))
        for i in range(len(self.freqs)):
            cmb[i, :, :] = sky_cmb.get_emission(self.freqs[i]*u.GHz).value
        cmb = cmb.astype(np.float32)

        total = foreground_1 + cmb
        total = total.astype(np.float32)

        total_1 = []
        for i in range(total.shape[0]):
            total_i = total[i, :, :]
            if self.Nside != self.Nside_exp:
                alm_totali = hp.map2alm(total_i)
                nside_in = self.Nside_exp
                nside_out = self.Nside

                # Verificar si alm tiene varias componentes
                if alm_totali.ndim > 1:
                    alm_T_total = alm_totali[0]  # Tomar la componente T
                    alm_E_total = alm_totali[1]  # Tomar la componente E
                    alm_B_total = alm_totali[2]  # Tomar la componente B
                else:
                    raise ValueError("alm no tiene suficientes componentes para T, E y B")

                # Obtener lmax de la serie alm
                lmax_in_total = hp.Alm.getlmax(len(alm_T_total))
                lmax_out_total = min(lmax_in_total, 3 * nside_out - 1)  # Limitado por nside_out

                # Obtener funciones de ventana de píxeles
                pixwin_2048 = hp.pixwin(nside_in, pol=True)  # Incluye T, E y B
                pixwin_512 = hp.pixwin(nside_out, pol=True)  # Incluye T, E y B

                # Degradar alm para T, E y B
                alm_degraded_T_total = hp.almxfl(alm_T_total, 1.0 / pixwin_2048[0][: lmax_in_total + 1])  # Remueve efecto de nside=2048
                alm_out_T_total = hp.almxfl(alm_degraded_T_total, pixwin_512[0][: lmax_out_total + 1])    # Aplica efecto de nside=512

                alm_degraded_E_total = hp.almxfl(alm_E_total, 1.0 / pixwin_2048[1][: lmax_in_total + 1])  # Remueve efecto de nside=2048
                alm_out_E_total = hp.almxfl(alm_degraded_E_total, pixwin_512[1][: lmax_out_total + 1])    # Aplica efecto de nside=512

                alm_degraded_B_total = hp.almxfl(alm_B_total, 1.0 / pixwin_2048[1][: lmax_in_total + 1])  # Remueve efecto de nside=2048
                alm_out_B_total = hp.almxfl(alm_degraded_B_total, pixwin_512[1][: lmax_out_total + 1])    # Aplica efecto de nside=512

                # Convertir alm a un único mapa combinando T, E y B en nside=512
                total_1.append(hp.alm2map([alm_out_T_total, alm_out_E_total, alm_out_B_total], nside=nside_out, pol=True))

            else:
                total_1 = total

        cmb_1 = []
        for i in range(cmb.shape[0]):
            cmb_i = cmb[i, :, :]
            if self.Nside != self.Nside_exp:
                alm_cmb = hp.map2alm(cmb_i)
                nside_in = self.Nside_exp
                nside_out = self.Nside

                # verificar si alm tiene varias componentes
                if alm_cmb.ndim > 1:
                    alm_T_cmb = alm_cmb[0]
                    alm_E_cmb = alm_cmb[1]
                    alm_B_cmb = alm_cmb[2]
                else:
                    raise ValueError("alm no tiene suficientes componentes para T, E y B")
                # Obtener lmax de la serie alm
                lmax_in_cmb = hp.Alm.getlmax(len(alm_T_cmb))
                lmax_out_cmb = min(lmax_in_cmb, 3 * nside_out - 1)  # Limitado por nside_out

                # degradar alm para T, E y B
                alm_degraded_T_cmb = hp.almxfl(alm_T_cmb, 1.0 / pixwin_2048[0][: lmax_in_cmb + 1])  # Remueve efecto de nside=2048
                alm_out_T_cmb = hp.almxfl(alm_degraded_T_cmb, pixwin_512[0][: lmax_out_cmb + 1])    # Aplica efecto de nside=512

                alm_degraded_E_cmb = hp.almxfl(alm_E_cmb, 1.0 / pixwin_2048[1][: lmax_in_cmb + 1])  # Remueve efecto de nside=2048
                alm_out_E_cmb = hp.almxfl(alm_degraded_E_cmb, pixwin_512[1][: lmax_out_cmb + 1])    # Aplica efecto de nside=512

                alm_degraded_B_cmb = hp.almxfl(alm_B_cmb, 1.0 / pixwin_2048[1][: lmax_in_cmb + 1])  # Remueve efecto de nside=2048
                alm_out_B_cmb = hp.almxfl(alm_degraded_B_cmb, pixwin_512[1][: lmax_out_cmb + 1])    # Aplica efecto de nside=512
                # Convertir alm a un único mapa combinando T, E y B en nside=512
                cmb_1.append(hp.alm2map([alm_out_T_cmb, alm_out_E_cmb, alm_out_B_cmb], nside=nside_out, pol=True))
            else:
                cmb_1 = cmb

        if self.out_unit:
            Uc_signal = np.array(convert_units("uK_RJ", self.out_unit, self.freqs))
            if not len(self.freqs)>1:  # one frequence
                cmb_1 = cmb_1 * Uc_signal[:, None, None]
                total_1 = total_1 * Uc_signal[:, None, None]
            else:
                cmb_1 = cmb_1 * Uc_signal[:, None, None]
                total_1 = total_1 * Uc_signal[:, None, None]


        total_1 = self.data_proce_beam(total_1)
        cmb_1 = self.data_proce_beam(cmb_1)

        return cmb_1, total_1

    def data_proce_beam(self, map_da):
        if self.using_beam and self.beam is not None:
            beam = self.beam
            map_n = np.array(
                [hp.smoothing(m, fwhm=np.pi / 180. * b / 60.) for (m, b) in zip(map_da, beam)])
        else:
            map_n = map_da
        return map_n

    def noiser(self, Sens, is_half_split_map = True):
        """Calculate white noise maps for given sensitivities.  Returns noise, and noise maps at the given nside in (T, Q, U). Input
        sensitivities are expected to be in uK_CMB amin for the rest of
        PySM.

        :param is_half_split_map: If it is an half-split map, the noise level will increase by sqrt(2) times

        """
        # solid angle per pixel in amin2
        npix = hp.nside2npix(self.Nside)
        # solid angle per pixel in amin2, Note!!!!!
        pix_amin2 = 4. * np.pi / float(hp.nside2npix(self.Nside)) * (180. * 60. / np.pi) ** 2
        """sigma_pix_I/P is std of noise per pixel. It is an array of length
        equal to the number of input maps."""
        if is_half_split_map:
            sigma_pix_I = np.sqrt(Sens ** 2 / pix_amin2)*np.sqrt(2)
        else:
            sigma_pix_I = np.sqrt(Sens ** 2 / pix_amin2)
        noise = np.random.randn(len(Sens), 3,npix)
        noise *= sigma_pix_I[:, None,None]
        if self.out_unit is not None:
            Uc_noise = np.array(convert_units("K_CMB", self.out_unit, self.freqs))
            noise = noise * Uc_noise[:, None, None]
        return noise.astype(np.float32)
