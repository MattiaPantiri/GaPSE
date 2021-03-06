# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
# *  CLASS input parameter file  *
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

# The explation for each paramenter can be found in 'explanatory.ini'
# Here I set the input paramenters for the python script in the form of py dict.

parameters = {
'h':0.7,
'T_cmb':2.726,
'Omega_b':0.0489,
'N_eff':3.04,
'Omega_cdm':0.251020,
'N_ncdm':0,
'Omega_k':0.,
'Omega_fld':0,
'YHe':0.25,
'recombination':'RECFAST',
'reio_parametrization':'reio_camb',
'z_reio':10.,
'reionization_exponent':1.5,
'reionization_width':1.5,
'helium_fullreio_redshift':3.5,
'helium_fullreio_width':0.5,
'annihilation':0.,
'decay':0.,
'output':'mPk,mTk',
'modes':'s',
'lensing':'no',
'ic':'ad',
'P_k_ini type':'analytic_Pk',
'k_pivot':0.05,
'A_s':1.947931e-9,
'n_s':0.96,
'alpha_s':0.,
'P_k_max_h/Mpc':20.,
'z_pk':0,
#'root':'output/WideA_ZA_',
#'write background':'no',
#'write parameters':'yeap',
'background_verbose':1,
'thermodynamics_verbose':1,
'perturbations_verbose':1,
'transfer_verbose':1,
'primordial_verbose':1,
'spectra_verbose':1,
'nonlinear_verbose':1,
'lensing_verbose':1,
'output_verbose':1,
'recfast_Nz0':100000,
'tol_thermo_integration':1.e-5,
'recfast_x_He0_trigger_delta':0.01,
'recfast_x_H0_trigger_delta':0.01,
'evolver':0,
'k_min_tau0':0.002,
'k_max_tau0_over_l_max':3.,
'k_step_sub':0.015,
'k_step_super':0.000001,
'k_step_super_reduction':0.6,
'k_per_decade_for_pk':50,
'start_small_k_at_tau_c_over_tau_h':0.0004,
'start_large_k_at_tau_h_over_tau_k':0.05,
'tight_coupling_trigger_tau_c_over_tau_h':0.005,
'tight_coupling_trigger_tau_c_over_tau_k':0.008,
'start_sources_at_tau_c_over_tau_h':0.006,
'l_max_g':50,
'l_max_pol_g':25,
'l_max_ur':150,
'tol_perturb_integration':1.e-6,
'perturb_sampling_stepsize':0.01,
'radiation_streaming_approximation':2,
'radiation_streaming_trigger_tau_over_tau_k':240.,
'radiation_streaming_trigger_tau_c_over_tau':100.,
'ur_fluid_approximation':2,
'ur_fluid_trigger_tau_over_tau_k':50.,
'ncdm_fluid_approximation':3,
'ncdm_fluid_trigger_tau_over_tau_k':51.,
'l_logstep':1.026,
'l_linstep':25,
'hyper_sampling_flat':12.,
'hyper_sampling_curved_low_nu':10.,
'hyper_sampling_curved_high_nu':10.,
'hyper_nu_sampling_step':10.,
'hyper_phi_min_abs':1.e-10,
'hyper_x_tol':1.e-4,
'hyper_flat_approximation_nu':1.e6,
'q_linstep':0.20,
'q_logstep_spline':20.,
'q_logstep_trapzd':0.5,
'q_numstep_transition':250,
'transfer_neglect_delta_k_S_t0':100.,
'transfer_neglect_delta_k_S_t1':100.,
'transfer_neglect_delta_k_S_t2':100.,
'transfer_neglect_delta_k_S_e':100.,
'transfer_neglect_delta_k_V_t1':100.,
'transfer_neglect_delta_k_V_t2':100.,
'transfer_neglect_delta_k_V_e':100.,
'transfer_neglect_delta_k_V_b':100.,
'transfer_neglect_delta_k_T_t2':100.,
'transfer_neglect_delta_k_T_e':100.,
'transfer_neglect_delta_k_T_b':100.,
'neglect_CMB_sources_below_visibility':1.e-30,
'transfer_neglect_late_source':3000.,
'halofit_k_per_decade':3000.,
'l_switch_limber':40.,
'accurate_lensing':1,
'num_mu_minus_lmax':1000.,
'delta_l_max':1000.}
