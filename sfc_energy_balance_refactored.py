import xarray as xr
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from saturation_thermodynamics import get_saturation_thermodynamics
import constants as const

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def lon_structure(lon, lon_east, lon_west, lon_width):
    """Idealized plus and minus Gaussian lobes longitude structure."""
    return -np.exp(-(lon-lon_east)**2/lon_width**2) + np.exp(-(lon-lon_west)**2/lon_width**2)

def lh_flux_dtas(SST, RH, windspeed, dtas):
    """Bulk formula with surface air disequilibrium."""
    rho = 1.3
    cd = 1.1e-3

    es, qs_sst, rs, L = get_saturation_thermodynamics(SST, 1e5, thermo_type='era')
    
    # assumes dtas is positive: dtas = SST - TAS
    es, qs_tas, rs, L = get_saturation_thermodynamics(SST-dtas, 1e5, thermo_type='era')

    le_flux = const.Lv*rho*cd*windspeed*(qs_sst-RH*qs_tas)
    
    return le_flux

def sfc_rad(T, A, B):
    """Surface radiation."""
    return A + B*T

def surf_en_tendency(t, T, F, OHF, RH, windspeed, dtas, A, B):
    """Surface energy balance tendency."""
    SFCrad = sfc_rad(T, A, B)
    LE = lh_flux_dtas(T, RH, windspeed, dtas)

    # heat capacity params
    rho_w = 1e3
    cp_w = 4.218e3
    h = 1
    
    tendency = (SFCrad + OHF - LE + F)/((rho_w * cp_w * h))
    return tendency

def solve_surf_en_bal(F, OHF, RH, windspeed, dtas, A, B, lon, lonlim1=0, lonlim2=360):
    """Function to wrap solve_ivp for surf en balance and return key quantities."""
    # time integrate
    ndays = 600 
    t_span = (0, ndays*86400)
    times = np.linspace(t_span[0], t_span[1], ndays)

    # limit calculation to longitudes of interest
    F = F.where((lon>=lonlim1) & (lon<=lonlim2), drop=True)
    OHF = OHF.where((lon>=lonlim1) & (lon<=lonlim2), drop=True)
    RH = RH.where((lon>=lonlim1) & (lon<=lonlim2), drop=True)
    windspeed = windspeed.where((lon>=lonlim1) & (lon<=lonlim2), drop=True)
    dtas = dtas.where((lon>=lonlim1) & (lon<=lonlim2), drop=True)
    A = A.where((lon>=lonlim1) & (lon<=lonlim2), drop=True)
    B = B.where((lon>=lonlim1) & (lon<=lonlim2), drop=True)        
    
    T0 = 300.*np.ones_like(OHF)
    
    sol = solve_ivp(surf_en_tendency, t_span, T0, args=(F, OHF, RH, windspeed, dtas, A, B), 
                    method='Radau', t_eval=times)
    Tfinal = sol.y[:,-1]

    # diagnostic call of individual energy tendency terms
    SFCrad = sfc_rad(Tfinal, A, B)
    LE = lh_flux_dtas(Tfinal, RH, windspeed, dtas)
    
    # package solution 
    sol['T'] = Tfinal
    sol['SFCrad'] = SFCrad
    sol['OHF'] = OHF
    sol['LE'] = -LE
    sol['F'] = F
    sol['RH'] = RH
    sol['windspeed'] = windspeed
    sol['lon'] = lon.where((lon>=lonlim1) & (lon<=lonlim2), drop=True)
    return sol

def west_east_contrast(lon, var, lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east):
    """Calculate difference between west and east boxes."""
    return (np.nanmean(np.where((lon>lon_lim1_west) & (lon<lon_lim2_west), var, np.nan)) - 
            np.nanmean(np.where((lon>lon_lim1_east) & (lon<lon_lim2_east), var, np.nan)))

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_era5_variable(varname, filename, lon, lon_window=None, do_smoothing=True):
    """Load ERA5 variable and optionally smooth it."""
    fn = f'era5/{filename}'
    ds = xr.open_dataset(fn)
    
    # Get time-averaged variable
    if varname in ds:
        var = ds[varname].mean(dim='time')
    else:
        # Handle cases where variable name differs
        var = ds[list(ds.data_vars)[0]].mean(dim='time')
    
    # Apply smoothing if requested
    if do_smoothing and lon_window is not None:
        var = var.rolling(lon=lon_window, center=True).mean()
    
    # Calculate Pacific mean
    pac_mean = np.nanmean(np.where((lon>110) & (lon<280), var, np.nan))
    
    return var, pac_mean

def calculate_era5_trend(filename, varname, time_divisor=24*365.25, trend_years=44):
    """Calculate trends from ERA5 data."""
    fn = f'era5/{filename}'
    ds = xr.open_dataset(fn, decode_times=False)
    ds['time'] = ds.time / time_divisor
    
    pfit_results = ds.polyfit('time', deg=1)
    slope = pfit_results.sel(degree=1)
    
    # Get the appropriate coefficient
    coeff_name = f'{varname}_polyfit_coefficients'
    trend = slope[coeff_name] * trend_years
    
    return trend, ds.lon

def load_era5_data(do_smoothing=True, lon_window=20):
    """Load all ERA5 data needed for the analysis."""
    print("Loading ERA5 data...")
    
    # Load SST
    ds_sst = xr.open_dataset('era5/era5_sst_1979-2023_lat_pm5.nc')
    ds_sst_avg = ds_sst.mean(dim='time')
    Tref = ds_sst_avg.sst
    lon = ds_sst.lon
    
    # Load T2m
    ds_t2m = xr.open_dataset('era5/era5_t2m_1979-2023_lat_pm5.nc')
    Tas_ref = ds_t2m.mean(dim='time').t2m
    
    # Load surface fluxes
    sfc_sw, _ = load_era5_variable('ssr', 'era5_ssr_1979-2023_lat_pm5.nc', lon)
    sfc_lw, _ = load_era5_variable('str', 'era5_str_1979-2023_lat_pm5.nc', lon)
    sfc_sens, _ = load_era5_variable('sshf', 'era5_sshf_1979-2023_lat_pm5.nc', lon)
    sfc_latent, _ = load_era5_variable('slhf', 'era5_slhf_1979-2023_lat_pm5.nc', lon)
    
    # Calculate derived quantities
    sfcrad = sfc_sw + sfc_lw + sfc_sens
    era_oht = -1.0*(sfc_sw + sfc_lw + sfc_latent + sfc_sens)
    
    if do_smoothing:
        sfcrad = sfcrad.rolling(lon=lon_window, center=True).mean()
        era_oht = era_oht.rolling(lon=lon_window, center=True).mean()
        Trefsmooth = Tref.rolling(lon=lon_window, center=True).mean()
        Tassmooth = Tas_ref.rolling(lon=lon_window, center=True).mean()
        dtas = Trefsmooth - Tassmooth
    else:
        dtas = Tref - Tas_ref
        Trefsmooth = Tref
    
    # Load RH and windspeed
    RH, RH_pacmean = load_era5_variable('rh', 'era5_rh_1979-2023_lat_pm5.nc', 
                                        lon, lon_window, do_smoothing)
    windspeed, windspeed_pacmean = load_era5_variable('windspeed', 'era5_windspeed_1979-2023_lat_pm5.nc', 
                                                       lon, lon_window, do_smoothing)
    
    # Package results
    era5_data = {
        'lon': lon,
        'Tref': Tref,
        'Trefsmooth': Trefsmooth,
        'Tas_ref': Tas_ref,
        'dtas': dtas,
        'sfc_sw': sfc_sw,
        'sfc_lw': sfc_lw,
        'sfc_sens': sfc_sens,
        'sfc_latent': sfc_latent,
        'sfcrad': sfcrad,
        'OHF': era_oht,
        'OHF_pacmean': np.nanmean(np.where((lon>110) & (lon<280), era_oht, np.nan)),
        'RH': RH,
        'RH_pacmean': RH_pacmean,
        'windspeed': windspeed,
        'windspeed_pacmean': windspeed_pacmean
    }
    
    # Print diagnostics
    print_era5_diagnostics(era5_data)
    
    return era5_data

def print_era5_diagnostics(era5_data):
    """Print diagnostic information about ERA5 data."""
    lon = era5_data['lon']
    
    # Define regions
    lon_lim1_west, lon_lim2_west = 110, 180
    lon_lim1_east, lon_lim2_east = 180, 280
    
    print(f"ERA5 SST contrast: {west_east_contrast(lon, era5_data['Tref'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east):.2f}")
    print(f"ERA5 SST Pac mean: {np.nanmean(np.where((lon>110) & (lon<280), era5_data['Tref'], np.nan)):.2f}")
    print(f"OHF Pac mean: {era5_data['OHF_pacmean']:.2f}")
    print(f"OHF contrast: {west_east_contrast(lon, era5_data['OHF'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east):.2f}")

# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def create_base_parameters(era5_data, lon_struct):
    """Create base parameters for experiments."""
    lon = era5_data['lon']
    
    # Feedback parameters
    B0 = -1.8  # W/m2/K
    B = B0 + 0.0*lon_struct
    Basym = -1.8 + 4.0*lon_struct
    Basym_weak = -1.8 + (4.0/5)*lon_struct
    
    # Calculate A parameters
    Tref_for_A = era5_data['Trefsmooth']
    
    A = -(B*Tref_for_A - era5_data['sfcrad'])
    Aasym = -(Basym*Tref_for_A - era5_data['sfcrad'])
    Aasym_weak = -(Basym_weak*Tref_for_A - era5_data['sfcrad'])
    
    # Handle NaNs
    A = A.where(~np.isnan(A), 700)
    Aasym = Aasym.where(~np.isnan(Aasym), 700)
    Aasym_weak = Aasym_weak.where(~np.isnan(Aasym_weak), 700)
    
    # Control forcing (zero)
    F0 = xr.DataArray(0.*np.ones_like(lon), dims=['lon'], coords={'lon': lon})
    
    # Warm forcing (uniform 4 W/m2)
    Fconst = xr.DataArray(4.*np.ones_like(lon), dims=['lon'], coords={'lon': lon})
    
    return {
        'A': A,
        'B': B,
        'Aasym': Aasym,
        'Basym': Basym,
        'Aasym_weak': Aasym_weak,
        'Basym_weak': Basym_weak,
        'F0': F0,
        'Fconst': Fconst
    }

def create_perturbations(era5_data, base_params, lon_struct, experiment_type, **kwargs):
    """Create perturbations for different experiment types."""
    lon = era5_data['lon']

    # Start with base values
    pert = {
        'F': base_params['F0'] if 'control' in experiment_type else base_params['Fconst'],
        'OHF': era5_data['OHF'],
        'RH': era5_data['RH'],
        'windspeed': era5_data['windspeed'],
        'dtas': era5_data['dtas'],
        'A': base_params['A'],
        'B': base_params['B']
    }
    
    # Apply experiment-specific perturbations
    if experiment_type == 'OHF_asymmetry':
        factor = kwargs.get('factor', 1.1)
        pac_mean = era5_data['OHF_pacmean']*np.ones_like(era5_data['OHF'])
        pert['OHF'] = (era5_data['OHF'] - pac_mean)*factor + pac_mean
        
    elif experiment_type == 'forcing_asymmetry':
        Fmean = 4.
        Fasym_mag = kwargs.get('Fasym_mag', 2.5)
        pert['F'] = xr.DataArray(Fmean*np.ones_like(lon) + Fasym_mag*lon_struct,
                                dims=['lon'], coords={'lon': lon})
        
    elif experiment_type == 'feedback_asymmetry':
        pert['A'] = base_params['Aasym']
        pert['B'] = base_params['Basym']

    elif experiment_type == 'feedback_asymmetry_weak':
        pert['A'] = base_params['Aasym_weak']
        pert['B'] = base_params['Basym_weak']        
        
    elif experiment_type == 'RH_asymmetry':
        RH_change = kwargs.get('RH_change', 0.007)
        pert['RH'] = era5_data['RH'] + RH_change*lon_struct
        
    elif experiment_type == 'windspeed_asymmetry':
        factor = kwargs.get('factor', 1.08)
        pac_mean = era5_data['windspeed_pacmean']
        pert['windspeed'] = (era5_data['windspeed'] - pac_mean)*factor + pac_mean
        
    elif experiment_type == 'combined_weak':
        if kwargs.get('use_control_forcing', False):
            pert['F'] = base_params['F0']
            # Feedback asymmetry
            pert['A'] = base_params['Aasym_weak']
            pert['B'] = base_params['Basym_weak']
        else:
            # All perturbations at 20% amplitude
            # Forcing asymmetry
            Fasym_mag = 2.5/5
            pert['F'] = xr.DataArray(4.*np.ones_like(lon) + Fasym_mag*lon_struct,
                                     dims=['lon'], coords={'lon': lon})
            # OHF asymmetry
            pac_mean = era5_data['OHF_pacmean']*np.ones_like(era5_data['OHF'])
            pert['OHF'] = (era5_data['OHF'] - pac_mean)*1.02 + pac_mean
            # RH asymmetry
            pert['RH'] = era5_data['RH'] + (0.007/5)*lon_struct
            # Windspeed asymmetry
            ws_pac_mean = era5_data['windspeed_pacmean']
            pert['windspeed'] = (era5_data['windspeed'] - ws_pac_mean)*1.016 + ws_pac_mean
            # Feedback asymmetry
            pert['A'] = base_params['Aasym_weak']
            pert['B'] = base_params['Basym_weak']

        
        
    elif experiment_type == 'bjerknes':
        # OHF and windspeed coupled
        # OHF
        pac_mean = era5_data['OHF_pacmean']*np.ones_like(era5_data['OHF'])
        pert['OHF'] = (era5_data['OHF'] - pac_mean)*1.05 + pac_mean
        # Windspeed
        ws_pac_mean = era5_data['windspeed_pacmean']
        pert['windspeed'] = (era5_data['windspeed'] - ws_pac_mean)*1.05 + ws_pac_mean
        
    elif experiment_type == 'uniform_RH':
        pert['RH'] = xr.DataArray(0.8*np.ones_like(era5_data['RH']),
                                 dims=['lon'], coords={'lon': lon})

    if kwargs.get('use_control_forcing', False):
        pert['F'] = base_params['F0']
        
    return pert

def run_perturbation_experiment(control_params, warm_params, lon, 
                               lon_lims, box_lims):
    """Run control and perturbed experiments and return results."""
    
    # Unpack limits
    lon_lim1_west, lon_lim2_east = lon_lims
    lon_lim1_box_west, lon_lim2_box_west, lon_lim1_box_east, lon_lim2_box_east = box_lims
    
    # Solve control
    sol_control = solve_surf_en_bal(
        control_params['F'], control_params['OHF'], control_params['RH'],
        control_params['windspeed'], control_params['dtas'],
        control_params['A'], control_params['B'],
        lon, lon_lim1_west, lon_lim2_east
    )
    
    # Solve warm/perturbed
    sol_warm = solve_surf_en_bal(
        warm_params['F'], warm_params['OHF'], warm_params['RH'],
        warm_params['windspeed'], warm_params['dtas'],
        warm_params['A'], warm_params['B'],
        lon, lon_lim1_west, lon_lim2_east
    )
    
    # Calculate diagnostics
    Tcontrol = sol_control['T']
    Twarm = sol_warm['T']
    
    delT_control = west_east_contrast(sol_control['lon'], Tcontrol,
                                     lon_lim1_box_west, lon_lim2_box_west,
                                     lon_lim1_box_east, lon_lim2_box_east)
    delT_warm = west_east_contrast(sol_control['lon'], Twarm,
                                  lon_lim1_box_west, lon_lim2_box_west,
                                  lon_lim1_box_east, lon_lim2_box_east)
    
    delT_change = delT_warm - delT_control
    dT = sol_warm['T'] - sol_control['T']
    
    return {
        'sol_control': sol_control,
        'sol_warm': sol_warm,
        'delT_control': delT_control,
        'delT_warm': delT_warm,
        'delT_change': delT_change,
        'dT': dT,
        'dT_pacmean': np.nanmean(np.where((sol_control['lon']>110) & 
                                          (sol_control['lon']<280), dT, np.nan))
    }

def run_all_experiments(era5_data, experiments_config):
    """Run all experiments and return results."""
    
    # Create longitude structure
    lon_west = 145.
    lon_east = 230.
    lon_width = 40.
    lon_struct = lon_structure(era5_data['lon'], lon_east, lon_west, lon_width)
    
    # Create base parameters
    base_params = create_base_parameters(era5_data, lon_struct)
    
    # Define spatial limits
    lon_lims = (110, 280)  # Calculation domain
    box_lims = (110, 180, 180, 280)  # West-east contrast boxes
    
    results = {}
    
    for exp_name, exp_config in experiments_config.items():
        print(f"Running experiment: {exp_name}")
        
        # Create control parameters
        # For some experiments (like feedback_asymmetry), we need the control to use
        # the same A and B parameters as the perturbation
        if exp_config.get('needs_special_control', False):
            # Use the experiment type but with control forcing
            control_config = exp_config.copy()
            control_config['use_control_forcing'] = True
            control_params = create_perturbations(era5_data, base_params, lon_struct,
                                                exp_config['type'], **control_config)
            print('9999')
        else:
            # Standard control
            control_params = create_perturbations(era5_data, base_params, lon_struct,
                                                'control', **exp_config)

        # Create warm/perturbed parameters
        warm_params = create_perturbations(era5_data, base_params, lon_struct,
                                         exp_config['type'], **exp_config)

        #print(control_params)
        #print(warm_params)
        # Run experiment
        results[exp_name] = run_perturbation_experiment(
            control_params, warm_params, era5_data['lon'],
            lon_lims, box_lims
        )
    
    return results

# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Configuration
    do_era5_basic_state = 1
    do_smoothing = 1
    lon_window = 20
    
    # Load ERA5 data
    era5_data = load_era5_data(do_smoothing, lon_window)
    
    # Define experiments
    experiments_config = {
        'control': {'type': 'control'},
        'uniform_warming': {'type': 'warm'},
        'uniform_RH': {'type': 'uniform_RH', 'needs_special_control': True},
        'OHF_asymmetry': {'type': 'OHF_asymmetry', 'factor': 1.1},
        'forcing_asymmetry': {'type': 'forcing_asymmetry', 'Fasym_mag': 2.5},
        'feedback_asymmetry': {'type': 'feedback_asymmetry', 'needs_special_control': True},
        'RH_asymmetry': {'type': 'RH_asymmetry', 'RH_change': 0.007},
        'windspeed_asymmetry': {'type': 'windspeed_asymmetry', 'factor': 1.08},
        'combined_weak': {'type': 'combined_weak', 'needs_special_control': True},
        'bjerknes': {'type': 'bjerknes'}
    }

    # Run all experiments
    results = run_all_experiments(era5_data, experiments_config)

    # Calculate additional weak perturbation experiments for linearity test
    weak_experiments = {
        'OHF_weak': {'type': 'OHF_asymmetry', 'factor': 1.02},
        'forcing_weak': {'type': 'forcing_asymmetry', 'Fasym_mag': 2.5/5},
        'feedback_weak': {'type': 'feedback_asymmetry_weak', 'needs_special_control': True},
        'RH_weak': {'type': 'RH_asymmetry', 'RH_change': 0.007/5},
        'windspeed_weak': {'type': 'windspeed_asymmetry', 'factor': 1.016}
    }
    
    # Run weak experiments
    weak_results = run_all_experiments(era5_data, weak_experiments)
    
    # Combine results
    results.update(weak_results)

    # Print summary of results
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    for exp_name, result in results.items():
        print(f"\n{exp_name}:")
        print(f"  ΔT contrast change: {result['delT_change']:.3f} K")
        print(f"  Pacific mean ΔT: {result['dT_pacmean']:.3f} K")

    # Save results as pkl
    #import pickle
    #with open('experiment_results.pkl', 'wb') as f:
    #    pickle.dump({'era5_data': era5_data, 'results': results}, f)

    # Save results to NetCDF format
    # Create xarray datasets for each experiment
    experiment_datasets = {}
    
    for exp_name, result in results.items():
        # Create dataset for this experiment
        ds = xr.Dataset({
            'T_control': (['lon'], result['sol_control']['T'].data),
            'T_warm': (['lon'], result['sol_warm']['T'].data),
            'dT': (['lon'], result['dT'].data),
            'RH_control': (['lon'], result['sol_control']['RH'].data),
            'RH_warm': (['lon'], result['sol_warm']['RH'].data),
            'windspeed_control': (['lon'], result['sol_control']['windspeed'].data),
            'windspeed_warm': (['lon'], result['sol_warm']['windspeed'].data),            
            'SFCrad_control': (['lon'], result['sol_control']['SFCrad'].data),
            'SFCrad_warm': (['lon'], result['sol_warm']['SFCrad'].data),
            'OHF_control': (['lon'], result['sol_control']['OHF'].data),
            'OHF_warm': (['lon'], result['sol_warm']['OHF'].data),            
            'LE_control': (['lon'], result['sol_control']['LE'].data),
            'LE_warm': (['lon'], result['sol_warm']['LE'].data),
            'F_control': (['lon'], result['sol_control']['F'].data),
            'F_warm': (['lon'], result['sol_warm']['F'].data),
        }, coords={
            'lon': result['sol_control']['lon']
        })
        
        ds.attrs['experiment_name'] = exp_name
        
        experiment_datasets[exp_name] = ds
    
    # Save each experiment as a separate NetCDF file
    for exp_name, ds in experiment_datasets.items():
        filename = f'exper_results/experiment_results_{exp_name}.nc'
        ds.to_netcdf(filename)
        print(f"Saved {exp_name} results to {filename}")
    
    # Also save ERA5 data as reference
    era5_ds = xr.Dataset({
        'Tref': (['lon'], era5_data['Tref'].data),
        'Tref_smooth': (['lon'], era5_data['Trefsmooth'].data),
        'T2m_ref': (['lon'], era5_data['Tas_ref'].data),
        'dtas': (['lon'], era5_data['dtas'].data),
        'sfcrad': (['lon'], era5_data['sfcrad'].data),
        'sfc_sw': (['lon'], era5_data['sfc_sw'].data),
        'sfc_lw': (['lon'], era5_data['sfc_lw'].data),
        'sfc_sens': (['lon'], era5_data['sfc_sens'].data),
        'sfc_latent': (['lon'], era5_data['sfc_latent'].data),        
        'OHF': (['lon'], era5_data['OHF'].data),
        'RH': (['lon'], era5_data['RH'].data),
        'windspeed': (['lon'], era5_data['windspeed'].data),
    }, coords={
        'lon': era5_data['lon']
    })
    
    era5_ds.to_netcdf('era5_reference_data.nc')
    
