import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.colors as mcolors

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
lon_lim1_west, lon_lim2_west = 110, 180
lon_lim1_east, lon_lim2_east = 180, 280

    
def west_east_contrast(lon, var, lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east):
    """Calculate difference between west and east boxes."""
    return (np.nanmean(np.where((lon>lon_lim1_west) & (lon<lon_lim2_west), var, np.nan)) - 
            np.nanmean(np.where((lon>lon_lim1_east) & (lon<lon_lim2_east), var, np.nan)))

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

def setup_figure_style():
    """Set up consistent figure styling."""
    return {
        'legfontsize': 7.5,
        'tickfontsize': 8.5,
        'smfontsize': 10,
        'lgfontsize': 12
    }

def add_longitude_labels(ax):
    """Add standard longitude tick labels."""
    xtick_labels = ['120$^\circ$ E','','180$^\circ$','','120$^\circ$ W','']
    plt.xticks(np.arange(120, 300, 30), xtick_labels)

def add_experiment_label(ax, xlim1, xlim2, delT_change, y_pos=1.0):
    """Add standard delta SST label to a plot."""
    str1 = f'$\Delta SST_{{wp}} - \Delta SST_{{ep}}$: {delT_change:.2f}'
    ax.text(xlim1+.35*(xlim2-xlim1), y_pos, str1)

# ============================================================================
# FIGURE 1: ERA5 SUMMARY
# ============================================================================

def plot_figure_1(era5_data, results):
    """Figure 1: ERA5 summary figure."""
    fig = plt.figure(1, figsize=(6.5, 6.5/1.618*1.5))
    styles = setup_figure_style()
    plt.rc('font', size=styles['tickfontsize'])
    
    xlim1, xlim2 = 110, 280
    lon = era5_data['lon']

    # panel a is climo SST map
    ax = fig.add_subplot(321,projection=ccrs.Robinson(central_longitude=-180))

    ds_sst_mean = xr.open_dataset('era5/era5_sst_1979-2023mean.nc')
    ds_sst_trend = xr.open_dataset('era5/era5_sst_1979-2023trend.nc')

    levs = np.arange(286,306,2)
    ctrs = ax.contourf(ds_sst_mean.lon,ds_sst_mean.lat,ds_sst_mean.sst, transform=ccrs.PlateCarree(), levels=levs, cmap='RdYlBu_r')

    # Plot boxes for Wills 2022 averaging region
    ax.plot([lon_lim1_east, lon_lim2_east],[-5,-5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim1_east, lon_lim2_east],[5,5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim1_east, lon_lim1_east],[-5,5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim2_east, lon_lim2_east],[-5,5],'--',color='gray', transform=ccrs.PlateCarree())

    ax.plot([lon_lim1_west, lon_lim2_west],[-5,-5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim1_west, lon_lim2_west],[5,5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim1_west, lon_lim1_west],[-5,5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim2_west, lon_lim2_west],[-5,5],'--',color='gray', transform=ccrs.PlateCarree())    
    
    ax.coastlines()
    ax.set_extent([lon_lim1_west-30, lon_lim2_east+30, -35, 35], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True,alpha=.4,
                      xlocs=[120, 180, -120, -60],
                      ylocs=[-30, -10, 10, 30])
    gl.top_labels = False
    gl.right_labels = False
    ax.text(lon_lim1_west-75, 38, 'a', fontweight='bold',transform=ccrs.PlateCarree())

    fig.colorbar(ctrs,orientation='horizontal')
    
    plt.title('SST (K)', fontweight='bold')
    
    # panel b is SST trend map
    ax = fig.add_subplot(322,projection=ccrs.Robinson(central_longitude=-180))

    ds_sst_trend = xr.open_dataset('era5/era5_sst_1979-2023trend.nc')

    levs = np.arange(-1,2.2,.2)
    norm_trend = mcolors.TwoSlopeNorm(vmin=np.min(levs),vcenter=0,vmax=np.max(levs))
    ctrs = ax.contourf(ds_sst_mean.lon,ds_sst_mean.lat,ds_sst_trend.sst_polyfit_coefficients, transform=ccrs.PlateCarree(), levels=levs,cmap='RdBu_r',norm=norm_trend)

    ax.plot([lon_lim1_east, lon_lim2_east],[-5,-5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim1_east, lon_lim2_east],[5,5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim1_east, lon_lim1_east],[-5,5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim2_east, lon_lim2_east],[-5,5],'--',color='gray', transform=ccrs.PlateCarree())


    ax.plot([lon_lim1_west, lon_lim2_west],[-5,-5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim1_west, lon_lim2_west],[5,5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim1_west, lon_lim1_west],[-5,5],'--',color='gray', transform=ccrs.PlateCarree())
    ax.plot([lon_lim2_west, lon_lim2_west],[-5,5],'--',color='gray', transform=ccrs.PlateCarree())    
    
    ax.coastlines()
    ax.set_extent([lon_lim1_west-30, lon_lim2_east+30, -35, 35], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True,alpha=.4,
                      xlocs=[120, 180, -120, -60],
                      ylocs=[-30, -10, 10, 30])
    gl.top_labels = False
    gl.right_labels = False
    
    ax.text(lon_lim1_west-75, 38, 'b', fontweight='bold',transform=ccrs.PlateCarree()) 

    fig.colorbar(ctrs,orientation='horizontal')

    plt.title('Trend in SST (K per 44 year)', fontweight='bold')
    
    # Panel c: SST line plot
    ax = fig.add_subplot(323)
    plt.plot(lon, era5_data['Tref'], 'k-', label='$SST$, ERA5')
    plt.plot(results['control']['lon'], 
             results['control']['T_control'], 'k:', label='$SST$, Approx. En. Bal.')
    
    # Calculate biases
    #lon_lim1_west, lon_lim2_east = 110, 280
    #Trefl = era5_data['Tref'].where((lon>=lon_lim1_west) & (lon<=lon_lim2_east), drop=True)
    #lonl = lon.where((lon>=lon_lim1_west) & (lon<=lon_lim2_east), drop=True)
    #T_model = results['control']['T_control']
    #print(f"West Pacific bias: {np.nanmean(np.where((lonl>110) & (lonl<160), Trefl-T_model, np.nan)):.2f}")
    #print(f"Central Pacific bias: {np.nanmean(np.where((lonl>180) & (lonl<240), Trefl-T_model, np.nan)):.2f}")
    
    ax.set_xlim([xlim1, xlim2])
    ax.set_ylim([297, 304.5])
    plt.legend(frameon=False, loc='lower left', fontsize=styles['legfontsize'])
    add_longitude_labels(ax)
    plt.text(xlim1-.15*(xlim2-xlim1), 304.5, 'c', fontweight='bold')
    plt.title('SST (K)', fontweight='bold')

    # Panel d: SST trend
    ax = fig.add_subplot(324)
    sst_trend, trend_lon = calculate_era5_trend('era5_sst_1979-2023_lat_pm5.nc', 'sst')
    plt.plot(trend_lon, sst_trend, 'k-', label='$\Delta SST$')
    
    # Calculate trend statistics
    delT_trend = west_east_contrast(trend_lon, sst_trend, lon_lim1_west, lon_lim2_west, 
                                    lon_lim1_east, lon_lim2_east)
    pac_mean_trend = np.nanmean(np.where((trend_lon>110) & (trend_lon<280), sst_trend, np.nan))
    print(f"ERA5 SST trend contrast: {delT_trend:.3f}")
    print(f"ERA5 SST Pacific mean trend: {pac_mean_trend:.3f}")
    
    ax.set_xlim([xlim1, xlim2])
    ax.set_ylim([-0.15, 1.15])
    plt.legend(frameon=False, loc='upper right', fontsize=styles['legfontsize'])
    add_longitude_labels(ax)
    plt.text(xlim1-.15*(xlim2-xlim1), 1.15, 'd', fontweight='bold')
    plt.title('Trend in SST (K per 44 year)', fontweight='bold')
    plt.xlabel('Longitude', fontweight='bold')

    # Panel e: Energy fluxes
    ax = fig.add_subplot(325)
    plt.plot(lon, era5_data['sfc_latent'], 'k-', label='LE')
    plt.plot(lon, era5_data['sfc_sens'], '-', color='orange', label='SH')
    plt.plot(lon, era5_data['sfc_sw'], 'r-', label='Net SW')
    plt.plot(lon, era5_data['sfc_lw'], 'r--', label='Net LW')
    plt.plot(lon, era5_data['OHF'], 'b-', label='OHF')
    
    ax.set_xlim([xlim1, xlim2])
    ax.set_ylim([-150, 270])
    plt.text(xlim1-.15*(xlim2-xlim1), 270, 'e', fontweight='bold')
    plt.title('Energy Flux (W m$^{-2}$)', fontweight='bold')
    plt.legend(frameon=False, fontsize=styles['legfontsize'], loc='upper left', bbox_to_anchor=(.5,1))
    add_longitude_labels(ax)
    plt.xlabel('Longitude', fontweight='bold')
    
    
    fig.tight_layout()
    fig.savefig('figs/fig1.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# FIGURE 2: INDIVIDUAL PERTURBATIONS
# ============================================================================

def plot_figure_2(era5_data, results):
    """Figure 2: Individual perturbation experiments."""
    fig = plt.figure(2, figsize=(6.5, 6.5/1.618*2))
    styles = setup_figure_style()
    plt.rc('font', size=styles['tickfontsize'])
    
    xlim1, xlim2 = 110, 280
    
    experiments = [
        ( 'uniform_warming', 'Control'),
        ( 'OHF_asymmetry', 'Perturbed OHF'),
        ( 'forcing_asymmetry', 'Asymmetric Radiative Forcing'),
        ( 'feedback_asymmetry', 'Asymmetric Radiative Feedback')
    ]
    
    for idx, (exper, title) in enumerate(experiments):
        
        # Temperature subplot
        ax_T = fig.add_subplot(4, 2, 2*idx + 1)
        plt.plot(results[exper]['lon'], results[exper]['dT'], 'r-', label='$\Delta T_s$')
        
        if idx == 0:
            # Also plot uniform RH case for control
            plt.plot(results['uniform_RH']['lon'], 
                     results['uniform_RH']['dT'], 'r:', label='$\Delta T_s, \, RH = 80\%$')
        
        add_experiment_label(ax_T, xlim1, xlim2, west_east_contrast(results[exper]['lon'],results[exper]['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east))
        ax_T.set_xlim([xlim1, xlim2])
        ax_T.set_ylim([0, 1.2])
        
        if idx == 0:
            plt.legend(frameon=False, loc='lower left')
            plt.title('Change in SST (K)', fontweight='bold')
        
        if idx == 3:
            plt.xlabel('Longitude', fontweight='bold')
        
        add_longitude_labels(ax_T)
        plt.text(xlim1-.15*(xlim2-xlim1), 1.2, chr(97+2*idx), fontweight='bold')
        
        # Energy flux subplot
        ax_E = fig.add_subplot(4, 2, 2*idx + 2)
        
        plt.plot(results[exper]['lon'], results[exper]['F_warm'] - results[exper]['F_control'], 'c-', label='$\mathcal{F}$')
        plt.plot(results[exper]['lon'], results[exper]['OHF_warm'] - results[exper]['OHF_control'], 'b-', label='$\Delta OHF$')
        plt.plot(results[exper]['lon'], results[exper]['SFCrad_warm'] - results[exper]['SFCrad_control'], 'r-', label='$\lambda \Delta SST$')
        plt.plot(results[exper]['lon'], results[exper]['LE_warm'] - results[exper]['LE_control'], 'k-', label='$\Delta LE$')
        
        if idx == 0:
            # Also plot uniform RH case with control
            exper = 'uniform_RH'
            plt.plot(results[exper]['lon'], results[exper]['F_warm'] - results[exper]['F_control'], 'c:')
            plt.plot(results[exper]['lon'], results[exper]['OHF_warm'] - results[exper]['OHF_control'], 'b:')
            plt.plot(results[exper]['lon'], results[exper]['SFCrad_warm'] - results[exper]['SFCrad_control'], 'r:')
            plt.plot(results[exper]['lon'], results[exper]['LE_warm'] - results[exper]['LE_control'], 'k:')

            plt.legend(frameon=False, loc='upper left')
            plt.title('Change in Energy Flux (W m$^{-2}$)', fontweight='bold')
        
        ax_E.set_xlim([xlim1, xlim2])
        ax_E.set_ylim([-7.5, 7.5])
        
        if idx == 3:
            plt.xlabel('Longitude', fontweight='bold')
        
        add_longitude_labels(ax_E)
        plt.text(xlim1-.15*(xlim2-xlim1), 7.5, chr(98+2*idx), fontweight='bold')
        
        # Add title with appropriate offset
        title_offsets = [-.28, -.41, -.68, -.72]
        plt.text(xlim1+title_offsets[idx]*(xlim2-xlim1), 10.4 if idx==0 else 9.4, 
                 title, fontweight='bold', fontsize=styles['smfontsize'])
    
    plt.subplots_adjust(hspace=.45, wspace=.3)
    fig.savefig('figs/fig2.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# FIGURE 3: RH AND WINDSPEED
# ============================================================================

def plot_figure_3(era5_data, results):
    """Figure 3: RH and windspeed focus."""
    fig = plt.figure(3, figsize=(6.5, 6.5/1.618*(3/2)))
    styles = setup_figure_style()
    plt.rc('font', size=styles['tickfontsize'])
    
    xlim1, xlim2 = 110, 280
    lon = era5_data['lon']
    
    # Panel a: RH climatology and perturbation
    exper = 'RH_asymmetry'
    ax = fig.add_subplot(321)
    plt.plot(lon, era5_data['RH']*100, 'm-', label='$RH$ (%)')
    # Get the perturbed RH from the experiment
    RH_pert = results['RH_asymmetry']['RH_warm']
    plt.plot(results['RH_asymmetry']['lon'], RH_pert*100, 'm--', label='$RH\'$ (%)')
    
    ax.set_ylim([76, 86])
    plt.legend(frameon=False, loc='upper left', fontsize=styles['legfontsize'])
    ax.set_xlim([xlim1, xlim2])
    add_longitude_labels(ax)
    plt.text(xlim1-.15*(xlim2-xlim1), 86, 'a', fontweight='bold')
    
    # Panel b: Windspeed climatology and perturbation
    ax = fig.add_subplot(322)
    plt.plot(lon, era5_data['windspeed'], 'g-', label='$|\mathbf{u}|$ (m s$^{-1}$)')
    # Get the perturbed windspeed
    ws_pert = results['windspeed_asymmetry']['windspeed_warm']
    plt.plot(results['windspeed_asymmetry']['lon'], ws_pert, 'g--', 
             label='$|\mathbf{u}|\'$ (m s$^{-1}$)')
    
    plt.legend(frameon=False, loc='lower right', fontsize=styles['legfontsize'])
    ax.set_xlim([xlim1, xlim2])
    add_longitude_labels(ax)
    ax.set_ylim([3, 6.5])
    plt.text(xlim1-.15*(xlim2-xlim1), 6.5, 'b', fontweight='bold')
    plt.text(xlim1-.59*(xlim2-xlim1), 7.2, 'ERA5 RH and Wind Speed', 
             fontweight='bold', fontsize=styles['smfontsize'])
    
    # Panel c: RH perturbation temperature response
    ax = fig.add_subplot(323)
    result = results['RH_asymmetry']
    plt.plot(result['lon'], result['dT'], 'r-', label='$\Delta T_s$')
    add_experiment_label(ax, xlim1, xlim2, west_east_contrast(results[exper]['lon'],results[exper]['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east))
    
    ax.set_xlim([xlim1, xlim2])
    ax.set_ylim([0, 1.2])
    plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.05,0.0))
    add_longitude_labels(ax)
    plt.text(xlim1-.15*(xlim2-xlim1), 1.2, 'c', fontweight='bold')
    plt.title('Change in SST (K)', fontweight='bold')
    
    # Panel d: RH perturbation energy fluxes
    ax = fig.add_subplot(324)

    plt.plot(results[exper]['lon'], results[exper]['F_warm'] - results[exper]['F_control'], 'c-', label='$\mathcal{F}$')
    plt.plot(results[exper]['lon'], results[exper]['OHF_warm'] - results[exper]['OHF_control'], 'b-', label='$\Delta OHF$')
    plt.plot(results[exper]['lon'], results[exper]['SFCrad_warm'] - results[exper]['SFCrad_control'], 'r-')
    plt.plot(results[exper]['lon'], results[exper]['LE_warm'] - results[exper]['LE_control'], 'k-')
    
    plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.05,-0.08))
    add_longitude_labels(ax)
    ax.set_xlim([xlim1, xlim2])
    ax.set_ylim([-7.5, 7.5])
    plt.text(xlim1-.15*(xlim2-xlim1), 7.5, 'd', fontweight='bold')
    plt.title('Change in Energy Flux (W m$^{-2}$)', fontweight='bold')
    plt.text(xlim1-.385*(xlim2-xlim1), 10.2, 'Perturbed RH', 
             fontweight='bold', fontsize=styles['smfontsize'])
    
    # Panel e: Windspeed perturbation temperature response
    ax = fig.add_subplot(325)
    exper = 'windspeed_asymmetry'
    result = results['windspeed_asymmetry']
    plt.plot(result['lon'], result['dT'], 'r-', label='$\Delta T_s$')
    add_experiment_label(ax, xlim1, xlim2, west_east_contrast(results[exper]['lon'],results[exper]['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east))

    ax.set_xlim([xlim1, xlim2])
    ax.set_ylim([0, 1.2])
    plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.05,0.0))
    add_longitude_labels(ax)
    plt.text(xlim1-.15*(xlim2-xlim1), 1.2, 'e', fontweight='bold')
    plt.xlabel('Longitude', fontweight='bold')
    
    # Panel f: Windspeed perturbation energy fluxes
    ax = fig.add_subplot(326)

    plt.plot(results[exper]['lon'], results[exper]['SFCrad_warm'] - results[exper]['SFCrad_control'], 'r-', label='$\lambda \Delta SST$')
    plt.plot(results[exper]['lon'], results[exper]['LE_warm'] - results[exper]['LE_control'], 'k-', label='$\Delta LE$')
    plt.plot(results[exper]['lon'], results[exper]['F_warm'] - results[exper]['F_control'], 'c-')
    plt.plot(results[exper]['lon'], results[exper]['OHF_warm'] - results[exper]['OHF_control'], 'b-')
    
    plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.05,-0.08))
    add_longitude_labels(ax)
    ax.set_xlim([xlim1, xlim2])
    ax.set_ylim([-7.5, 7.5])
    plt.text(xlim1-.15*(xlim2-xlim1), 7.5, 'f', fontweight='bold')
    plt.xlabel('Longitude', fontweight='bold')
    plt.text(xlim1-.55*(xlim2-xlim1), 10.2, 'Perturbed Wind Speed', 
             fontweight='bold', fontsize=styles['smfontsize'])
    
    plt.subplots_adjust(hspace=.44, wspace=.3)
    fig.savefig('figs/fig3.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# FIGURE 4: COMBINED PERTURBATIONS
# ============================================================================

def plot_figure_4(results):
    """Figure 4: Combined perturbations."""
    fig = plt.figure(4, figsize=(6.5, 6.5/1.618))
    styles = setup_figure_style()
    plt.rc('font', size=styles['tickfontsize'])
    
    xlim1, xlim2 = 110, 280
    
    # Panel a: Combined weak perturbations - temperature
    ax = fig.add_subplot(221)
    exper = 'combined_weak'
    plt.plot(results['combined_weak']['lon'], results['combined_weak']['dT'], 'r-', label='$\Delta T_s$')
    add_experiment_label(ax, xlim1, xlim2, west_east_contrast(results[exper]['lon'],results[exper]['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east))
    
    
    ax.set_xlim([xlim1, xlim2])
    ax.set_ylim([0, 1.2])
    plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.05,0.0))
    add_longitude_labels(ax)
    plt.text(xlim1-.15*(xlim2-xlim1), 1.2, 'a', fontweight='bold')
    plt.title('Change in SST (K)', fontweight='bold')
    
    # Panel b: Combined weak perturbations - energy fluxes
    ax = fig.add_subplot(222)

    plt.plot(results[exper]['lon'], results[exper]['F_warm'] - results[exper]['F_control'], 'c-', label='$\mathcal{F}$')
    plt.plot(results[exper]['lon'], results[exper]['OHF_warm'] - results[exper]['OHF_control'], 'b-', label='$\Delta OHF$')
    plt.plot(results[exper]['lon'], results[exper]['SFCrad_warm'] - results[exper]['SFCrad_control'], 'r-')
    plt.plot(results[exper]['lon'], results[exper]['LE_warm'] - results[exper]['LE_control'], 'k-')

    
    plt.legend(frameon=False, loc='lower right', fontsize=styles['legfontsize'])
    add_longitude_labels(ax)
    ax.set_xlim([xlim1, xlim2])
    ax.set_ylim([-7.5, 7.5])
    plt.text(xlim1-.15*(xlim2-xlim1), 7.5, 'b', fontweight='bold')
    plt.title('Change in Energy Flux (W m$^{-2}$)', fontweight='bold')
    plt.text(xlim1-.58*(xlim2-xlim1), 10.2, 'Combined Perturbation', 
             fontweight='bold', fontsize=styles['smfontsize'])
    
    # Panel c: Bjerknes - temperature
    ax = fig.add_subplot(223)
    exper = 'bjerknes'
    plt.plot(results[exper]['lon'], results[exper]['dT'], 'r-', label='$\Delta T_s$')
    add_experiment_label(ax, xlim1, xlim2, west_east_contrast(results[exper]['lon'],results[exper]['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east))
    
    ax.set_xlim([xlim1, xlim2])
    ax.set_ylim([0, 1.2])
    plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.05,0.0))
    add_longitude_labels(ax)
    plt.text(xlim1-.15*(xlim2-xlim1), 1.2, 'c', fontweight='bold')
    plt.xlabel('Longitude', fontweight='bold')
    
    # Panel d: Bjerknes - energy fluxes
    ax = fig.add_subplot(224)

    plt.plot(results[exper]['lon'], results[exper]['SFCrad_warm'] - results[exper]['SFCrad_control'], 'r-', label='$\lambda \Delta SST$')
    plt.plot(results[exper]['lon'], results[exper]['LE_warm'] - results[exper]['LE_control'], 'k-', label='$\Delta LE$')
    plt.plot(results[exper]['lon'], results[exper]['F_warm'] - results[exper]['F_control'], 'c-')
    plt.plot(results[exper]['lon'], results[exper]['OHF_warm'] - results[exper]['OHF_control'], 'b-')

    ax.set_xlim([xlim1, xlim2])
    plt.legend(frameon=False, loc='lower right', fontsize=styles['legfontsize'])
    add_longitude_labels(ax)
    ax.set_ylim([-7.5, 7.5])
    plt.text(xlim1-.15*(xlim2-xlim1), 7.5, 'd', fontweight='bold')
    plt.xlabel('Longitude', fontweight='bold')
    plt.text(xlim1-.51*(xlim2-xlim1), 9.4, 'OHF and Wind Speed', 
             fontweight='bold', fontsize=styles['smfontsize'])
    
    plt.subplots_adjust(hspace=.44, wspace=.3)
    fig.savefig('figs/fig4.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# FIGURE 5: BAR CHART SUMMARY
# ============================================================================

def plot_figure_5(results):
    """Figure 5: Bar chart summary of west minus east deltaT with linearity and energy fluxes ."""
    fig = plt.figure(5, figsize=(6.5/2*1.5, 6.5/1.618*1.5))
    
    ax = fig.add_subplot(211)

    plt.plot([-.5,8.],[0,0],'k:')
    # Plot full perturbations
    plt.plot(0, west_east_contrast(results['control']['lon'],results['control']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'ko', markersize=8)
    plt.plot(1, west_east_contrast(results['OHF_asymmetry']['lon'],results['OHF_asymmetry']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'bo', markersize=8)
    plt.plot(2, west_east_contrast(results['forcing_asymmetry']['lon'],results['forcing_asymmetry']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'co', markersize=8)
    plt.plot(3, west_east_contrast(results['feedback_asymmetry']['lon'],results['feedback_asymmetry']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'ro', markersize=8)
    plt.plot(4, west_east_contrast(results['RH_asymmetry']['lon'],results['RH_asymmetry']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'mo', markersize=8)            
    plt.plot(5, west_east_contrast(results['windspeed_asymmetry']['lon'],results['windspeed_asymmetry']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'go', markersize=8)    
    plt.plot(6.5, west_east_contrast(results['bjerknes']['lon'],results['bjerknes']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), '*',color='gray', markersize=8)
    plt.plot(7.7, west_east_contrast(results['combined_weak']['lon'],results['combined_weak']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'k*', markersize=8)    
        
    # Plot weak perturbations
    dx = 0.2
    plt.plot(1-dx, west_east_contrast(results['OHF_weak']['lon'],results['OHF_weak']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'b<', fillstyle='none', markersize=6)
    plt.plot(2-dx, west_east_contrast(results['forcing_weak']['lon'],results['forcing_weak']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'c<', fillstyle='none', markersize=6)
    plt.plot(3-dx, west_east_contrast(results['feedback_weak']['lon'],results['feedback_weak']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'r<', fillstyle='none', markersize=6)
    plt.plot(4-dx, west_east_contrast(results['RH_weak']['lon'],results['RH_weak']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'm<', fillstyle='none', markersize=6)            
    plt.plot(5-dx, west_east_contrast(results['windspeed_weak']['lon'],results['windspeed_weak']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'g<', fillstyle='none', markersize=6)    

    # Plot sum of weak perturbations
    weak_sum = west_east_contrast(results['OHF_weak']['lon'],results['OHF_weak']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east) + west_east_contrast(results['forcing_weak']['lon'],results['forcing_weak']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east) + west_east_contrast(results['feedback_weak']['lon'],results['feedback_weak']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east) + west_east_contrast(results['RH_weak']['lon'],results['RH_weak']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east) + west_east_contrast(results['windspeed_weak']['lon'],results['windspeed_weak']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east)
    plt.plot(7.5-dx, weak_sum, 'k<', markersize=6)
    
    # Plot 1/5 of full perturbations (linearity test)
    plt.plot(1+dx, west_east_contrast(results['OHF_asymmetry']['lon'],results['OHF_asymmetry']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east)/5, 'bo', fillstyle='none', markersize=6)
    plt.plot(2+dx, west_east_contrast(results['forcing_asymmetry']['lon'],results['forcing_asymmetry']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east)/5, 'co', fillstyle='none', markersize=6)
    plt.plot(3+dx, west_east_contrast(results['feedback_asymmetry']['lon'],results['feedback_asymmetry']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east)/5, 'ro', fillstyle='none', markersize=6)
    plt.plot(4+dx, west_east_contrast(results['RH_asymmetry']['lon'],results['RH_asymmetry']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east)/5, 'mo', fillstyle='none', markersize=6)            
    plt.plot(5+dx, west_east_contrast(results['windspeed_asymmetry']['lon'],results['windspeed_asymmetry']['dT'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east)/5, 'go', fillstyle='none', markersize=6)    
    
    plt.xticks([0, 1, 2, 3, 4, 5, 6.5, 7.5], [])

    # alternating gray background
    for idx in range(0,8,1):
        offset = 0
        if idx > 5:
            offset = .5
        if np.mod(idx,2) == 1:
            plt.axvspan(idx-.5+offset, idx+.5+offset, alpha=0.3, color='gray')
            
    plt.title('$\Delta$ SST$_{wp}$ - $\Delta$ SST$_{ep}$', fontweight='bold')
    plt.ylabel('K', fontweight='bold')    

    plt.text(-0.5-.15*8.5, .5, 'a', fontweight='bold')
    
    ax.set_xlim([-.5, 8])        
    ax.set_ylim([-0.05, .5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b',markeredgecolor='b', markersize=8, label='Full'),
        Line2D([0], [0], marker='<', color='w', markerfacecolor='none',
               markeredgecolor='b', markersize=6, label='20%'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
               markeredgecolor='b', markersize=6, label='Full/5')
    ]
    ax.legend(handles=legend_elements, loc='center left', fontsize=6) #,frameon=False)


    ax = fig.add_subplot(212)

    plt.plot([-.5,8.],[0,0],'k:')
    experiments = [
        ( 'uniform_warming', 'Control'),
        ( 'OHF_asymmetry', 'Perturbed OHF'),
        ( 'forcing_asymmetry', 'Asymmetric Radiative Forcing'),
        ( 'feedback_asymmetry', 'Asymmetric Radiative Feedback'),
        ( 'RH_asymmetry', 'Asymmetric Radiative Feedback'),
        ( 'windspeed_asymmetry', 'Asymmetric Radiative Feedback'),
        ( 'bjerknes', 'Asymmetric Radiative Feedback'),
        ( 'combined_weak', 'Asymmetric Radiative Feedback')               
    ]
    
    for idx, (exper, title) in enumerate(experiments):
        offset = 0
        if idx > 5:
            offset = .5
            
        plt.plot(idx - .15 + offset, west_east_contrast(results[exper]['lon'],results[exper]['OHF_warm'] - results[exper]['OHF_control'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'bo', markersize=6)
        plt.plot(idx - .05 + offset, west_east_contrast(results[exper]['lon'],results[exper]['F_warm'] - results[exper]['F_control'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'co', markersize=6)
        plt.plot(idx + .05 + offset, west_east_contrast(results[exper]['lon'],results[exper]['SFCrad_warm'] - results[exper]['SFCrad_control'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'ro', markersize=6)        
        plt.plot(idx + .15 + offset, west_east_contrast(results[exper]['lon'],results[exper]['LE_warm'] - results[exper]['LE_control'], lon_lim1_west, lon_lim2_west, lon_lim1_east, lon_lim2_east), 'ko', markersize=6)
        if np.mod(idx,2) == 1:
            plt.axvspan(idx-.5+offset, idx+.5+offset, alpha=0.3, color='gray')  

    # Set labels
    xtick_labels = ['Control', 
                    'OHF asym\n$+20\%$ K$^{-1}$',
                    '$\mathcal{F}$ asym\n$3.3$ W m$^{-2}$',
                    '$\lambda_{SFC}$ asym\n$4$ W m$^{-2}$ K$^{-1}$',
                    'RH asym\n$0.5\%$ K$^{-1}$',
                    '$|u|$ asym\n$+16\%$ K$^{-1}$',
                    'OHF $+10\%$ K$^{-1}$\n& $|u|$ $+10\%$ K$^{-1}$',
                    'All at 20%\namplitude']

    plt.xticks([0, 1, 2, 3, 4, 5, 6.5, 7.5], xtick_labels, rotation=-75, ha='center',fontsize=8)
    plt.yticks(np.arange(-3,4,1))
    
    # Set colors for each x-tick label to match marker colors
    ax = plt.gca()
    labels = ax.get_xticklabels()
    colors = ['k', 'b', 'c', 'r', 'm', 'g', 'gray', 'k']  # black, blue, cyan, red, magenta, green, gray, black
    for label, color in zip(labels, colors):
        label.set_color(color)

    plt.text(-0.5-.15*8.5, 3.6, 'b', fontweight='bold')
    ax.set_xlim([-.5, 8])        
    ax.set_ylim([-3.6, 3.6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

        
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='c',markeredgecolor='c', markersize=6, label='$\mathcal{F}$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b',markeredgecolor='b', markersize=6, label='$\Delta OHF$')
    ]
    legend_elements2 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r',markeredgecolor='r', markersize=6, label='$\lambda \Delta SST$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k',markeredgecolor='k', markersize=6, label='$\Delta LE$')
    ]
    legend_elements_all = legend_elements + legend_elements2

    plt.title('$\Delta$ En Flux$_{wp}$ - $\Delta$ En Flux$_{ep}$', fontweight='bold')
    plt.ylabel('W m$^{-2}$', fontweight='bold')

    plt.subplots_adjust(hspace=.3, wspace=.3)

    fig.savefig('figs/fig5.pdf', bbox_inches='tight')
    plt.close()

# ============================================================================
# ADDITIONAL PLOTTING UTILITIES
# ============================================================================

def create_summary_table(results):
    """Create a summary table of all experiment results."""
    print("\n" + "="*70)
    print("DETAILED EXPERIMENT RESULTS")
    print("="*70)
    print(f"{'Experiment':<25} {'ΔT contrast (K)':<20} {'Pac mean ΔT (K)':<20}")
    print("-"*70)
    
    for exp_name, result in results.items():
        print(f"{exp_name:<25} {result['delT_change']:>15.3f}      {result['dT_pacmean']:>15.3f}")
    
    print("\n" + "="*70)
    print("LINEARITY TEST")
    print("="*70)
    
    # Compare weak perturbations with scaled full perturbations
    experiments = ['OHF', 'forcing', 'feedback', 'RH', 'windspeed']
    exp_map = {
        'OHF': 'OHF_asymmetry',
        'forcing': 'forcing_asymmetry', 
        'feedback': 'feedback_asymmetry',
        'RH': 'RH_asymmetry',
        'windspeed': 'windspeed_asymmetry'
    }
    
    print(f"{'Perturbation':<15} {'Weak (K)':<12} {'Full/5 (K)':<12} {'Ratio':<10}")
    print("-"*50)
    
    for exp in experiments:
        weak_key = f"{exp}_weak"
        full_key = exp_map[exp]
        weak_val = results[weak_key]['delT_change']
        full_val = results[full_key]['delT_change'] / 5
        ratio = weak_val / full_val if full_val != 0 else float('inf')
        print(f"{exp:<15} {weak_val:>10.3f}  {full_val:>10.3f}  {ratio:>8.2f}")
    
    # Sum test
    weak_sum = sum(results[f"{exp}_weak"]['delT_change'] for exp in experiments)
    combined = results['combined_weak']['delT_change']
    print(f"\n{'Sum of weak':<15} {weak_sum:>10.3f}")
    print(f"{'Combined weak':<15} {combined:>10.3f}")
    print(f"{'Ratio':<15} {combined/weak_sum if weak_sum != 0 else float('inf'):>10.2f}")

# ============================================================================
# MAIN PLOTTING SCRIPT
# ============================================================================

def main():
    """Main function to load results and create all plots."""
    
    # Load saved results
    print("Loading saved results...")
    #with open('experiment_results.pkl', 'rb') as f:
    #    data = pickle.load(f)
    
    #era5_data = data['era5_data']
    #results = data['results']

    era5_data = xr.open_dataset('era5_reference_data.nc')

    experiment_names= [ 'control',
                         'uniform_warming',
	                 'uniform_RH',
                         'OHF_asymmetry',
                         'forcing_asymmetry',
                         'feedback_asymmetry',
	                 'RH_asymmetry',
                         'windspeed_asymmetry',                      
                         'combined_weak',
                         'bjerknes',
                         'OHF_weak',
                         'forcing_weak',
                         'feedback_weak',
	                 'RH_weak',
                         'windspeed_weak']
    results = {}
    for e in experiment_names:
        exper = 'exper_results/experiment_results_' + e + '.nc'
        print(exper)
        results[e] = xr.open_dataset(exper)
        pac_mean = np.nanmean(np.where((results[e].lon>110) & (results[e].lon<280), results[e].dT, np.nan))
        print('pac_mean dT:', pac_mean)
        
    print("Creating figures...")

    # Create all figures
    plot_figure_1(era5_data, results)
    
    plot_figure_2(era5_data, results)
    
    plot_figure_3(era5_data, results)
    
    plot_figure_4(results)
    
    plot_figure_5(results)
    
    # Create summary table
    # only works when loading results from pkl
    #create_summary_table(results)
    

if __name__ == "__main__":
    main()
