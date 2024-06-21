import astropy.io.fits as fits
import matplotlib.pyplot as plt

from completeness import *
from mass_function import *


# All the objects that is observed in the Sciencegkv catalog, with or without spectroscopic redshift, but no Z and NQ in the columns. this catalog has stars and ambitious objects.
all_sciencegkv_galaxies_raw = fits.open('/home/farnoosh/farnoosh/Master_Thesis_all/Data/GAMA/gkvScienceCatv02/gkvScienceCatv02.fits')[1].data

# the Objects that have Spec-Z, they can be also stars or ambitious. It is a merged catalog of gkvScience and StellarMass.
galaxy_SpecMass_catalog_raw = fits.open('/home/farnoosh/farnoosh/Master_Thesis_all/Data/GAMA/merged/StellarMass-gkvScience/mergedStellarMass-gkvScience')[1].data



# Masks
'''
We now start by adopting the magnitude limit of rKiDSDR4 = 19.65 mag, as discussed in Section 2.2, for the
four primary GAMA regions (G09, G12, G15, and G23). This sub-sample contains 205 540 galaxies
if apply NQ>2, then 95.1 percent of the galaxies will have reliable redshift, as Driver said.
this catalog does NOT have: mass but it has: Z, NQ, SC. with these three masks the number of glaxies reduces to 153 601
'''
all_sciencegkv_galaxies_raw_Basic_Masks = (
    # (all_sciencegkv_galaxies_raw['Z'] != -9.999) &
    # (all_sciencegkv_galaxies_raw['SC'] > 7) &
    # (all_sciencegkv_galaxies_raw['NQ'] > 2) &
    (all_sciencegkv_galaxies_raw['uberclass'] == 1) &   # galaxy
    (all_sciencegkv_galaxies_raw['duplicate'] == 0) &   # unique object
    (all_sciencegkv_galaxies_raw['mask'] == False) &
    (all_sciencegkv_galaxies_raw['starmask'] == False) &
    (all_sciencegkv_galaxies_raw['flux_rt'] >= 5.011928e-05)   # maximum magnitude of 19.65 in r-band
)
all_sciencegkv_galaxies_Basic_Masks = all_sciencegkv_galaxies_raw[all_sciencegkv_galaxies_raw_Basic_Masks]


COMPLETENESS_MASKS = (
    (galaxy_SpecMass_catalog_raw['uberclass'] == 1 ) & #galaxy
    (galaxy_SpecMass_catalog_raw['duplicate'] == 0) &  # unique object
    (galaxy_SpecMass_catalog_raw['mask'] == False) &
    (galaxy_SpecMass_catalog_raw['NQ'] > 2) &
    (galaxy_SpecMass_catalog_raw['SC'] > 7) &   # 95% redshift completeness limit
    (galaxy_SpecMass_catalog_raw['starmask'] == False) &
    (galaxy_SpecMass_catalog_raw['Z'] != 0 ) &
    (galaxy_SpecMass_catalog_raw['Z'] < 0.4) & # z_max= 0.1 ~ 430 Mpc
    # (galaxy_SpecMass_catalog_raw['flux_rt'] > 10**-4.3) &   # estimate mag = 22.24
    (galaxy_SpecMass_catalog_raw['flux_rt'] > 5.011928e-05) &  ## maximum magnitude of 19.65 in r-band
    (galaxy_SpecMass_catalog_raw['mstar'] > 0)
)
completeness_Masked_of_SpecMass_catalog = galaxy_SpecMass_catalog_raw[COMPLETENESS_MASKS]


'''
We now limit our sample to the nearby Universe by imposing a redshift cutoff of z < 0.1.
'''
MASS_HISTOGRAM_MASKS = (
    (galaxy_SpecMass_catalog_raw['uberclass'] == 1 ) &  # galaxy
    (galaxy_SpecMass_catalog_raw['duplicate'] == 0) &   # unique object
    (galaxy_SpecMass_catalog_raw['mask'] == False) &
    (galaxy_SpecMass_catalog_raw['starmask'] == False) &
    (galaxy_SpecMass_catalog_raw['mstar'] > 0) &
    (galaxy_SpecMass_catalog_raw['mstar'] < 10**14) &
    (galaxy_SpecMass_catalog_raw['NQ'] > 2) &
    (galaxy_SpecMass_catalog_raw['SC'] > 7) &   # 95% redshift completeness limit
    (galaxy_SpecMass_catalog_raw['Z'] != 0) &
    (galaxy_SpecMass_catalog_raw['Z'] < 0.4) & # z_max= 0.1 ~ 430 Mpc
    (galaxy_SpecMass_catalog_raw['flux_rt'] >= 5.011928e-05)   # maximum magnitude of 19.65 in r-band
)
mass_histogram_Masked_cat_from_SpecMass = galaxy_SpecMass_catalog_raw[MASS_HISTOGRAM_MASKS]




completeness_all_df = pd.DataFrame()
# Loop over regions and create completeness DataFrames
for region_name, region_params in REGIONS_OF_SKY.items():
    completeness_region_df = create_completeness_dataframe(big_survey=all_sciencegkv_galaxies_Basic_Masks,
                                                           small_survey=completeness_Masked_of_SpecMass_catalog,
                                                           flux_type='flux_rt',
                                                           region=region_name)
    completeness_region_df['region'] = region_name
    completeness_all_df = pd.concat([completeness_all_df, completeness_region_df], ignore_index=True)




mass_histogram_Masked_cat_from_SpecMass_dataframe = pd.DataFrame()
mass_histogram_Masked_cat_from_SpecMass_dataframe['uberID'] = mass_histogram_Masked_cat_from_SpecMass['uberID'].byteswap().newbyteorder()
mass_histogram_Masked_cat_from_SpecMass_dataframe['RA'] = mass_histogram_Masked_cat_from_SpecMass['RAcen'].byteswap().newbyteorder()
mass_histogram_Masked_cat_from_SpecMass_dataframe['DEC'] = mass_histogram_Masked_cat_from_SpecMass['DECcen'].byteswap().newbyteorder()
mass_histogram_Masked_cat_from_SpecMass_dataframe['mstar'] = mass_histogram_Masked_cat_from_SpecMass['mstar'].byteswap().newbyteorder()
mass_histogram_Masked_cat_from_SpecMass_dataframe['z'] = mass_histogram_Masked_cat_from_SpecMass['Z'].byteswap().newbyteorder()
mass_histogram_Masked_cat_from_SpecMass_dataframe['mstar'] = mass_histogram_Masked_cat_from_SpecMass['mstar'].byteswap().newbyteorder()
mass_histogram_Masked_cat_from_SpecMass_dataframe['comovingdist'] = mass_histogram_Masked_cat_from_SpecMass['comovingdist'].byteswap().newbyteorder()


full_mass_completeness_dataframe = pd.merge(mass_histogram_Masked_cat_from_SpecMass_dataframe, completeness_all_df, on='uberID', how='left')




region_colors = {'G09': 'red', 'G12': 'lime', 'G15': 'blue', 'G23': 'cyan'}
fig, ax = plt.subplots(figsize=(24, 16))
for region_name in REGIONS_OF_SKY.keys():
    mass_hist, error = get_weighted_mass_histogram(input_mass_completeness_dataframe=full_mass_completeness_dataframe, region_name=region_name)

    plt.errorbar(
        MASS_BINS[:-1],
        mass_hist,
        yerr=error,
        label=region_name,
        color=region_colors.get(region_name),
    )

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Stellar Mass ($M_{\odot} \ h_{70}^{-2}$)', fontsize=16)
plt.ylabel('Number Density ($\mathrm{Mpc}^{-3} \ dex^{-1} \ h_{70}^{3}$)', fontsize=16)
plt.xlim((10**6.2, 10**12))
plt.ylim((1e-5, 1e0))
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()