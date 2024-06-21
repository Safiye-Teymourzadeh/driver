import numpy as np
import healpy as hp
import pandas as pd

"""
what is this function for?

input:
dataframe of galaxies with information:
- mass
- comoving distance
- completeness
- region
output:
galaxy mass function
"""

NSIDE=4096
Z_MAX = 0.1 # ~430 Mpc
DEX = 0.25  # width of the mass bins
MASS_BINS = 10**np.arange(5,12.25,DEX)
HUBBLE_CST = 0.7
REGIONS_OF_SKY = {
    # 'G02': {'RAcen': (30.20, 38.80), 'DECcen': (-10.25, -3.72)},
    'G09': {'RAcen': (129.0, 141.0), 'DECcen': (-2.0, 3.0)},
    'G12': {'RAcen': (174.0, 186.0), 'DECcen': (-3.0, 2.0)},
    'G15': {'RAcen': (211.5, 223.5), 'DECcen': (-2.0, 3.0)},
    'G23': {'RAcen': (339.0, 351.0), 'DECcen': (-35.0, -30.0)},
}


# Function to get the weighted histogram of the objects of each region
def get_weighted_mass_histogram(input_mass_completeness_dataframe: pd.DataFrame, region_name: str):
    """
    this is the main function
    """
    filtered_by_region_dataframe = input_mass_completeness_dataframe[input_mass_completeness_dataframe['region'] == region_name]
    mass_column = filtered_by_region_dataframe['mstar']
    completeness_column = filtered_by_region_dataframe['completeness']
    volume_richard_curve = get_region_volume(region_name=region_name, mass_list=mass_column)
    weight = np.log(10)/ (volume_richard_curve * completeness_column * DEX)
    return np.histogram(mass_column, MASS_BINS, weights=weight)[0]


# Function to get the volume of each region
def get_region_volume(region_name: str, mass_list: list):
    region = REGIONS_OF_SKY[region_name]
    region_area = calculate_patch_area(patch=region, nside=4096)
    average_pixel_area = 4 * np.pi / (12 * NSIDE**2)
    total_area_sphere = hp.nside2npix(NSIDE) * average_pixel_area
    fraction_region = region_area / total_area_sphere
    return calculate_volume([calculate_richard_curve(np.log10(mass)) for mass in mass_list], fraction_region)


# Function to calculate the area for a given patch
def calculate_patch_area(patch: dict, nside: int=4096):
    RA_min, RA_max = np.deg2rad(patch['RAcen'])
    DEC_min, DEC_max = np.deg2rad(patch['DECcen'])

    # Calculate the pixel indices for the given patch
    pix_indices = np.arange(hp.nside2npix(nside))
    pix_indices_patch = pix_indices[
        (hp.pixelfunc.pix2ang(nside, pix_indices)[0] >= np.pi/2 - DEC_max) &
        (hp.pixelfunc.pix2ang(nside, pix_indices)[0] <= np.pi/2 - DEC_min) &
        (hp.pixelfunc.pix2ang(nside, pix_indices)[1] >= RA_min) &
        (hp.pixelfunc.pix2ang(nside, pix_indices)[1] <= RA_max)
    ]

    # Calculate the area of the given patch using the average solid angle of a pixel
    average_pixel_area = 4 * np.pi / (12 * nside**2)
    patch_area = len(pix_indices_patch) * average_pixel_area

    return patch_area


# Function to calculate volume (Mpc^3)
def calculate_volume(radius, fraction):
    return 4 / 3 * np.pi * np.power(radius, 3)  * fraction


# Function to calculate error
def calculate_error(y_data, number_of_galaxies):
    return np.sqrt((y_data * number_of_galaxies**(-1/2))**2 + (y_data * 0.043)**2)


def calculate_richard_curve(log_mass):
    """
    Richards curve from GAMA based on Table 5, Eq. 2 from Driver et al. 2022
    :param log_mass: log10 of stellar mass limit
    :return: co moving distance  in Mpc
    """
    A = -0.016
    K = 2742.0
    C = 0.9412
    B = 1.1483
    M = 11.815
    nu = 1.691
    y = A + (K - A) / (C + np.exp(-B * (log_mass - M))) ** (1 / nu)
    return y
