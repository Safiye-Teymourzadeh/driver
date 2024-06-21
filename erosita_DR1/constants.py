# constant file:

import numpy as np

H0 = 70
c = 3e5
NSIDE=4096
# Z_MAX = 0.1 # ~430 Mpc
Z_MAX = 0.4 # ~1551.5 Mpc

# todo: check for definition of Mass_Bin and Dex
DEX = 0.25  # width of the mass bins
MASS_BINS = 10**np.arange(5,12.25,DEX)

REGIONS_OF_SKY = {
    # 'G02': {'RAcen': (30.20, 38.80), 'DECcen': (-10.25, -3.72)},
    'G09': {'RAcen': (129.0, 141.0), 'DECcen': (-2.0, 3.0)},
    'G12': {'RAcen': (174.0, 186.0), 'DECcen': (-3.0, 2.0)},
    'G15': {'RAcen': (211.5, 223.5), 'DECcen': (-2.0, 3.0)},
    'G23': {'RAcen': (339.0, 351.0), 'DECcen': (-35.0, -30.0)},
}
