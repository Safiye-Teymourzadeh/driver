#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Define the path to your FITS file
fits_path = "/home/farnoosh/farnoosh/Master_Thesis_all/Data/GAMA/gkvScienceCatv02/gkvScienceCatv02.fits"

# Check if the columns exist
if 'flux_rt' not in data.names or 'NQ' not in data.names:
    print("The expected columns are not present in the FITS file.")
else:
    # Check if there are entries where NQ >= 3
    if np.sum(data['NQ'] >= 3) == 0:
        print("There are no entries where NQ >= 3.")
    else:
        # Define magnitude bins (adjust this based on your actual data range)
        bins = np.linspace(np.min(data['flux_rt']), np.max(data['flux_rt']), 100)

        # Compute histograms
        all_counts_r, _ = np.histogram(data['flux_rt'], bins=bins)
        reliable_counts_r, _ = np.histogram(data['flux_rt'][data['NQ'] >= 3], bins=bins)

        # Plot histograms
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        ax[0].bar(bins[:-1], all_counts_r, width=np.diff(bins)[0], align='edge', alpha=0.7)
        ax[0].set_title('All Galaxies (r-band)')
        ax[0].set_xlabel('Magnitude')
        ax[0].set_ylabel('Count')

        ax[1].bar(bins[:-1], reliable_counts_r, width=np.diff(bins)[0], align='edge', alpha=0.7, color='g')
        ax[1].set_title('Reliable Galaxies (r-band)')
        ax[1].set_xlabel('Magnitude')
        ax[1].set_ylabel('Count')

        plt.tight_layout()
        plt.show()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Define the path to your FITS file
fits_path = "/home/farnoosh/farnoosh/Master_Thesis_all/Data/GAMA/gkvScienceCatv02/gkvScienceCatv02.fits"

# Load the data from the .fits file
with fits.open('/home/farnoosh/farnoosh/Master_Thesis_all/Data/GAMA/gkvScienceCatv02/gkvScienceCatv02.fits') as hdul:
    data = hdul[1].data

# Define magnitude bins
bins = np.linspace(10, 20, 100)  # Adjust bin edges as necessary

# Compute histograms for all galaxies and galaxies with reliable redshift measurements (NQ >= 3)
all_counts_r, _ = np.histogram(data['flux_rt'], bins=bins)
reliable_counts_r, _ = np.histogram(data['flux_rt'][data['NQ'] >= 3], bins=bins)

# Plot histograms to inspect the data distribution
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].bar(bins[:-1], all_counts_r, width=np.diff(bins)[0], align='edge', alpha=0.7, label="All galaxies")
ax[0].set_title('Histogram of All Galaxies (r band)')
ax[0].set_xlabel('Magnitude')
ax[0].set_ylabel('Counts')
ax[0].legend()

ax[1].bar(bins[:-1], reliable_counts_r, width=np.diff(bins)[0], align='edge', alpha=0.7, color='g', label="Galaxies with NQ >= 3")
ax[1].set_title('Histogram of Galaxies with Reliable Redshift Measurements (r band)')
ax[1].set_xlabel('Magnitude')
ax[1].set_ylabel('Counts')
ax[1].legend()

plt.tight_layout()
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Load the data
with fits.open('/home/farnoosh/farnoosh/Master_Thesis_all/Data/GAMA/gkvScienceCatv02/gkvScienceCatv02.fits') as hdul:
    data = hdul[1].data


# Define magnitude bins
bins = np.linspace(10, 20, 100)  # Adjust bin edges as necessary

# Compute histograms for all galaxies and galaxies with reliable redshift measurements (NQ >= 3)
all_counts_r, _ = np.histogram(data['flux_rt'], bins=bins)
reliable_counts_r, _ = np.histogram(data['flux_rt'][data['NQ'] >= 3], bins=bins)

# Calculate the cumulative counts
cumulative_all_r = np.cumsum(all_counts_r)
cumulative_reliable_r = np.cumsum(reliable_counts_r)

cumulative_r = np.zeros_like(cumulative_all_r)
differential_r = np.zeros_like(all_counts_r)

# Explicitly compute the cumulative and differential completeness
for i in range(len(cumulative_r)):
    if cumulative_all_r[i] != 0:
        cumulative_r[i] = cumulative_reliable_r[i] / cumulative_all_r[i]
    if all_counts_r[i] != 0:
        differential_r[i] = reliable_counts_r[i] / all_counts_r[i]

# Plot the results
fig, axs = plt.subplots(1, 3, figsize=(15, 6))

# Left panel
axs[0].plot(bins[:-1], cumulative_r, color='r')
axs[0].axhline(0.5, linestyle='--', color='gray')
axs[0].axhline(0.9, linestyle=':', color='gray')
axs[0].set_xlabel('Magnitude')
axs[0].set_ylabel('Cumulative Completeness')
axs[0].set_title('Cumulative Completeness (r band)')

# Center panel (Zoomed in)
axs[1].plot(bins[:-1], cumulative_r, color='r')
axs[1].axhline(0.95, linestyle='--', color='gray')
axs[1].axhline(0.98, linestyle=':', color='gray')
axs[1].set_ylim(0.9, 1.0)
axs[1].set_xlabel('Magnitude')
axs[1].set_title('Zoomed Cumulative Completeness (r band)')

# Right panel
axs[2].bar(bins[:-1], differential_r, width=np.diff(bins)[0], align='edge', color='r', alpha=0.7)
axs[2].axhline(0.5, linestyle='--', color='gray')
axs[2].axhline(0.9, linestyle=':', color='gray')
axs[2].set_xlabel('Magnitude')
axs[2].set_ylabel('Differential Completeness')
axs[2].set_title('Differential Completeness (r band)')

plt.tight_layout()
plt.show()


# In[ ]:




