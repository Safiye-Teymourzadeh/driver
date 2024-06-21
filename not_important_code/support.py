import numpy as np

deltaMag = 0.05
magnitude_intervals = np.arange(14, 24, deltaMag)

def get_stat(big_survey, small_survey, flux_type, regions):
    countBigByRegion = {}
    countSmallByRegion = {}
    for region_name, region in regions.items():
        big_survey_region = filter_objects_by_region(big_survey, region)
        small_survey_region = filter_objects_by_region(small_survey, region)
        counts_of_big_survey = get_interval_counts_from_surveys(survey_name=big_survey_region , flux_type=flux_type)
        counts_of_small_survey = get_interval_counts_from_surveys(survey_name=small_survey_region , flux_type=flux_type)
        countBigByRegion[region_name] = counts_of_big_survey
        countSmallByRegion[region_name] = counts_of_small_survey
    return countBigByRegion, countSmallByRegion

def filter_objects_by_region(survey_name, region):
    RAcen_mask = (survey_name['RAcen'] >= region['RAcen'][0]) & (survey_name['RAcen'] <= region['RAcen'][1])
    DECcen_mask = (survey_name['DECcen'] >= region['DECcen'][0]) & (survey_name['DECcen'] <= region['DECcen'][1])
    return survey_name[RAcen_mask & DECcen_mask]

def get_interval_counts_from_surveys(survey_name, flux_type):
    flux_list = read_flux(survey_name,flux_type)
    apparent_magnitude = calculate_apparent_magnitude(flux_list)
    interval_counts = count_objects_in_magnitude_intervals(apparent_magnitude, magnitude_intervals)
    return interval_counts

def calculate_apparent_magnitude(flux_data):
    ap_mag_rt = 8.9 - 2.5 * np.log10(flux_data)
    return ap_mag_rt

#function to count the objects in each interval
def count_objects_in_magnitude_intervals(ap_mag_data, intervals):
    counts = np.histogram(ap_mag_data, bins=intervals)[0]
    return counts

def read_flux(survey_name, flux_type):
    return survey_name[flux_type]
