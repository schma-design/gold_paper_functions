import numpy as np
from datetime import datetime, timedelta
import re
import os
import tarfile

def generate_custom_bins(latitude_range, longitude_range):

    longitude_bins = []
    current_lon = longitude_range[0]
    while current_lon < longitude_range[1]:
        step = 4.0 if -95 <= current_lon < 3 else 7.0
        longitude_bins.append(current_lon)
        current_lon += step
    longitude_bins.append(longitude_range[1])
    longitude_bins = np.array(longitude_bins)

    latitude_bins = []
    current_lat = latitude_range[0]
    while current_lat < latitude_range[1]:
        step = 3.8 if -35 <= current_lat < 35 else 6.0
        latitude_bins.append(current_lat)
        current_lat += step
    latitude_bins.append(latitude_range[1])
    latitude_bins = np.array(latitude_bins)

    return latitude_bins, longitude_bins

def compute_average(sum, npoints):
    avg = np.divide(sum, npoints, where=npoints > 0, out=np.full_like(sum, np.nan))
    avg[avg < 0.01] = np.nan  
    return avg
    
def create_meshgrid(lon_bins, lat_bins):
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    return np.meshgrid(lon_centers, lat_centers)

def combine_averages(avg1, avg2, npoints1, npoints2):
    both_valid = ~np.isnan(avg1) & ~np.isnan(avg2)
    only_one_valid = np.isnan(avg1) ^ np.isnan(avg2)
    combined = np.full_like(avg1, np.nan)
    combined[both_valid] = (avg1[both_valid] + avg2[both_valid]) / 2
    combined[only_one_valid] = np.where(np.isnan(avg1[only_one_valid]), avg2[only_one_valid], avg1[only_one_valid])
    combined[(npoints1 + npoints2) == 0] = np.nan
    return combined

def find_closest_files(start_times, directory):
    file_pattern = re.compile(r'3DALL_t(\d{6})\_(\d{6})\.bin')
    offset_minutes=11.5

    # Store parsed file datetimes
    file_time_map = {}

    # Scan directory for matching files
    for filename in os.listdir(directory):
        match = file_pattern.match(filename)
        if match:
            date_str, time_str = match.groups()
            try:
                file_datetime = datetime.strptime(date_str + time_str, '%y%m%d%H%M%S')
                full_path = os.path.join(directory, filename)
                file_time_map[file_datetime] = full_path
            except ValueError:
                print(f"Skipping file with unparseable date: {filename}")

    # Ensure there are files to compare
    if not file_time_map:
        raise ValueError(f"No matching files found in {directory}.")

    # Sort file datetimes
    sorted_file_times = sorted(file_time_map.keys())

    closest_files = []

    for start_time in start_times:
        # Add offset to the start time
        target_time = start_time + timedelta(minutes=offset_minutes)

        # Find the closest file time
        closest_time = min(sorted_file_times, key=lambda ft: abs(ft - target_time))
        closest_file_path = file_time_map[closest_time]
        closest_files.append(closest_file_path)

    return closest_files

def extract_nc_from_tar(tar_path, extract_dir=None):

    if not tarfile.is_tarfile(tar_path):
        raise ValueError(f"{tar_path} is not a valid tar file.")

    if extract_dir is None:
        extract_dir = os.path.dirname(tar_path)

    extracted_files = []

    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.nc'):
                tar.extract(member, path=extract_dir, filter='data')
                extracted_path = os.path.join(extract_dir, member.name)
                extracted_files.append(os.path.abspath(extracted_path))

    if not extracted_files:
        raise FileNotFoundError("No .nc files found in the tar archive.")

    return extracted_files

def is_tar_file(filename):
    return filename.endswith(".tar")

def is_nc_file(filename):
    return filename.endswith(".nc")

def gather_nc_files(filelist):
    nc_files = []

    for file in filelist:
        if is_tar_file(file):
            extracted_files = extract_nc_from_tar(file)
            nc_files.extend(extracted_files)
        elif is_nc_file(file):
            nc_files.append(file)
        else:
            print(f"Skipping unsupported file type: {file}")
    
    nc_files.sort()
    return nc_files

def find_closest_filesv2(time_list, directory):

    file_pattern = re.compile(r'3DALL_t(\d{6})\_(\d{6})\.(bin|nc)')

    # Store parsed file datetimes
    file_time_map = {}

    # Scan directory for matching files
    for filename in os.listdir(directory):
        match = file_pattern.match(filename)
        if match:
            date_str, time_str, _ = match.groups()
            try:
                file_datetime = datetime.strptime(date_str + time_str, '%y%m%d%H%M%S')
                full_path = os.path.join(directory, filename)
                file_time_map[file_datetime] = full_path
            except ValueError:
                print(f"Skipping file with unparseable date: {filename}")

    # Ensure there are files to compare
    if not file_time_map:
        raise ValueError(f"No matching files found in {directory}.")

    # Sort file datetimes
    sorted_file_times = sorted(file_time_map.keys())
    time_to_file_map = {}

    for t in time_list:
        if t is None:
            continue

        closest_time = min(sorted_file_times, key=lambda ft: abs(ft - t))
        time_to_file_map[t] = file_time_map[closest_time]

    return time_to_file_map