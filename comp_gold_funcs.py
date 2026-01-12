from netCDF4 import Dataset
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from comp_gen_misc import generate_custom_bins, create_meshgrid, compute_average, combine_averages

def get_scan_pair_times(valid_points, nscans):
    """
    Compute earliest (start) and latest (end) valid times for each scan pair.
    valid_points: output list from extract_valid_[var]_points()
    nscans: total number of scans in data
    """
    start_times = []
    end_times = []

    # Group by scan index i
    scan_groups = defaultdict(list)
    for p in valid_points:
        scan_groups[p["i"]].append(p["time"])

    for i in range(0, nscans, 2):
        if scan_groups[i]:
            start_time = min(scan_groups[i])
        else:
            start_time = None

        if (i + 1) in scan_groups and scan_groups[i + 1]:
            end_time = max(scan_groups[i + 1])
        else:
            end_time = None

        start_times.append(start_time)
        end_times.append(end_time)

    return start_times, end_times

def extract_gold_on2(file_path):
    dataset = Dataset(file_path, 'r')
    gold_on2_vars = {
        "longitude_on2": dataset.variables['longitude'][:],
        "latitude_on2": dataset.variables['latitude'][:],
        "time_data_on2": dataset.variables['time_utc'][:],
        "on2": dataset.variables['on2'][:],
        "n_scans": dataset.variables['on2'].shape[0]}
    return gold_on2_vars

def extract_valid_on2_points(on2, time_data, lat_grid, lon_grid):

    nscans, nlats, nlons = on2.shape
    results = []

    for i in range(nscans):
        for j in range(nlats):
            for k in range(nlons):
                on2_val = on2[i, j, k]
                
                # Optional: Skip fill/masked values
                if np.isnan(on2_val):
                    continue

                time_entry = time_data[i, j, k]  # Shape: (max_string_len,)
                if b'*' in time_entry:
                    continue

                try:
                    valid_chars = [entry.decode('utf-8') for entry in time_entry[:20] if entry not in [b'--', b'*', b' ']]
                    time_str = ''.join(valid_chars)
                    time_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
                except (ValueError, UnicodeDecodeError):
                    continue

                lat = lat_grid[j, k]
                lon = lon_grid[j, k]

                results.append({
                    "i": i,
                    "j": j,
                    "k": k,
                    "lat": lat,
                    "lon": lon,
                    "time": time_obj,
                    "on2": on2_val
                })

    return results

def extract_gold_tdisk(file_path):
    dataset = Dataset(file_path, 'r')
    gold_tdisk_vars = {
        "longitude_tdisk": dataset.variables['longitude'][:],
        "latitude_tdisk": dataset.variables['latitude'][:],
        "time_data_tdisk": dataset.variables['time_utc'][:],
        "temperature": dataset.variables['tdisk'][:],
        "n_scans_tdisk": dataset.variables['tdisk'].shape[0],
        "sza": dataset.variables['solar_zenith_angle']}
    return gold_tdisk_vars

def extract_valid_tdisk_points(tdisk, time_data, lat_grid, lon_grid, sza):

    nscans, nlats, nlons = tdisk.shape
    results = []

    for i in range(nscans):
        for j in range(nlats):
            for k in range(nlons):
                tdisk_val = tdisk[i, j, k]
                
                if np.isnan(tdisk_val):
                    continue

                sza_val = sza[i, j, k]
                if np.isnan(sza_val):
                    continue

                time_entry = time_data[i, j, k]  # Shape: (max_string_len,)
                if b'*' in time_entry:
                    continue

                try:
                    valid_chars = [entry.decode('utf-8') for entry in time_entry[:20] if entry not in [b'--', b'*', b' ']]
                    time_str = ''.join(valid_chars)
                    time_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
                except (ValueError, UnicodeDecodeError):
                    continue

                lat = lat_grid[j, k]
                lon = lon_grid[j, k]

                results.append({
                    "i": i,
                    "j": j,
                    "k": k,
                    "lat": lat,
                    "lon": lon,
                    "time": time_obj,
                    "tdisk": tdisk_val,
                    "sza": sza_val
                })

    return results

def gold_on2_processing(gold_on2_vars):

    n_scans = gold_on2_vars["n_scans"]
    on2 = gold_on2_vars["on2"]
    latitude_on2 = gold_on2_vars["latitude_on2"]
    longitude_on2 = gold_on2_vars["longitude_on2"]
    time_data_on2 = gold_on2_vars["time_data_on2"]

    start_times_on2 = []
    end_times_on2 = []
    for i in range(0, n_scans, 2):  
        scan_index1 = i
        scan_index2 = i + 1 if i + 1 < n_scans else i 

        # Define longitude and latitude ranges
        longitude_range = (-120, 60)
        latitude_range = (-60, 65)

        latitude_bins, longitude_bins = generate_custom_bins(latitude_range, longitude_range)

        # Initialize grids to store ON2 sums and counts
        n_latitude_buckets = len(latitude_bins) - 1
        n_longitude_buckets = len(longitude_bins) - 1
        on2_sum1 = np.zeros((n_latitude_buckets, n_longitude_buckets))
        on2_sum2 = np.zeros_like(on2_sum1)
        npoints1 = np.zeros_like(on2_sum1)
        npoints2 = np.zeros_like(on2_sum1)

        # Extract time data for each scan
        valid_times_scan1 = []
        valid_times_scan2 = []

        for j in range(longitude_on2.shape[0]):  
            for k in range(longitude_on2.shape[1]):  
                lon = longitude_on2[j, k]  # Assign separately
                lat = latitude_on2[j, k]  

                if longitude_range[0] <= lon <= longitude_range[1] and latitude_range[0] <= lat <= latitude_range[1]:
                    lon_idx = np.digitize(lon, longitude_bins) - 1
                    lat_idx = np.digitize(lat, latitude_bins) - 1

                    # Ensure indices are within bounds
                    lon_idx = min(lon_idx, len(longitude_bins) - 2)
                    lat_idx = min(lat_idx, len(latitude_bins) - 2)

                    # Aggregate ON2 values
                    on2_sum1[lat_idx, lon_idx] += on2[scan_index1, j, k]
                    npoints1[lat_idx, lon_idx] += 1
                    on2_sum2[lat_idx, lon_idx] += on2[scan_index2, j, k]
                    npoints2[lat_idx, lon_idx] += 1

                # Extract time data for Scan 1
                time_entry1 = time_data_on2[scan_index1, j, k]
                if b'*' not in time_entry1:
                    valid_chars1 = [entry.decode('utf-8') for entry in time_entry1[:20] if entry not in [b'--', b'*', b' ']]
                    time_str1 = ''.join(valid_chars1)
                    try:
                        time_obj1 = datetime.strptime(time_str1, "%Y-%m-%dT%H:%M:%SZ")
                        valid_times_scan1.append(time_obj1)
                    except ValueError:
                        continue  

                # Extract time data for Scan 2
                time_entry2 = time_data_on2[scan_index2, j, k]
                if b'*' not in time_entry2:
                    valid_chars2 = [entry.decode('utf-8') for entry in time_entry2[:20] if entry not in [b'--', b'*', b' ']]
                    time_str2 = ''.join(valid_chars2)
                    try:
                        time_obj2 = datetime.strptime(time_str2, "%Y-%m-%dT%H:%M:%SZ")
                        valid_times_scan2.append(time_obj2)
                    except ValueError:
                        continue  

        # Sort the valid times
        valid_times_scan1.sort()
        valid_times_scan2.sort()

        # Get start and end times for Scan 1 and Scan 2
        start_time_scan1 = valid_times_scan1[0] if valid_times_scan1 else None
        end_time_scan2 = valid_times_scan2[-1] if valid_times_scan2 else None

        start_times_on2.append(start_time_scan1)
        end_times_on2.append(end_time_scan2)

        # Calculate the average ON2 score for each bucket
        average_on2_1 = np.divide(on2_sum1, npoints1, where=npoints1 > 0)
        average_on2_2 = np.divide(on2_sum2, npoints2, where=npoints2 > 0)

        # Set values < 0.01 to NaN
        average_on2_1[average_on2_1 < 0.01] = np.nan
        average_on2_2[average_on2_2 < 0.01] = np.nan
        longitude_mesh, latitude_mesh = create_meshgrid(longitude_bins, latitude_bins)

    gold_on2_outputs = {
    "average_on2_1": average_on2_1,
    "average_on2_2": average_on2_2,
    "latitude_mesh": latitude_mesh,
    "longitude_mesh": longitude_mesh,
    "start_times_on2": start_times_on2,
    "end_times_on2": end_times_on2, 
    "pixel_time_on2": time_data_on2
    }
    return gold_on2_outputs

def gold_tdisk_processing(gold_tdisk_vars):

    n_scans_tdisk = gold_tdisk_vars["n_scans_tdisk"]
    longitude_tdisk = gold_tdisk_vars["longitude_tdisk"]
    latitude_tdisk = gold_tdisk_vars["latitude_tdisk"]
    temperature = gold_tdisk_vars["temperature"]
    time_data_tdisk = gold_tdisk_vars["time_data_tdisk"]

    start_times_tdisk = []
    end_times_tdisk = []

    for i in range(0, n_scans_tdisk, 2):  

        scan_index1 = i
        scan_index2 = i + 1 if i + 1 < n_scans_tdisk else i

        # Define longitude and latitude ranges
        longitude_range = (-120, 60)
        latitude_range = (-60, 65)

        latitude_bins, longitude_bins = generate_custom_bins(latitude_range, longitude_range)

        n_latitude_buckets = len(latitude_bins) - 1
        n_longitude_buckets = len(longitude_bins) - 1
        temp_sum1 = np.zeros((n_latitude_buckets, n_longitude_buckets))
        temp_sum2 = np.zeros_like(temp_sum1)
        temp_points1 = np.zeros_like(temp_sum1)
        temp_points2 = np.zeros_like(temp_sum1)

        # Extract time data for each scan
        valid_times_scan1 = []
        valid_times_scan2 = []

        for j in range(longitude_tdisk.shape[0]):  
            for k in range(longitude_tdisk.shape[1]):  
                lon = longitude_tdisk[j, k]  # Assign separately
                lat = latitude_tdisk[j, k]  

                if longitude_range[0] <= lon <= longitude_range[1] and latitude_range[0] <= lat <= latitude_range[1]:
                    lon_idx = np.digitize(lon, longitude_bins) - 1
                    lat_idx = np.digitize(lat, latitude_bins) - 1

                    # Ensure indices are within bounds
                    lon_idx = min(lon_idx, len(longitude_bins) - 2)
                    lat_idx = min(lat_idx, len(latitude_bins) - 2)

                    # Aggregate temperature values
                    temp_sum1[lat_idx, lon_idx] += temperature[scan_index1, j, k]
                    temp_points1[lat_idx, lon_idx] += 1
                    temp_sum2[lat_idx, lon_idx] += temperature[scan_index2, j, k]
                    temp_points2[lat_idx, lon_idx] += 1

                # Extract time data for Scan 1
                time_entry1 = time_data_tdisk[scan_index1, j, k]
                if b'*' not in time_entry1:
                    valid_chars1 = [entry.decode('utf-8') for entry in time_entry1[:20] if entry not in [b'--', b'*', b' ']]
                    time_str1 = ''.join(valid_chars1)
                    try:
                        time_obj1 = datetime.strptime(time_str1, "%Y-%m-%dT%H:%M:%SZ")
                        valid_times_scan1.append(time_obj1)
                    except ValueError:
                        continue  

                # Extract time data for Scan 2
                time_entry2 = time_data_tdisk[scan_index2, j, k]
                if b'*' not in time_entry2:
                    valid_chars2 = [entry.decode('utf-8') for entry in time_entry2[:20] if entry not in [b'--', b'*', b' ']]
                    time_str2 = ''.join(valid_chars2)
                    try:
                        time_obj2 = datetime.strptime(time_str2, "%Y-%m-%dT%H:%M:%SZ")
                        valid_times_scan2.append(time_obj2)
                    except ValueError:
                        continue

        # Sort the valid times
        valid_times_scan1.sort()
        valid_times_scan2.sort()

        # Get start and end times for Scan 1 and Scan 2
        start_time_scan1 = valid_times_scan1[0] if valid_times_scan1 else None
        end_time_scan2 = valid_times_scan2[-1] if valid_times_scan2 else None

        start_times_tdisk.append(start_time_scan1)
        end_times_tdisk.append(end_time_scan2)

        # Calculate the average temperature score for each bucket
        average_temp1 = np.divide(temp_sum1, temp_points1, where=temp_points1 > 0)
        average_temp2 = np.divide(temp_sum2, temp_points2, where=temp_points2 > 0)

        # Set values < 0.01 to NaN
        average_temp1[average_temp1 < 0.01] = np.nan
        average_temp2[average_temp2 < 0.01] = np.nan

        # Create a meshgrid for plotting
        longitude_mesh, latitude_mesh = create_meshgrid(longitude_bins, latitude_bins)

    gold_tdisk_outputs = {
    "average_temp1": average_temp1,
    "average_temp2": average_temp2,
    "latitude_mesh": latitude_mesh,
    "longitude_mesh": longitude_mesh,
    "start_times_tdisk": start_times_tdisk,
    "end_times_tdisk": end_times_tdisk
    }
    return gold_tdisk_outputs

def plot_gold_on2(gold_on2_vars, start_times_on2, end_times_on2):

    latitude_on2 = gold_on2_vars["latitude_on2"]
    longitude_on2 = gold_on2_vars["longitude_on2"]
    on2 = gold_on2_vars["on2"]
    n_scans = gold_on2_vars["n_scans"]
    pairs = int(n_scans / 2)

    if start_times_on2 and isinstance(start_times_on2[0], datetime):
        date_str = start_times_on2[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')

    longitude_range = (-180, 180)
    latitude_range = (-90, 90)
    latitude_bins, longitude_bins = generate_custom_bins(latitude_range, longitude_range)

    fig = None
    fig_index = 0

    for i in range(pairs):
        subplot_index = i % 6
        row = subplot_index // 3
        col = subplot_index % 3

        # Start new figure every 6 plots
        if subplot_index == 0:
            if fig is not None:
                plt.savefig(f"on2_gold_plot_{date_str}_{fig_index}.png")
                plt.close()
                fig_index += 1
            fig = plt.figure(figsize=(12, 6))

        # Custom positioning in a 2x3 layout
        left_margin = 0.07 + col * 0.27   # Horizontal spacing
        bottom_margin = 0.55 - row * 0.45 # Vertical spacing 

        ax = fig.add_axes([left_margin, bottom_margin, 0.26, 0.36])

        scan_index1 = 2 * i
        scan_index2 = 2 * i + 1

        on2_sum1 = np.full((len(latitude_bins) - 1, len(longitude_bins) - 1), np.nan)
        on2_sum2 = np.full((len(latitude_bins) - 1, len(longitude_bins) - 1), np.nan)
        npoints1 = np.zeros((len(latitude_bins) - 1, len(longitude_bins) - 1))
        npoints2 = np.zeros((len(latitude_bins) - 1, len(longitude_bins) - 1))

        for j in range(longitude_on2.shape[0]):
            for k in range(longitude_on2.shape[1]):
                lon, lat = longitude_on2[j, k], latitude_on2[j, k]

                if longitude_range[0] <= lon <= longitude_range[1] and latitude_range[0] <= lat <= latitude_range[1]:
                    lon_idx = np.clip(np.digitize(lon, longitude_bins) - 1, 0, len(longitude_bins) - 2)
                    lat_idx = np.clip(np.digitize(lat, latitude_bins) - 1, 0, len(latitude_bins) - 2)

                    if not np.isnan(on2[scan_index1, j, k]):
                        on2_sum1[lat_idx, lon_idx] = np.nansum([on2_sum1[lat_idx, lon_idx], on2[scan_index1, j, k]])
                        npoints1[lat_idx, lon_idx] += 1

                    if not np.isnan(on2[scan_index2, j, k]):
                        on2_sum2[lat_idx, lon_idx] = np.nansum([on2_sum2[lat_idx, lon_idx], on2[scan_index2, j, k]])
                        npoints2[lat_idx, lon_idx] += 1

        avg1 = compute_average(on2_sum1, npoints1)
        avg2 = compute_average(on2_sum2, npoints2)
        avg_combined = combine_averages(avg1, avg2, npoints1, npoints2)

        lon_mesh, lat_mesh = create_meshgrid(longitude_bins, latitude_bins)

        pcm = ax.pcolormesh(lon_mesh, lat_mesh, avg_combined,
                            cmap='coolwarm', shading='auto', alpha=0.95, vmin=0.1, vmax=1.8)
        if i % 3 == 0:
            ax.set_ylabel("Latitude", fontsize=8)
        else:
            ax.set_yticks([])

        # Determine if current plot is in top row
        if row == 1:
            ax.set_xlabel("Longitude", fontsize=8)
        else:
            # If there's no corresponding subplot below (i + 3 exceeds total pairs),
            # then show xticks on top-row subplot
            if i + 3 >= pairs:
                ax.set_xlabel("Longitude", fontsize=8)
            else:
                ax.set_xticks([])
        ax.set_xlim(-120, 60)
        ax.set_ylim(-60, 65)
        ax.set_title(f"{start_times_on2[i]} to {end_times_on2[i]}", fontsize=8)

        cbar_ax = fig.add_axes([0.9, 0.10, (10 * 0.02)/12, 0.81])
        fig.colorbar(pcm, cax=cbar_ax, label="ON2 Intensity")

    # Save any remaining figure
    if fig is not None:
        plt.savefig(f"on2_gold_plot_{date_str}_{fig_index}.png")
        plt.close()

def plot_gold_tdisk(gold_tdisk_vars, start_times_tdisk, end_times_tdisk):

    latitude_tdisk = gold_tdisk_vars["latitude_tdisk"]
    longitude_tdisk = gold_tdisk_vars["longitude_tdisk"]
    temperature = gold_tdisk_vars["temperature"]
    n_scans = gold_tdisk_vars["n_scans_tdisk"]
    pairs = int(n_scans / 2)

    if start_times_tdisk and isinstance(start_times_tdisk[0], datetime):
        date_str = start_times_tdisk[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')

    longitude_range = (-180, 180)
    latitude_range = (-90, 90)
    latitude_bins, longitude_bins = generate_custom_bins(latitude_range, longitude_range)

    fig = None
    fig_index = 0

    for i in range(pairs):
        subplot_index = i % 6
        row = subplot_index // 3
        col = subplot_index % 3

        # Start new figure every 6 plots
        if subplot_index == 0:
            if fig is not None:
                plt.savefig(f"tdisk_gold_plot_{date_str}_{fig_index}.png")
                plt.close()
                fig_index += 1
            fig = plt.figure(figsize=(12, 6))

        # Custom positioning in a 2x3 layout
        left_margin = 0.07 + col * 0.27   # Horizontal spacing
        bottom_margin = 0.55 - row * 0.45 # Vertical spacing (two rows)

        ax = fig.add_axes([left_margin, bottom_margin, 0.26, 0.36])

        scan_index1 = 2 * i
        scan_index2 = 2 * i + 1

        temp_sum1 = np.full((len(latitude_bins) - 1, len(longitude_bins) - 1), np.nan)
        temp_sum2 = np.full((len(latitude_bins) - 1, len(longitude_bins) - 1), np.nan)
        npoints1 = np.zeros((len(latitude_bins) - 1, len(longitude_bins) - 1))
        npoints2 = np.zeros((len(latitude_bins) - 1, len(longitude_bins) - 1))

        for j in range(longitude_tdisk.shape[0]):
            for k in range(longitude_tdisk.shape[1]):
                lon, lat = longitude_tdisk[j, k], latitude_tdisk[j, k]

                if longitude_range[0] <= lon <= longitude_range[1] and latitude_range[0] <= lat <= latitude_range[1]:
                    lon_idx = np.clip(np.digitize(lon, longitude_bins) - 1, 0, len(longitude_bins) - 2)
                    lat_idx = np.clip(np.digitize(lat, latitude_bins) - 1, 0, len(latitude_bins) - 2)

                    if not np.isnan(temperature[scan_index1, j, k]):
                        temp_sum1[lat_idx, lon_idx] = np.nansum([temp_sum1[lat_idx, lon_idx], temperature[scan_index1, j, k]])
                        npoints1[lat_idx, lon_idx] += 1

                    if not np.isnan(temperature[scan_index2, j, k]):
                        temp_sum2[lat_idx, lon_idx] = np.nansum([temp_sum2[lat_idx, lon_idx], temperature[scan_index2, j, k]])
                        npoints2[lat_idx, lon_idx] += 1

        avg1 = compute_average(temp_sum1, npoints1)
        avg2 = compute_average(temp_sum2, npoints2)
        avg_combined = combine_averages(avg1, avg2, npoints1, npoints2)

        lon_mesh, lat_mesh = create_meshgrid(longitude_bins, latitude_bins)

        pcm = ax.pcolormesh(lon_mesh, lat_mesh, avg_combined,
                            cmap='coolwarm', shading='auto', alpha=0.95, vmin=600, vmax=1600)
        if i % 3 == 0:
            ax.set_ylabel("Latitude", fontsize=8)
        else:
            ax.set_yticks([])

        if row == 1:
            ax.set_xlabel("Longitude", fontsize=8)
        else:
            # If there's no corresponding subplot below (i + 3 exceeds total pairs),
            # then show xticks on top-row subplot
            if i + 3 >= pairs:
                ax.set_xlabel("Longitude", fontsize=8)
            else:
                ax.set_xticks([])

        ax.set_xlim(-120, 60)
        ax.set_ylim(-60, 65)
        ax.set_title(f"{start_times_tdisk[i]} to {end_times_tdisk[i]}", fontsize=8)

        cbar_ax = fig.add_axes([0.9, 0.10, (10 * 0.02)/12, 0.81])
        fig.colorbar(pcm, cax=cbar_ax, label="Temperature (K)")

    # Save any remaining figure
    if fig is not None:
        plt.savefig(f"tdisk_gold_plot_{date_str}_{fig_index}.png")
        plt.close()

def plot_gold_combined(gold_on2_vars, start_times_on2, end_times_on2,
                            gold_tdisk_vars, start_times_tdisk, end_times_tdisk):

    '''
    Used to plot all GOLD data for a given date. Mostly legacy, but can be helpful to identify days from GOLD worth comparing in GITM
    '''
    
    latitude_on2 = gold_on2_vars["latitude_on2"]
    longitude_on2 = gold_on2_vars["longitude_on2"]
    on2 = gold_on2_vars["on2"]
    n_scans_on2 = gold_on2_vars["n_scans"]
    pairs_on2 = int(n_scans_on2 / 2)

    if start_times_on2 and isinstance(start_times_on2[0], datetime):
        date_str = start_times_on2[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')
    
    latitude_tdisk = gold_tdisk_vars["latitude_tdisk"]
    longitude_tdisk = gold_tdisk_vars["longitude_tdisk"]
    temperature = gold_tdisk_vars["temperature"]
    n_scans_tdisk = gold_tdisk_vars["n_scans_tdisk"]
    pairs_tdisk = int(n_scans_tdisk / 2)

    print(f"ON2 scans: {pairs_on2}, TDISK scans: {pairs_tdisk}")

    if start_times_tdisk and isinstance(start_times_tdisk[0], datetime):
        date_str = start_times_tdisk[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')

    longitude_range = (-180, 180)
    latitude_range = (-90, 90)
    latitude_bins, longitude_bins = generate_custom_bins(latitude_range, longitude_range)

    fig = None
    fig_index = 0
    plot_height = 0.13
    plot_width = 0.22
    v_spacing = 0.03
    on2_left = 0.255
    tdisk_left = 0.525

    first_bar_on2 = None
    first_bar_tdisk = None
    last_on2_ax = None
    last_tdisk_ax = None

    max_pairs = max(pairs_on2, pairs_tdisk)
    for i in range(max_pairs):
        subplot_index = i % 6
        row = subplot_index // 3
        col = subplot_index % 3

        # Start new figure every 6 plots
        if subplot_index == 0:
            if fig is not None:
                # Add colorbars before closing the previous figure
                cbar_ax_on2 = fig.add_axes([0.14, 0.25, 0.02, 0.5])
                cbar_ax_tdisk = fig.add_axes([0.83, 0.25, 0.02, 0.5])

                cbar_on2 = fig.colorbar(first_bar_on2, cax=cbar_ax_on2, orientation="vertical")
                cbar_on2.set_label("ON2 Intensity", labelpad=16, fontsize=13)
                cbar_on2.ax.yaxis.set_label_position('left')
                cbar_on2.ax.yaxis.tick_left()

                cbar_tdisk = fig.colorbar(first_bar_tdisk, cax=cbar_ax_tdisk, orientation="vertical")
                cbar_tdisk.set_label("Temperature (K)", labelpad=22, rotation=270, fontsize=13)
                cbar_tdisk.ax.yaxis.set_label_position('right')

                plt.savefig(f"combined_gold_plot_{date_str}_{fig_index}.png")
                plt.close()
                fig_index += 1

            fig = plt.figure(figsize=(10, 14))
            first_bar_on2 = None
            first_bar_tdisk = None

        bottom = 1.0 - (subplot_index + 1) * (plot_height + v_spacing)
        ax_on2 = fig.add_axes([on2_left, bottom, plot_width, plot_height])
        if i < pairs_on2:
            #ax_on2 = fig.add_axes([on2_left, bottom, plot_width, plot_height])
            scan_index1 = 2 * i
            scan_index2 = 2 * i + 1

            on2_sum1 = np.full((len(latitude_bins) - 1, len(longitude_bins) - 1), np.nan)
            on2_sum2 = np.full((len(latitude_bins) - 1, len(longitude_bins) - 1), np.nan)
            npoints1 = np.zeros((len(latitude_bins) - 1, len(longitude_bins) - 1))
            npoints2 = np.zeros((len(latitude_bins) - 1, len(longitude_bins) - 1))

            for j in range(longitude_on2.shape[0]):
                for k in range(longitude_on2.shape[1]):
                    lon, lat = longitude_on2[j, k], latitude_on2[j, k]

                    if longitude_range[0] <= lon <= longitude_range[1] and latitude_range[0] <= lat <= latitude_range[1]:
                        lon_idx = np.clip(np.digitize(lon, longitude_bins) - 1, 0, len(longitude_bins) - 2)
                        lat_idx = np.clip(np.digitize(lat, latitude_bins) - 1, 0, len(latitude_bins) - 2)

                        if not np.isnan(on2[scan_index1, j, k]):
                            on2_sum1[lat_idx, lon_idx] = np.nansum([on2_sum1[lat_idx, lon_idx], on2[scan_index1, j, k]])
                            npoints1[lat_idx, lon_idx] += 1

                        if not np.isnan(on2[scan_index2, j, k]):
                            on2_sum2[lat_idx, lon_idx] = np.nansum([on2_sum2[lat_idx, lon_idx], on2[scan_index2, j, k]])
                            npoints2[lat_idx, lon_idx] += 1

            avg1_on2 = compute_average(on2_sum1, npoints1)
            avg2_on2 = compute_average(on2_sum2, npoints2)
            avg_combined_on2 = combine_averages(avg1_on2, avg2_on2, npoints1, npoints2)

            lon_mesh, lat_mesh = create_meshgrid(longitude_bins, latitude_bins)
            pcm_on2 = ax_on2.pcolormesh(lon_mesh, lat_mesh, avg_combined_on2,
                                cmap='coolwarm', shading='auto', alpha=0.95, vmin=0.1, vmax=1.8)
            ax_on2.set_title(f"{start_times_on2[i]} to {end_times_on2[i]}", fontsize=8)
            ax_on2.set_ylabel("Latitude", fontsize=8)
            ax_on2.set_xlim(-120, 60)
            ax_on2.set_ylim(-60, 65)
            if i - (6*fig_index) == 5:
                ax_on2.set_xlabel("Longitude", fontsize=8)
            else:
                ax_on2.set_xticks([])
            if first_bar_on2 is None:
                first_bar_on2 = pcm_on2
            last_on2_ax = ax_on2
        else:
            #ax_on2 = fig.add_axes([on2_left, bottom, plot_width, plot_height])
            ax_on2.text(0.5, 0.5, "No ON2 Data", ha='center', va='center')
            ax_on2.set_xticks([])
            ax_on2.set_yticks([])

        if i < pairs_tdisk:
            ax_tdisk = fig.add_axes([tdisk_left, bottom, plot_width, plot_height])

            scan_index1 = 2 * i
            scan_index2 = 2 * i + 1

            temp_sum1 = np.full((len(latitude_bins) - 1, len(longitude_bins) - 1), np.nan)
            temp_sum2 = np.full((len(latitude_bins) - 1, len(longitude_bins) - 1), np.nan)
            npoints1 = np.zeros((len(latitude_bins) - 1, len(longitude_bins) - 1))
            npoints2 = np.zeros((len(latitude_bins) - 1, len(longitude_bins) - 1))

            for j in range(longitude_tdisk.shape[0]):
                for k in range(longitude_tdisk.shape[1]):
                    lon, lat = longitude_tdisk[j, k], latitude_tdisk[j, k]

                    if longitude_range[0] <= lon <= longitude_range[1] and latitude_range[0] <= lat <= latitude_range[1]:
                        lon_idx = np.clip(np.digitize(lon, longitude_bins) - 1, 0, len(longitude_bins) - 2)
                        lat_idx = np.clip(np.digitize(lat, latitude_bins) - 1, 0, len(latitude_bins) - 2)

                        if not np.isnan(temperature[scan_index1, j, k]):
                            temp_sum1[lat_idx, lon_idx] = np.nansum([temp_sum1[lat_idx, lon_idx], temperature[scan_index1, j, k]])
                            npoints1[lat_idx, lon_idx] += 1

                        if not np.isnan(temperature[scan_index2, j, k]):
                            temp_sum2[lat_idx, lon_idx] = np.nansum([temp_sum2[lat_idx, lon_idx], temperature[scan_index2, j, k]])
                            npoints2[lat_idx, lon_idx] += 1

            avg1 = compute_average(temp_sum1, npoints1)
            avg2 = compute_average(temp_sum2, npoints2)
            avg_combined = combine_averages(avg1, avg2, npoints1, npoints2)

            lon_mesh, lat_mesh = create_meshgrid(longitude_bins, latitude_bins)

            pcm_tdisk = ax_tdisk.pcolormesh(lon_mesh, lat_mesh, avg_combined,
                                cmap='coolwarm', shading='auto', alpha=0.95, vmin=600, vmax=1600)
            ax_tdisk.set_xlim(-120, 60)
            ax_tdisk.set_ylim(-60, 65)
            ax_tdisk.set_title(f"{start_times_tdisk[i]} to {end_times_tdisk[i]}", fontsize=8)
            ax_tdisk.set_ylabel("Latitude", fontsize=8, rotation=270)
            ax_tdisk.yaxis.tick_right() 
            ax_tdisk.yaxis.set_label_position("right") 
            if i - (6*fig_index) == 5:
                ax_tdisk.set_xlabel("Longitude", fontsize=8)
            else:
                ax_tdisk.set_xticks([])
            if first_bar_tdisk is None:
                first_bar_tdisk = pcm_tdisk
            last_tdisk_ax = ax_tdisk
        else:
            ax_tdisk = fig.add_axes([tdisk_left, bottom, plot_width, plot_height])
            ax_tdisk.text(0.5, 0.5, "N/A", ha='center', va='center')
            ax_tdisk.set_xticks([])
            ax_tdisk.set_yticks([])

    if fig is not None:
        # Custom axes for colorbars
        cbar_ax_on2 = fig.add_axes([0.14, 0.25, 0.02, 0.5])
        cbar_ax_tdisk = fig.add_axes([0.83, 0.25, 0.02, 0.5])

        cbar_on2 = fig.colorbar(first_bar_on2, cax=cbar_ax_on2, orientation="vertical")
        cbar_on2.set_label("ON2 Intensity", labelpad=16, fontsize = 13)
        cbar_on2.ax.yaxis.set_label_position('left')
        cbar_on2.ax.yaxis.tick_left()

        cbar_tdisk = fig.colorbar(first_bar_tdisk, cax=cbar_ax_tdisk, orientation="vertical")
        cbar_tdisk.set_label("Temperature (K)", labelpad=22, rotation=270, fontsize = 13)
        cbar_tdisk.ax.yaxis.set_label_position('right')

        # Only show x-axis labels on the last used axes
        if last_on2_ax:
            last_on2_ax.set_xlabel("Longitude", fontsize=8)
            last_on2_ax.set_xticks([-100, -50, 0, 50])

        if last_tdisk_ax:
            last_tdisk_ax.set_xlabel("Longitude", fontsize=8)
            last_tdisk_ax.set_xticks([-100, -50, 0, 50])

        plt.savefig(f"combined_gold_plot_{date_str}_{fig_index}.png")
        plt.close()
        first_bar_on2 = None
        first_bar_tdisk = None