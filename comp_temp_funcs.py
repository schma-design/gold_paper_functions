import numpy as np
from scipy.interpolate import interp1d
import os
from nc_funcs import read_gitm_headers_nc, read_gitm_one_file_nc
from gitm_routines import read_gitm_headers, remap_variable_names, read_gitm_one_file
from datetime import datetime
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import distance_transform_edt
from comp_on2_funcs import gitm_on2_processing
from comp_gen_misc import generate_custom_bins, compute_average, combine_averages, create_meshgrid, find_closest_filesv2
from comp_gold_funcs import extract_gold_tdisk, extract_valid_tdisk_points, get_scan_pair_times

def expon(alts, altl):
    fit_radiance = []
    fit_radiance = 0.6 + 10 * np.exp(-((alts - (altl))**2)/70**2)
    return fit_radiance

def weighted_temp_average(temp_data, alts, altl):
    """
    Computes the weighted average of temperature over altitude using the expon weighting function.

    Parameters:
    - temp_data: 3D numpy array of shape (lat, lon, altitude_index)
    - alts: 1D numpy array of altitudes (in km) corresponding to the third axis of temp_data

    Returns:
    - 2D array (lat x lon) of weighted average temperatures
    """
    # Mask for altitudes under range
    mask = alts < 400
    valid_alts = alts[mask]
    valid_weights = expon(valid_alts, altl)
    
    # Initialize accumulators
    weighted_sum = np.zeros_like(temp_data[:, :, 0])
    total_weight = 0.0

    for idx, weight in zip(np.where(mask)[0], valid_weights):
        weighted_sum += temp_data[:, :, idx] * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight != 0 else np.zeros_like(weighted_sum)

def weighted_temp_average_dynamic(temp_data, alts, limit_altitudes):
    """
    Computes the weighted average of temperature over altitude using a 
    pixel-specific 'peak altitude' from limit_altitudes.

    Parameters:
    - temp_data: 3D numpy array of shape (lat, lon, altitude_index)
    - alts: 1D numpy array of altitudes (in km)
    - limit_altitudes: 2D numpy array (lat x lon) of peak altitudes

    Returns:
    - 2D array (lat x lon) of weighted average temperatures
    """
    nLat, nLon, nAlt = temp_data.shape
    weighted_avg_temp = np.zeros((nLat, nLon))
    
    # Precompute valid altitude indices
    mask = alts < 400
    valid_alts = alts[mask]
    valid_indices = np.where(mask)[0]

    for i in range(nLat):
        for j in range(nLon):
            altl = limit_altitudes[i, j]
            if np.isnan(altl):
                weighted_avg_temp[i, j] = np.nan
                continue

            weights = expon(valid_alts, altl)
            temps = temp_data[i, j, valid_indices]

            weighted_sum = np.sum(temps * weights)
            total_weight = np.sum(weights)

            if total_weight != 0:
                weighted_avg_temp[i, j] = weighted_sum / total_weight
            else:
                weighted_avg_temp[i, j] = np.nan

    return weighted_avg_temp

def weighted_temp_cf(temp_data, alts, value, cf_data, sza):

    cf_sza = cf_data.variables['sol_zen'][:]
    cf_alt = cf_data.variables['altitude'][:]
    cf_raw = cf_data.variables['CF'][:]

    sza_idx = np.abs(cf_sza - sza).argmin()
    cf_sel = cf_raw[sza_idx, :]
    valid = np.isfinite(cf_sel)

    # Shift CF altitude grid
    cf_alt_shifted = cf_alt + value

    cf_interp = interp1d(cf_alt_shifted[valid], cf_sel[valid], bounds_error=False, fill_value=0.0)
    cf_on_gitm = cf_interp(alts)

    mask = (alts >= 100) & (alts <= 1000)
    z = alts[mask]
    cf_z = cf_on_gitm[mask]

    norm = np.trapezoid(cf_z, z)
    if norm == 0:
        shape = temp_data[:, :, 0].shape
        return np.zeros(shape), np.zeros_like(cf_z)

    cf_norm = cf_z / norm
    weighted_temp = np.trapezoid(temp_data[:, :, mask] * cf_norm[None, None, :], z,axis=2)

    return weighted_temp, cf_norm

def gitm_temp_data_processing(file_paths_tdisk, weighted_option, value, gitm_on2_outputs, cache, cf_data):
    
    '''
    Takes in GITM files and produces scatter data. Does so for one file, not time dependent for each GOLD data point.
    Each method processes the GITM data differently, read the "help" section in the get_args function from the routines 
    script to learn more.
    '''
    if cache is None:
        cache = {}
    
    scatter_data_tdisk = []
    mesh_data_tdisk = []
    scan_alts = []

    if weighted_option == 2 or weighted_option == 3:
        avg_n2_lim = gitm_on2_outputs["avg_n2_lim"]
        n2_lim_map = gitm_on2_outputs["n2_lim_map"]
        thresh = gitm_on2_outputs["thresh"]
    else:
        thresh = 0.7

    # Iterate over the list of file paths
    for i, file_path in enumerate(file_paths_tdisk):
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            continue

        temp_key = f"{file_path}_temp"
        if temp_key in cache:
            headers, data, temp_data = cache[temp_key]
        else:
            if file_path.endswith(".nc"):
                headers = read_gitm_headers_nc(file_path)
                vars = headers["vars"]
                iTemp_ = vars.index("Tn (K)")
                iVars_ = [0, 1, 2, iTemp_]
                data = read_gitm_one_file_nc(file_path, iVars_)
            else:
                # Read GITM headers
                headers = read_gitm_headers(files=[file_path])
                vars = remap_variable_names(headers["vars"])
                iTemp_ = vars.index("Tn (K)")
                iVars_ = [0, 1, 2, iTemp_]
                data = read_gitm_one_file(headers["filename"][0], iVars_)

            temp_data = data[iTemp_]
            cache[temp_key] = (headers, data, temp_data)

        # Extract longitude, latitude, altitude information
        times = headers["time"]
        Alts = data[2][0, 0, :] / 1000.0  # Convert to km
        Lons = data[0][:, 0, 0] * (180.0 / np.pi)  # Convert to degrees
        Lats = data[1][0, :, 0] * (180.0 / np.pi)  # Convert to degrees

        ########################################## RAW COMPARISON ############################################################
        if weighted_option == 0 or weighted_option == 5:
            alt_guess = value
            closest_alt_index = np.argmin(np.abs(Alts - alt_guess))
            temp_at_altitude = temp_data[:, :, closest_alt_index]
            print(f"Closest altitude to {alt_guess} km in GITM data is {Alts[closest_alt_index]:.2f} km")
            true_alt = Alts[closest_alt_index]
            
            # Create a meshgrid for longitude and latitude
            Lons_grid, Lats_grid = np.meshgrid(Lons, Lats)

            # Flatten the 2D temperature, longitude, and latitude arrays
            temp_flat = temp_at_altitude.flatten()
            
            #lons_flat = Lons_grid.flatten()
            #lats_flat = Lats_grid.flatten()

            longitudes, latitudes = np.meshgrid(Lons, Lats, indexing='ij')
            longitudes = longitudes.flatten()
            latitudes = latitudes.flatten()

            temp_cleaned = temp_at_altitude.copy()
            temp_flat = temp_cleaned.flatten()

            #Create the valid mask for longitude and latitude values
            valid_mask = (longitudes >= 0) & (longitudes <= 360) & (latitudes >= -90) & (latitudes <= 90)

            #Apply mask to filter out invalid data points
            longitudes_valid = longitudes[valid_mask]
            latitudes_valid = latitudes[valid_mask]
            temp_valid = temp_flat[valid_mask]

            #Convert longitudes to the -180 to 180 range after filtering invalid values
            longitudes_valid[longitudes_valid > 180] -= 360 
            longitudes_valid[longitudes_valid < -180] += 360 
            longitude_range = (-180, 180)
            latitude_range = (-90, 90)

            # Append the filtered and adjusted data to the list
            scatter_data_tdisk.append((longitudes_valid, latitudes_valid, temp_valid, times[0]))

            latitude_bins, longitude_bins = generate_custom_bins(latitude_range, longitude_range)

            # Initialize grid to store sum of temperature values and number of points in each bucket
            n_latitude_buckets = len(latitude_bins) - 1
            n_longitude_buckets = len(longitude_bins) - 1
            temp_sum = np.zeros((n_latitude_buckets, n_longitude_buckets))
            npoints = np.zeros((n_latitude_buckets, n_longitude_buckets))

            # Loop through the cleaned data and allocate points to their respective buckets
            for i in range(len(longitudes_valid)): 
                lon = longitudes_valid[i]
                lat = latitudes_valid[i]
                temp_value = temp_valid[i]

                # Find the corresponding bucket for this point
                lon_idx = np.digitize(lon, longitude_bins) - 1
                lat_idx = np.digitize(lat, latitude_bins) - 1
                if not (0 <= lon_idx < n_longitude_buckets and 0 <= lat_idx < n_latitude_buckets):
                    continue

                # Add the temperature value to the corresponding bucket and increment npoints
                temp_sum[lat_idx, lon_idx] += temp_value
                npoints[lat_idx, lon_idx] += 1

            # Calculate the average temperature for each bucket (avoid division by zero)
            average_temp = np.divide(temp_sum, npoints, where=npoints > 0)

            # Set values < 0.01 to NaN (failsafe to protect from invalid values/cells)
            average_temp[average_temp < 0.01] = np.nan

            # Store the mesh data
            mesh_data_tdisk.append((longitude_bins, latitude_bins, average_temp, times[0]))

        ######################################### FIXED PEAK #####################################################################
        elif weighted_option == 1:
            weighted_avg = weighted_temp_average(temp_data, Alts, value)

        ####################################### WEIGHTED AVERAGE (CHANGES OVER TIME) ###########################################################
        elif weighted_option == 2:
            altl = avg_n2_lim[i] + value
            scan_alts.append(altl)
            weighted_avg = weighted_temp_average(temp_data, Alts, altl)
            print(f"Processing data from {altl} km")

        ########################################### DYAMIC WEIGHTED AVERAGE ###################################################
        elif weighted_option == 3:
            if len(n2_lim_map) == 1:
                n2_lim_map[0] = n2_lim_map[0] / 1000.00
                current_limit_map = n2_lim_map[0] + value
            else:
                n2_lim_map[i] = n2_lim_map[i] / 1000.00
                current_limit_map = n2_lim_map[i] + value
            print(f"Processing data from {value} km offset")
            weighted_avg = weighted_temp_average_dynamic(temp_data, Alts, current_limit_map)

        ##########################################################################################
        elif weighted_option == 6: 
            sza = 53
            weighted_avg, weights = weighted_temp_cf(temp_data, Alts, value, cf_data, sza)

        elif weighted_option == 7:
            weighted_avg = weighted_temp_average(temp_data, Alts, (195 + value))
        ####################################################################################################

        if weighted_option != 0 and weighted_option != 4 and weighted_option != 5:

            # Flatten temperature to match longitude and latitude points
            temp_flat2 = weighted_avg.flatten()

            # Clean Data for Mesh Plot
            temp_cleaned2 = weighted_avg.copy()

            # Flatten arrays
            longitudes, latitudes = np.meshgrid(Lons, Lats, indexing='ij')
            longitudes = longitudes.flatten()
            latitudes = latitudes.flatten()
            temp_flat2 = temp_cleaned2.flatten()

            # Step 1: Create the valid mask for longitude and latitude values
            valid_mask = (longitudes >= 0) & (longitudes <= 360) & (latitudes >= -90) & (latitudes <= 90)

            # Step 2: Apply mask to filter out invalid data
            longitudes_valid = longitudes[valid_mask]
            latitudes_valid = latitudes[valid_mask]
            temp_valid2 = temp_flat2[valid_mask]

            # Step 3: Convert longitudes to the -180 to 180 range
            longitudes_valid[longitudes_valid > 180] -= 360
            longitudes_valid[longitudes_valid < -180] += 360

            # Define the longitude and latitude ranges
            longitude_range = (-180, 180)
            latitude_range = (-90, 90)

            # Append the filtered and adjusted data to the list
            scatter_data_tdisk.append((longitudes_valid, latitudes_valid, temp_valid2, times[0]))

            latitude_bins, longitude_bins = generate_custom_bins(latitude_range, longitude_range)

            # Initialize grid to store sum of temperature values and number of points in each bucket
            n_latitude_buckets = len(latitude_bins) - 1
            n_longitude_buckets = len(longitude_bins) - 1
            temp_sum2 = np.zeros((n_latitude_buckets, n_longitude_buckets))
            npoints = np.zeros((n_latitude_buckets, n_longitude_buckets))

            # Loop through the cleaned data and allocate points to their respective buckets
            for i in range(len(longitudes_valid)):
                lon = longitudes_valid[i]
                lat = latitudes_valid[i]
                temp_value2 = temp_valid2[i]

                # Find the corresponding bucket for this point
                lon_idx = np.digitize(lon, longitude_bins) - 1
                lat_idx = np.digitize(lat, latitude_bins) - 1
                if not (0 <= lon_idx < n_longitude_buckets and 0 <= lat_idx < n_latitude_buckets):
                    continue
                # Add the temperature value to the corresponding bucket and increment npoints
                temp_sum2[lat_idx, lon_idx] += temp_value2
                npoints[lat_idx, lon_idx] += 1

            # Calculate the average temperature for each bucket (avoid division by zero)
            average_temp2 = np.divide(temp_sum2, npoints, where=npoints > 0)

            # Set values < 0.01 to NaN 
            average_temp2[average_temp2 < 0.01] = np.nan

            # Store the mesh data
            mesh_data_tdisk.append((longitude_bins, latitude_bins, average_temp2, times[0]))

    if weighted_option != 0:
            true_alt = 0
    
    gitm_tdisk_outputs = {
    "mesh_data_tdisk": mesh_data_tdisk,
    "scatter_data_tdisk": scatter_data_tdisk,
    "scan_alts": scan_alts,
    "weighted_option": weighted_option,
    "value": value,
    "thresh": thresh,
    "true_alt": true_alt}
    return gitm_tdisk_outputs

def gitm_tdisk_processing_v2(points, time_to_file_map, weighted_option, value, thresh, cache, start_times, end_times, cf_data):
    """
    Processes GITM temperature data point by point for a matching GOLD point.
    Each method processes the GITM data differently, read the "help" section in the get_args function from the routines 
    script to learn more.
    Scatter data is produced by gitm_tdisk_processing_v2. Assumptions are made for some methods when producing the scatter data that
    can be changed if desired.
    """

    if weighted_option == 2 or weighted_option == 3:
        gitm_on2_outputs = {
            "thresh": thresh,
            "avg_n2_lim": [],
            "n2_lim_map": []}
        
    else:
        thresh = 0.7
        gitm_on2_outputs = 0.7
    
    gitm_file_cache = {}
    gitm_point_results = []
    mesh_data_tdisk = [] 
    central_time_files = []
    fixed_scatter_outputs = []
    scan_alts = []
    on2_cache = {}
    avg_n2_lim_list = []
    points_by_window = {i: [] for i in range(len(start_times))}
    scan_windows = list(zip(start_times, end_times))

    # Defined nodes produced by numerically integrating each SZA contribution function curve to produce an approximate "effective" alt
    sza_nodes = np.array([0.0, 37.0, 53.0, 66.0, 78.0, 90.0])
    z_eff_nodes = np.array([168.9, 174.8, 182.7, 193.7, 212.9, 259.5])
    z_eff_from_sza = interp1d(sza_nodes, z_eff_nodes, kind='linear', bounds_error=False, fill_value='extrapolate')

    for point in points:
        t = point.get("time")
        if not t:
            continue

        for i, (start, end) in enumerate(scan_windows):
            if start <= t < end:
                points_by_window[i].append(point)
                break

    for i, point_list in points_by_window.items():
        gitm_point_results = [] 
        if not point_list:
            continue
        start_time, end_time = scan_windows[i]
        file_times = []
        for point in point_list:
            t = point["time"]
            f = time_to_file_map.get(t)
            if f:
                file_times.append((t, f))
        scan_weighted_alts = []

        file_times = sorted(set(file_times), key=lambda x: x[0])
        if file_times:
            if weighted_option == 2:
                for _, gitm_file in file_times:
                    if gitm_file not in on2_cache:
                        result = gitm_on2_processing([gitm_file], thresh, 0, cache)
                        on2_cache[gitm_file] = np.average(result["avg_n2_lim"])
                tot_avg_n2_lim = np.mean([on2_cache[f] for _, f in file_times])
                gitm_on2_outputs["avg_n2_lim"].append(tot_avg_n2_lim)

            n = len(file_times)
            median_index = (n)//2
            central_file = file_times[median_index][1]
            central_time_files.append(central_file)

            points_by_file = defaultdict(list)
            for point in point_list:
                f = time_to_file_map.get(point["time"])
                if f:
                    points_by_file[f].append(point)

            for gitm_file, file_points in points_by_file.items():
                tdisk_key = f"{gitm_file}_tdisk"
                if tdisk_key in cache:
                    headers, data, temp_data, var_index_map, iTemp_ = cache[tdisk_key]
                else:
                    if gitm_file.endswith(".nc"):
                        headers = read_gitm_headers_nc(gitm_file)
                        vars = headers["vars"]
                        iTemp_ = vars.index("Tn (K)")
                        iVars_ = [0, 1, 2, iTemp_]
                        data = read_gitm_one_file_nc(gitm_file, iVars_)
                    else:
                        headers = read_gitm_headers(files=[gitm_file])
                        vars = remap_variable_names(headers["vars"])
                        iTemp_ = vars.index("Tn (K)")
                        iVars_ = [0, 1, 2, iTemp_]
                        data = read_gitm_one_file(headers["filename"][0], iVars_)

                    var_index_map = {i: j for j, i in enumerate(iVars_)}
                    temp_data = data[iTemp_]
                    cache[tdisk_key] = (headers, data, temp_data, var_index_map, iTemp_)

                # Grid setup
                iLon = var_index_map[0]
                iLat = var_index_map[1]
                #iAlt = var_index_map[2]
                Alts = data[2][0, 0, :] / 1000.0 
                #times = headers["time"]
                Lons = data[iLon][:, 0, 0] * (180.0 / np.pi)
                Lons[Lons > 180] -= 360  # Convert 0–360 to -180–180
                Lats = data[iLat][0, :, 0] * (180.0 / np.pi)
                nLons, nLats, nAlts = data[iLon].shape

                if weighted_option == 0:
                    alt_guess = value
                    closest_alt_index = np.argmin(np.abs(Alts - alt_guess))
                    true_alt = round(Alts[closest_alt_index], 2)

                    for point in file_points:
                        plat = point["lat"]
                        plon = point["lon"]
                        j = np.abs(Lats - plat).argmin()
                        k = np.abs(Lons - plon).argmin()
                        val = temp_data[k, j, closest_alt_index]
                        gitm_point_results.append({
                            "lat": Lats[j],
                            "lon": Lons[k],
                            "temp": val,
                            "gitm_time": start_time})

                elif weighted_option == 1: 
                    weighted_data = weighted_temp_average(temp_data, Alts, value)
                    for point in point_list:
                        if time_to_file_map.get(point["time"]) != gitm_file:
                            continue

                        plat = point["lat"]
                        plon = point["lon"]
                        j = np.abs(Lats - plat).argmin()
                        k = np.abs(Lons - plon).argmin()
                        try: 
                            val = weighted_data[k, j]
                        except IndexError:
                            val = np.nan
                        
                        gitm_point_results.append({
                            "lat": Lats[j],
                            "lon": Lons[k],
                            "temp": val,
                            "gitm_time": start_time})

                elif weighted_option == 2: 
                    altl = tot_avg_n2_lim + value
                    scan_alts.append(altl)
                    weighted_data = weighted_temp_average(temp_data, Alts, altl)
                    for point in point_list:
                        if time_to_file_map.get(point["time"]) != gitm_file:
                            continue

                        plat = point["lat"]
                        plon = point["lon"]
                        j = np.abs(Lats - plat).argmin()
                        k = np.abs(Lons - plon).argmin()
                        try: 
                            val = weighted_data[k,j]
                        except IndexError:
                            val = np.nan

                        gitm_point_results.append({
                            "lat": Lats[j],
                            "lon": Lons[k],
                            "temp": val,
                            "gitm_time": start_time})
                        
                elif weighted_option == 3:
                    # For each file, cache its ON2 limit map
                    n2_lim_maps = []
                    for _, gitm_file in file_times:
                        if gitm_file not in on2_cache:
                            result = gitm_on2_processing([gitm_file], thresh, 0, cache)
                            on2_cache[gitm_file] = result["n2_lim_map"][0] / 1000.0  # km
                        n2_lim_maps.append(on2_cache[gitm_file])

                    # Compute representative map (including a median for the scatter)
                    n2_lim_maps = np.array(n2_lim_maps)
                    if n2_lim_maps.ndim == 3:
                        base_limit_map = np.median(n2_lim_maps, axis=0)
                    else:
                        base_limit_map = n2_lim_maps[0]

                    # Add offset value and use for dynamic weighting
                    current_limit_map = base_limit_map + value
                    scan_alts.append(np.nanmean(base_limit_map) + value)

                    gitm_on2_outputs["n2_lim_map"].append(base_limit_map * 1000.0)
                    gitm_on2_outputs["avg_n2_lim"].append(np.nanmean(base_limit_map))

                    # Build weighted data
                    if gitm_file not in gitm_file_cache:
                        weighted_data = weighted_temp_average_dynamic(temp_data, Alts, current_limit_map)
                        gitm_file_cache[gitm_file] = weighted_data
                    else:
                        weighted_data = gitm_file_cache[gitm_file]

                    for point in file_points:
                        plat = point["lat"]
                        plon = point["lon"]
                        j = np.abs(Lats - plat).argmin()
                        k = np.abs(Lons - plon).argmin()
                        try:
                            val = weighted_data[k, j]
                        except IndexError:
                            val = np.nan
                        gitm_point_results.append({
                            "lat": Lats[j],
                            "lon": Lons[k],
                            "temp": val,
                            "gitm_time": start_time})
                        
                elif weighted_option == 5:
                    for point in file_points:
                        point_sza = point["sza"]
                        if np.ma.is_masked(point_sza):
                            continue
                        point_sza = float(point_sza)

                        # Linear interpolation
                        z_eff = float(z_eff_from_sza(point_sza))

                        # pick closest GITM altitude
                        closest_alt_index = np.argmin(np.abs(Alts - z_eff))
                        true_alt = Alts[closest_alt_index]

                        plat = point["lat"]
                        plon = point["lon"]
                        j = np.abs(Lats - plat).argmin()
                        k = np.abs(Lons - plon).argmin()
                        val = temp_data[k, j, closest_alt_index]

                        gitm_point_results.append({
                            "lat": Lats[j],
                            "lon": Lons[k],
                            "temp": val,
                            "z_eff": true_alt,
                            "sza": point_sza,
                            "gitm_time": start_time
                        })
                    all_alts = np.array([p["z_eff"] for p in gitm_point_results])
                    value = np.mean(all_alts)

                elif weighted_option == 6:
                    weighted_alts = []
                    cf_weight_cache = {}
                    def sza_key(sza, resolution=1.0):
                        return round(sza / resolution) * resolution

                    for point in file_points:
                        point_sza = point['sza']
                        if np.ma.is_masked(point_sza):
                            continue
                        point_sza = float(point_sza)

                        key = sza_key(point_sza, resolution=1.0)

                        if key not in cf_weight_cache:
                            weighted_data, weights = weighted_temp_cf(temp_data, Alts, value, cf_data, key)
                            z = Alts[(Alts >= 100) & (Alts <= 1000)]
                            z_eff = np.trapezoid(z * weights, z)
                            cf_weight_cache[key] = (weighted_data, z_eff)

                        weighted_data, z_eff = cf_weight_cache[key]
                        plat = point["lat"]
                        plon = point["lon"]
                        j = np.abs(Lats - plat).argmin()
                        k = np.abs(Lons - plon).argmin()

                        val = weighted_data[k, j]
                        weighted_alts.append(z_eff)
                        scan_weighted_alts.append(z_eff)

                        gitm_point_results.append({
                            "lat": Lats[j],
                            "lon": Lons[k],
                            "temp": val,
                            "gitm_time": start_time
                        }) 

                elif weighted_option == 7:
                    weighted_alt_cache = {}
                    scan_weighted_alts = []

                    def alt_key(z_eff, resolution=1.0):
                        return round(z_eff / resolution) * resolution
                    
                    for point in file_points: 
                        point_sza = point['sza']
                        if np.ma.is_masked(point_sza):
                            continue
                        point_sza = float(point_sza)
                        # Linear interpolation
                        z_eff = float(z_eff_from_sza(point_sza) + value)

                        # pick closest GITM altitude
                        closest_alt_index = np.argmin(np.abs(Alts - z_eff))
                        true_alt = Alts[closest_alt_index]
                        scan_weighted_alts.append(true_alt)

                        key = alt_key(z_eff, resolution=1.0)

                        if key not in weighted_alt_cache:
                            # Compute ONCE per altitude bin
                            weighted_data = weighted_temp_average(temp_data, Alts, key)
                            weighted_alt_cache[key] = weighted_data

                        weighted_data = weighted_alt_cache[key]
                        plat = point["lat"]
                        plon = point["lon"]
                        j = np.abs(Lats - plat).argmin()
                        k = np.abs(Lons - plon).argmin()
                        try: 
                            val = weighted_data[k,j]
                        except IndexError:
                            val = np.nan

                        gitm_point_results.append({
                            "lat": Lats[j],
                            "lon": Lons[k],
                            "temp": val,
                            "gitm_time": start_time})

            if (weighted_option == 6 or weighted_option == 7) and scan_weighted_alts:
                true_alt = np.nanmean(scan_weighted_alts)
                scan_alts.append(true_alt)
                print(f"Scan {i}: Avg CF-weighted altitude = {true_alt:.2f} km")

        # Generate mesh for this hour group
        if gitm_point_results:
            all_lats = np.array([p["lat"] for p in gitm_point_results])
            all_lons = np.array([p["lon"] for p in gitm_point_results])
            all_vals = np.array([p["temp"] for p in gitm_point_results])
            all_lons[all_lons > 180] -= 360

            latitude_range = (-90, 90)
            longitude_range = (-180, 180)
            latitude_bins, longitude_bins = generate_custom_bins(latitude_range, longitude_range)
            n_latitude_buckets = len(latitude_bins) - 1
            n_longitude_buckets = len(longitude_bins) - 1
            temp_sum = np.zeros((n_latitude_buckets, n_longitude_buckets))
            npoints = np.zeros((n_latitude_buckets, n_longitude_buckets))

            for lat, lon, val in zip(all_lats, all_lons, all_vals):
                lon_idx = np.digitize(lon, longitude_bins) - 1
                lat_idx = np.digitize(lat, latitude_bins) - 1

                if not (0 <= lon_idx < n_longitude_buckets and 0 <= lat_idx < n_latitude_buckets):
                    continue

                if np.isfinite(val):
                    temp_sum[lat_idx, lon_idx] += val
                    npoints[lat_idx, lon_idx] += 1

            with np.errstate(divide='ignore', invalid='ignore'):
                average_temp = np.divide(temp_sum, npoints)
                average_temp[npoints == 0] = np.nan

            invalid_mask = (average_temp < 100) | (average_temp > 3000) | np.isnan(average_temp)
            average_temp[invalid_mask] = np.nan
            mask = np.isnan(average_temp)
            if np.any(mask):
                nearest_idx = distance_transform_edt(mask, return_distances = False, return_indices=True)
                average_temp = average_temp[tuple(nearest_idx)]
            mesh_data_tdisk.append((longitude_bins, latitude_bins, average_temp, start_time))
    
    gitm_tdisk_results = gitm_temp_data_processing(central_time_files, weighted_option, value, gitm_on2_outputs, cache, cf_data)
    fixed_scatter_outputs = gitm_tdisk_results["scatter_data_tdisk"]
    if weighted_option != 0:
        true_alt = 0

    gitm_tdisk_outputs2 = {
        "mesh_data_tdisk": mesh_data_tdisk,
        "scatter_data_tdisk": fixed_scatter_outputs,
        "weighted_option": weighted_option,
        "value": value,
        "thresh": thresh,
        "true_alt": true_alt, 
        "scan_alts": scan_alts
    }
    return gitm_tdisk_outputs2

def temp_data_comparison(gitm_tdisk_outputs, gold_tdisk_vars, start_times_tdisk, end_times_tdisk, plot_option, output_dir):

    '''
    Main comparison function. Takes in processed GOLD and GITM data to produce a qualitative and quantitative temperature comparison.
    Generates mean difference, RMS, and cross correlation data, along with producing global contour plots if plot_option == 1. 
    If a variable has "left" in the name, it refers to GOLD data (i.e. left column in the global plots)
    If a variable has "right" in the name, it refers to the GITM data plotted on the GOLD mesh (i.e. central column in global plots)
    The "scatter" data is GITM data that does not go through the binning procedure (i.e. the right column in the global plots)
    '''

    longitude_tdisk = gold_tdisk_vars["longitude_tdisk"]
    latitude_tdisk = gold_tdisk_vars["latitude_tdisk"]
    n_scans_tdisk = gold_tdisk_vars["n_scans_tdisk"]
    temperature = gold_tdisk_vars["temperature"]
    pairs_tdisk = int(n_scans_tdisk / 2)
    mesh_data_tdisk = gitm_tdisk_outputs["mesh_data_tdisk"]
    scatter_data_tdisk = gitm_tdisk_outputs["scatter_data_tdisk"]
    print(f"Len of scatter: {len(scatter_data_tdisk)}")
    weighted_option = gitm_tdisk_outputs["weighted_option"]
    value = gitm_tdisk_outputs["value"]
    thresh = gitm_tdisk_outputs["thresh"]

    if start_times_tdisk and isinstance(start_times_tdisk[0], datetime):
        date_str = start_times_tdisk[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')  # fallback

    mean_diffs = []
    mean_percent_diffs = []
    rms_vals = []
    mean_percent_rms_vals = []
    corr_scores = []

    longitude_range = (-180, 180)
    latitude_range = (-90, 90)
    latitude_bins, longitude_bins = generate_custom_bins(latitude_range, longitude_range)

    counter = 0
    fig = plt.figure(figsize=(12, 7.5))

    for i in range(pairs_tdisk):
        if i % 3 == 0 and i > 0:
            fig = plt.figure(figsize=(12, 7.5))
            counter += 1

        scan_index1 = 2 * i
        scan_index2 = 2 * i + 1

        longitude_bins_left, latitude_bins_left = longitude_bins, latitude_bins
        longitude_bins_right, latitude_bins_right, average_gold_temp, plot_time2 = mesh_data_tdisk[i]

        temp_sum1 = np.full((len(latitude_bins_left) - 1, len(longitude_bins_left) - 1), np.nan)
        temp_sum2 = np.full_like(temp_sum1, np.nan)
        npoints1 = np.zeros_like(temp_sum1)
        npoints2 = np.zeros_like(temp_sum1)

        temp_sum_right1 = np.full((len(latitude_bins_right) - 1, len(longitude_bins_right) - 1), np.nan)
        temp_sum_right2 = np.full_like(temp_sum_right1, np.nan)
        npoints_right1 = np.zeros_like(temp_sum_right1)
        npoints_right2 = np.zeros_like(temp_sum_right1)

        for j in range(longitude_tdisk.shape[0]):
            for k in range(longitude_tdisk.shape[1]):
                lon, lat = longitude_tdisk[j, k], latitude_tdisk[j, k]

                if longitude_range[0] <= lon <= longitude_range[1] and latitude_range[0] <= lat <= latitude_range[1]:
                    lon_idx = np.clip(np.digitize(lon, longitude_bins_left) - 1, 0, len(longitude_bins_left) - 2)
                    lat_idx = np.clip(np.digitize(lat, latitude_bins_left) - 1, 0, len(latitude_bins_left) - 2)

                    if not np.isnan(temperature[scan_index1, j, k]):
                        temp_sum1[lat_idx, lon_idx] = np.nansum([temp_sum1[lat_idx, lon_idx], temperature[scan_index1, j, k]])
                        npoints1[lat_idx, lon_idx] += 1

                    if not np.isnan(temperature[scan_index2, j, k]):
                        temp_sum2[lat_idx, lon_idx] = np.nansum([temp_sum2[lat_idx, lon_idx], temperature[scan_index2, j, k]])
                        npoints2[lat_idx, lon_idx] += 1

                    lon_idx_right = np.clip(np.digitize(lon, longitude_bins_right) - 1, 0, len(longitude_bins_right) - 2)
                    lat_idx_right = np.clip(np.digitize(lat, latitude_bins_right) - 1, 0, len(latitude_bins_right) - 2)

                    if not np.isnan(temperature[scan_index1, j, k]):
                        temp_sum_right1[lat_idx_right, lon_idx_right] = np.nansum([temp_sum_right1[lat_idx_right, lon_idx_right], temperature[scan_index1, j, k]])
                        npoints_right1[lat_idx_right, lon_idx_right] += 1

                    if not np.isnan(temperature[scan_index2, j, k]):
                        temp_sum_right2[lat_idx_right, lon_idx_right] = np.nansum([temp_sum_right2[lat_idx_right, lon_idx_right], temperature[scan_index2, j, k]])
                        npoints_right2[lat_idx_right, lon_idx_right] += 1

        average_temp1 = compute_average(temp_sum1, npoints1)
        average_temp2 = compute_average(temp_sum2, npoints2)
        average_temp_right1 = compute_average(temp_sum_right1, npoints_right1)
        average_temp_right2 = compute_average(temp_sum_right2, npoints_right2)

        average_temp1 = combine_averages(average_temp1, average_temp2, npoints1, npoints2)
        average_temp_right1 = combine_averages(average_temp_right1, average_temp_right2, npoints_right1, npoints_right2)

        longitude_mesh_left, latitude_mesh_left = create_meshgrid(longitude_bins_left, latitude_bins_left)
        longitude_mesh_right, latitude_mesh_right = create_meshgrid(longitude_bins_right, latitude_bins_right)

        ax_left = fig.add_axes([0.06, 0.69 - ((0.29)*i) + (0.655*10)/7.5 * counter, (10 * 0.26)/12, (10 * 0.18)/7.5])
        ax_left.pcolormesh(longitude_mesh_left, latitude_mesh_left, average_temp1, cmap='coolwarm', shading='auto', alpha=1, vmin=600, vmax=1600)
        ax_left.set_ylabel("Latitude", fontsize = 10)
        ax_left.set_xlim(-120, 60)
        ax_left.set_ylim(-60, 65)
        ax_left.set_title(f"GOLD: {start_times_tdisk[i].strftime('%Y-%m-%d %H:%M')} to {end_times_tdisk[i].strftime('%H:%M')}", fontsize=10)

        # Apply mask to second dataset and compute difference
        masked_temp = np.where(np.isnan(average_temp1), np.nan, average_gold_temp)
        valid_mask = ~np.isnan(average_temp1) & ~np.isnan(masked_temp)
        diff = np.where(valid_mask, average_temp1 - masked_temp, np.nan)
        mean_diff = np.nanmean(diff)
        mean_diffs.append(mean_diff)
        rms_diff = np.sqrt(np.nanmean(diff**2))
        rms_vals.append(rms_diff)
        mean_percent_dif = (mean_diff / np.nanmean(average_temp1)) * 100
        mean_percent_diffs.append(mean_percent_dif)
        mean_percent_rms = (rms_diff / np.nanmean(average_temp1)) * 100
        mean_percent_rms_vals.append(mean_percent_rms)
        overlapping_bins = np.count_nonzero(valid_mask)

        print(f"Plot {i+1}: Mean Diff = {mean_diff:.1f}, RMS = {rms_diff:.1f}, Overlapping Bins = {overlapping_bins}")
        print(f"Plot {i+1}: MPD: {mean_percent_dif:.1f}, MP RMS: {mean_percent_rms:.1f}")

        flat1 = average_temp1.flatten()
        flat2 = average_gold_temp.flatten()

        #valid_mask = ~np.isnan(flat1) & ~np.isnan(flat2)
        #clean1 = flat1[valid_mask]
        #clean2 = flat2[valid_mask]
        clean1 = average_temp1[valid_mask]
        clean2 = masked_temp[valid_mask]

        if len(clean1) > 1:
            corr_score, p_value = pearsonr(clean1, clean2)
            corr_scores.append(corr_score)
        else:
            print("Not enough valid data points for correlation.")

        # Plot the right dataset in the second column
        ax_right = fig.add_axes([0.295, 0.69 - ((0.29)*i) + (0.655*10)/7.5 * counter, (10 * 0.26)/12, (10 * 0.18)/7.5])
        c2 = ax_right.pcolormesh(longitude_mesh_right, latitude_mesh_right, masked_temp, cmap='coolwarm', shading='auto', alpha=1, vmin=600, vmax=1600)
        ax_right.set_xlim(-120, 60)
        ax_right.set_ylim(-60, 65)
        ax_right.set_title(f"GITM: {start_times_tdisk[i].strftime('%Y-%m-%d %H:%M')} to {end_times_tdisk[i].strftime('%H:%M')}", fontsize=10)
        ax_right.set_yticks([])
        ax_right.text(0.05, 0.95, f'Diff: {mean_diff:.0f}, {mean_percent_dif:.0f}%', transform=ax_right.transAxes, fontsize=10, color='black', verticalalignment='top')
        ax_right.text(0.05, 0.85, f'RMS: {rms_diff:.0f}, {mean_percent_rms:.0f}%', transform=ax_right.transAxes, fontsize=10, color='black', verticalalignment='top')

        ############################ SCATTER PLOT ###############################
        ax_scatter = fig.add_axes([0.53, 0.69 - ((0.29)*i) + (0.655*10)/7.5 * counter, (10*0.36)/12, (10*0.18)/7.5])
        longitudes_valid, latitudes_valid, temp_flat, scatter_time = scatter_data_tdisk[i]
        ax_scatter.scatter(longitudes_valid, latitudes_valid, c=temp_flat, cmap='coolwarm', s=15, vmin=600, vmax=1600)

        # Draw the red bounding box
        lon_min, lon_max = -120, 60
        lat_min, lat_max = -60, 65
        ax_scatter.plot([lon_min, lon_max], [lat_min, lat_min], 'r-', linewidth=2)
        ax_scatter.plot([lon_min, lon_max], [lat_max, lat_max], 'r-', linewidth=2)
        ax_scatter.plot([lon_min, lon_min], [lat_min, lat_max], 'r-', linewidth=2)
        ax_scatter.plot([lon_max, lon_max], [lat_min, lat_max], 'r-', linewidth=2)
        ax_scatter.set_xlim(-180, 180)
        ax_scatter.set_ylim(-90, 90)
        ax_scatter.set_title(f"GITM Global at {scatter_time.strftime('%Y-%m-%d %H:%M')}", fontsize=10)
        ax_scatter.set_ylabel("Latitude", fontsize=10, rotation=270)
        ax_scatter.yaxis.tick_right() 
        ax_scatter.yaxis.set_label_position("right") 

        if (i+1)% 3 == 0 and i > 0:
            ax_left.set_xlabel("Longitude", fontsize = 10)
            ax_right.set_xlabel("Longitude", fontsize = 10)
            ax_scatter.set_xlabel("Longitude", fontsize = 10)
        else:
            ax_left.set_xticks([])
            ax_right.set_xticks([])
            ax_scatter.set_xticks([])

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.105, (10 * 0.02)/12, (10 * 0.63)/7.5])
        fig.colorbar(c2, cax=cbar_ax, label="Temperature (K)")

        if (i + 1) % 3 == 0:
            if plot_option == 1 and (weighted_option == 2 or weighted_option == 3):
                plt.savefig(os.path.join(output_dir, f"tdisk_comp_plot_{date_str}_{weighted_option}_{value:.0f}_threshset{thresh}_{((i+1)//3 - 1):d}.png"), dpi=300)
            elif plot_option == 1 and weighted_option == 0 and isinstance(value, list):
                plt.savefig(os.path.join(output_dir, f"tdisk_comp_plot_{date_str}_multialt_{value[0]:.0f}_{value[-1]:.0f}_{((i + 1)// 3 - 1):d}.png"), dpi=300)
            elif plot_option == 1 and (weighted_option == 0 or weighted_option == 1 or weighted_option == 6 or weighted_option == 7):
                plt.savefig(os.path.join(output_dir, f"tdisk_comp_plot_{date_str}_{weighted_option}_{value:.0f}_{((i+1)//3 - 1):d}.png"), dpi=300)
            elif plot_option == 1 and (weighted_option == 5):
                plt.savefig(os.path.join(output_dir, f"tdisk_comp_plot_{date_str}_{weighted_option}_{((i+1)//3 - 1):d}.png"), dpi=300)
            else:
                plt.close(fig)

    print(f"Averages for the day {date_str}: Mean Diff: {np.mean(mean_diffs):.4f}, RMS: {np.mean(rms_vals):.4f}, Corr: {np.mean(corr_scores):.4f}")
    if (pairs_tdisk % 3) != 0:
        if plot_option == 1 and (weighted_option == 2 or weighted_option == 3):
            plt.savefig(os.path.join(output_dir, f"tdisk_comp_plot_{date_str}_{weighted_option}_{value:.0f}_threshset{thresh}_{(pairs_tdisk // 3):d}.png"), dpi=300)
        elif plot_option == 1 and weighted_option == 0 and isinstance(value, list):
            plt.savefig(os.path.join(output_dir, f"tdisk_comp_plot_{date_str}_multialt_{value[0]:.0f}_{value[-1]:.0f}_{(pairs_tdisk // 3):d}.png"), dpi=300)
        elif plot_option == 1 and (weighted_option == 0 or weighted_option == 1 or weighted_option == 6 or weighted_option == 7):
            plt.savefig(os.path.join(output_dir, f"tdisk_comp_plot_{date_str}_{weighted_option}_{value:.0f}_{(pairs_tdisk // 3):d}.png"), dpi=300)
        elif plot_option == 1 and (weighted_option == 5):
            plt.savefig(os.path.join(output_dir, f"tdisk_comp_plot_{date_str}_{weighted_option}_{(pairs_tdisk // 3):d}.png"), dpi=300)
        else:
            plt.close(fig)

    tdisk_results = {
        "mean_diffs" : mean_diffs,
        "mean_percent_diffs" : mean_percent_diffs,
        "rms_vals" : rms_vals,
        "mean_percent_rms_vals" : mean_percent_rms_vals,
        "corr_scores": corr_scores,
        "nscans": pairs_tdisk
    }
    return tdisk_results

def tdisk_alt_comp(nc_files, alt_list, directory, output_dir):
    '''
    Used to generate multiple altitude comaprison plots on the same figure. Designed to take in 3 altitudes for each inputted day and will plot
    each scan window for all 3 alts on one figure, and then will plot all the typical figures.
    '''
    for file in nc_files:
        gold_tdisk_vars = extract_gold_tdisk(file)
        #gold_tdisk_data = gold_tdisk_processing(gold_tdisk_vars)
        valid_tdisk_results = extract_valid_tdisk_points(
            gold_tdisk_vars["temperature"],
            gold_tdisk_vars["time_data_tdisk"],
            gold_tdisk_vars["latitude_tdisk"],
            gold_tdisk_vars["longitude_tdisk"],
            gold_tdisk_vars["sza"])
        
        gold_tdisk_vars["valid_tdisk_results"] = valid_tdisk_results
        temperature = gold_tdisk_vars["temperature"]
        start_times_tdisk, end_times_tdisk = get_scan_pair_times(valid_tdisk_results, temperature.shape[0])
        gold_times = [p["time"] for p in valid_tdisk_results]
        time_to_file_map = find_closest_filesv2(gold_times, directory)

        # Process GITM for each threshold and store separately
        gitm_results_per_alt = []
        for alt in alt_list:
            cache = {}
            gitm_result = gitm_tdisk_processing_v2(valid_tdisk_results, time_to_file_map, 0, alt, None, cache, start_times_tdisk, end_times_tdisk)
            gitm_results_per_alt.append(gitm_result)

        # Interleave GITM data by scan
        n_scans = len(gitm_results_per_alt[0]["mesh_data_tdisk"])
        total_mesh = []
        total_scatter = []
        total_alt = []

        for scan_idx in range(n_scans):
            for alt_idx, alt_result in enumerate(gitm_results_per_alt):
                total_mesh.append(alt_result["mesh_data_tdisk"][scan_idx])
                total_scatter.append(alt_result["scatter_data_tdisk"][scan_idx])
                total_alt.append(alt_result["value"])

        pair_size = 2
        n_scans_gold = gold_tdisk_vars["n_scans_tdisk"]
        n_pairs = n_scans_gold // pair_size
        n_alt = len(alt_list)
        total_scans = n_scans_gold * n_alt

        gold_vars_repeated = {"n_scans_tdisk": total_scans}

        for k, v in gold_tdisk_vars.items():
            if isinstance(v, np.ndarray):
                if v.shape[0] == n_scans_gold:
                    v_pairs = v.reshape(n_pairs, pair_size, *v.shape[1:])
                    v_pairs_repeated = np.repeat(v_pairs[:, np.newaxis, ...], n_alt, axis=1)
                    gold_vars_repeated[k] = v_pairs_repeated.reshape(total_scans, *v.shape[1:])
                else:
                    gold_vars_repeated[k] = v
            else:
                gold_vars_repeated[k] = v

        start_times_tdisk_expanded = list(np.repeat(start_times_tdisk, 3))
        end_times_tdisk_expanded   = list(np.repeat(end_times_tdisk, 3))

        start_times_tdisk_repeated = list(np.tile(start_times_tdisk_expanded, n_alt))
        end_times_tdisk_repeated   = list(np.tile(end_times_tdisk_expanded, n_alt))
        gold_vars_repeated["n_scans_tdisk"] = total_scans

        total_gitm_results = {
            "mesh_data_tdisk": total_mesh,
            "scatter_data_tdisk": total_scatter,
            "value": total_alt,
            "weighted_option": 0, 
            "thresh": None}

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        tdisk_outputs = temp_data_comparison(total_gitm_results, gold_vars_repeated, start_times_tdisk_repeated, end_times_tdisk_repeated, 1, output_dir)