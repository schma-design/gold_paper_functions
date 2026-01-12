from datetime import datetime
import numpy as np
import os
from nc_test import read_gitm_headers_nc, read_gitm_one_file_nc
from argparse import Namespace
from gitm_routines import read_gitm_headers, remap_variable_names, read_gitm_one_file
from thermo_plot_tmpv0 import get_file_info, read_in_model_files
from collections import defaultdict
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from comp_gen_misc import generate_custom_bins, compute_average, combine_averages, create_meshgrid, find_closest_filesv2
from comp_gold_funcs import extract_gold_on2, get_scan_pair_times, extract_valid_on2_points

def vertically_integrate(value, alts, calc3D=False):
    [nLons, nLats, nAlts] = value.shape
    integrated = np.zeros((nLons, nLats, nAlts))
    descending = np.arange(nAlts-2, -1, -1)
    dz = alts[:, :, -1] - alts[:, :, -2]
    integrated[:, :, -1] = value[:, :, -1] * dz
    for i in descending:
        dz = alts[:, :, i+1] - alts[:, :, i]
        integrated[:, :, i] = integrated[:, :, i+1] + value[:, :, i] * dz
    if not calc3D:
        integrated = integrated[:, :, 0]
    return integrated, alts

def calculate_on2(headers, data, iO_, iN2_, Alts3d, nLons, nLats, nAlts, thresh):
    on2all = []
    altsall = []
    times = []
    
    times.append(headers["time"][0])
    oDensity = data[iO_]
    n2Density = data[iN2_]
    n2Int, n2alts = vertically_integrate(n2Density, Alts3d, calc3D = True)
    oInt, o2alts = vertically_integrate(oDensity, Alts3d, calc3D = True)
    altsall.append(o2alts)
    on2 = np.zeros((nLons, nLats))
    iAlts = np.arange(nAlts)
    #o_interp = np.zeros((nLons, nLats))
    #n2_interp = np.zeros((nLons, nLats))
    alt_interp = np.zeros((nLons, nLats))
    limit_altitudes = np.full((nLons, nLats), np.nan)

    for iLat in range(nLats):
        for iLon in range(nLons):
            n21d = n2Int[iLon, iLat, :] / (thresh * 1e21)
            o1d = oInt[iLon, iLat, :]
            alt1d = Alts3d[iLon, iLat, :]
            valid_indices = iAlts[n21d < 1.0]
            if len(valid_indices) == 0:
                continue
            i = valid_indices[0]
            r = (1.0 - n21d[i]) / (n21d[i-1] - n21d[i])
            n2 = (r * n21d[i-1] + (1.0 - r) * n21d[i]) * (thresh * 1e21)
            o = r * o1d[i-1] + (1.0 - r) * o1d[i]
            alt = r * alt1d[i-1] + (1.0-r) * alt1d[i]
            on2[iLon, iLat] = o / n2
            #n2_interp[iLon, iLat] = n2
            #o_interp[iLon, iLat] = o
            alt_interp[iLon, iLat] = alt
            alt_interp2 = r * Alts3d[iLon, iLat, i-1] + (1.0 - r) * Alts3d[iLon, iLat, i]
            limit_altitudes[iLon, iLat] = alt_interp2
    #print("Mean N2 threshold altitude:", np.nanmean(limit_altitudes) / 1000, "km")

    on2all.append(on2)
    #altsall.append(alt_interp)
    return np.array(times), np.array(on2all), limit_altitudes

def gitm_on2_processing(file_paths_on2, thresh, wind_option, cache=None):

    '''
    Older version of V2, takes in a single time file to generate comparisons. Used to generate scatter data in global plots
    to fill in gaps for the V2 function. This is also the only function that collects wind data.
    '''

    if cache is None:
        cache = {}
    wind_cache = {}
    
    scatter_data_on2 = []
    mesh_data_on2 = []
    xwind_loc = []
    ywind_loc = []
    xwind_data = []
    ywind_data = []
    avg_n2_lim = []
    n2_lim_map = []

    # Loop through each file and process it
    for i, file_path in enumerate(file_paths_on2):
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            continue

        on2_key = f"{file_path}_on2"
        if on2_key in cache:
            headers, data, var_index_map, iO_, iN2_ = cache[on2_key]
        else:
            if file_path.endswith(".nc"):
                headers = read_gitm_headers_nc(file_path)
                iO_ = 3
                iN2_ = 4
                iVars_ = [0, 1, 2, iO_, iN2_]
                data = read_gitm_one_file_nc(file_path, iVars_)
                var_index_map = {i: j for j, i in enumerate(iVars_)}
            else: 
                # Identify the variable indices for O and N2 
                headers = read_gitm_headers(files=[file_path])
                vars = remap_variable_names(headers["vars"])
                iO_ = vars.index('[O] (/m3)')
                iN2_ = vars.index('[N2] (/m3)')

                # Extract longitude, latitude, altitude information
                iVars_ = sorted(set([0, 1, 2, iO_, iN2_]))
                data = read_gitm_one_file(headers["filename"][0], iVars_)
                var_index_map = {i: j for j, i in enumerate(iVars_)}
            cache[on2_key] = (headers, data, var_index_map, iO_, iN2_)

        #iO_mapped = var_index_map[iO_]
        #iN2_mapped = var_index_map[iN2_]
        Alts3d = data[var_index_map[2]]
        #Alts = Alts3d[0, 0, :] / 1000.0  # Convert to km
        Lons = data[var_index_map[0]][:, 0, 0] * (180.0 / np.pi)  # Convert to degrees
        Lats = data[var_index_map[1]][0, :, 0] * (180.0 / np.pi)  # Convert to degrees
        [nLons, nLats, nAlts] = data[var_index_map[0]].shape

        # Compute O/N2 using the provided function
        times, gon2, limit_altitudes = calculate_on2(headers, data, iO_, iN2_, Alts3d, nLons, nLats, nAlts, thresh)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        limit_altitudes2 = limit_altitudes
        limit_altitudes = limit_altitudes.T  # Transpose to shape (184, 94)
        n2_lim_map.append(limit_altitudes2)
        avg_limit_alt_km = np.nanmean(limit_altitudes / 1000.0)
        avg_n2_lim.append(avg_limit_alt_km)

    ############################ Wind Pattern Studies ####################################
        #if file_path.endswith(".bin"):
        if wind_option == 1:
            wind_key = (file_path, avg_limit_alt_km)
            if wind_key in wind_cache:
                header_wind, data_wind = wind_cache[wind_key]
            else:
                args = Namespace(
                filelist=[file_path],  # Required argument
                winds=True,  # Enable wind plotting
                cut='alt', 
                var=3,    
                alt=avg_limit_alt_km,
                lat=-100.0,
                lon=-100.0,
                mean=False,
                tec=False,
                on2=False,
                nstep=5
                )   
                if file_path.endswith(".bin"):
                    header_wind = get_file_info(args)
                    data_wind = read_in_model_files(args, header_wind)
                else:
                    header_wind = get_file_info(args)
                    varname = 'O'
                    ivar = header_wind['vars'].index(varname)
                    if args.var != ivar:
                        print("ALB overriding variable index: %i -> %i" %(args.var, ivar))
                        args.var=ivar
                    data_wind = read_in_model_files(args, header_wind)
                wind_cache[wind_key] = (header_wind, data_wind)

            winds_x = data_wind['winds_x']
            winds_y = data_wind['winds_y']
            x_pos = data_wind['x_pos']
            y_pos = data_wind['y_pos']

            # Reshape winds_x and winds_y to remove the extra dimension (1)
            winds_x = winds_x[0, :, :]  # Shape will be (94, 184)
            winds_y = winds_y[0, :, :]   # Shape will be (94, 184)

            # Create meshgrid for xp and yp to match the filtered wind data grid
            xp, yp = np.meshgrid(x_pos, y_pos)
            xp = xp[::args.nstep, ::args.nstep]  # Adjust xp to match wind data grid
            xp = xp[1:, 1:]
            yp = yp[::args.nstep, ::args.nstep]  # Adjust yp to match wind data grid
            yp = yp[1:, 1:]
            xwinds = winds_x[::args.nstep, ::args.nstep]
            ywinds = winds_y[::args.nstep, ::args.nstep]
            xwinds = xwinds[1:, 1:]
            ywinds = ywinds[1:, 1:]

            # Copy xp to modify it to a -180 to 180 range
            xp_copy = xp.copy()
            # Reformat xp from 0-360 to -180 to 180 range
            xp_copy[xp_copy > 180] -= 360  # Shift values greater than 180 to the negative range
            xwind_loc.append(xp_copy)  
            ywind_loc.append(yp)
            xwind_data.append(xwinds)
            ywind_data.append(ywinds) 
            #else:
                #xwind_loc = None
                #ywind_loc = None
                #xwind_data = None
                #ywind_data = None

    ################################################################################################

        # Convert Lons and Lats into 1D arrays for scatter plot
        longitudes, latitudes = np.meshgrid(Lons, Lats, indexing='ij')
        longitudes = longitudes.flatten()
        latitudes = latitudes.flatten()

        # Flatten gon2 to match longitude and latitude points
        gon2_flat = gon2.mean(axis=0).flatten()

        # Clean Data for Mesh Plot
        gon2_cleaned = gon2.copy()

        # Flatten the Lons, Lats, and gon2 arrays for easier indexing
        longitudes, latitudes = np.meshgrid(Lons, Lats, indexing='ij')
        longitudes = longitudes.flatten()
        latitudes = latitudes.flatten()
        gon2_flat = gon2_cleaned.flatten()

        # Filter out invalid longitude and latitude values
        valid_mask = (longitudes >= 0) & (longitudes <= 360) & (latitudes >= -90) & (latitudes <= 90)
        longitudes[longitudes > 180] -= 360

        # Apply the mask to filter out invalid data points
        longitudes_valid = longitudes[valid_mask]
        latitudes_valid = latitudes[valid_mask]
        gon2_valid = gon2_flat[valid_mask]

        # Define the longitude and latitude ranges
        longitude_range = (-180, 180)
        latitude_range = (-90, 90)
        scatter_data_on2.append((longitudes, latitudes, gon2_flat, times[0]))

        latitude_bins, longitude_bins = generate_custom_bins(latitude_range, longitude_range)

        # Initialize grid to store sum of gon2 values and number of points in each bucket
        n_latitude_buckets = len(latitude_bins) - 1
        n_longitude_buckets = len(longitude_bins) - 1
        gon2_sum = np.zeros((n_latitude_buckets, n_longitude_buckets))
        npoints = np.zeros((n_latitude_buckets, n_longitude_buckets))

        # Loop through the cleaned data and allocate points to their respective buckets
        for i in range(len(longitudes_valid)):  # Loop over the cleaned longitude and latitude points
            lon = longitudes_valid[i]
            lat = latitudes_valid[i]
            gon2_value = gon2_valid[i]

            # Find the corresponding bucket for this point
            lon_idx = np.digitize(lon, longitude_bins) - 1
            lat_idx = np.digitize(lat, latitude_bins) - 1
            if not (0 <= lon_idx < n_longitude_buckets and 0 <= lat_idx < n_latitude_buckets):
                continue

            # Add the gon2 value to the corresponding bucket and increment npoints
            gon2_sum[lat_idx, lon_idx] += gon2_value
            npoints[lat_idx, lon_idx] += 1

        # Calculate the average gon2 score for each bucket (avoid division by zero)
        average_gon2 = np.divide(gon2_sum, npoints, where=npoints > 0)

        # Set values < 0.01 to NaN (will be plotted as white)
        average_gon2[average_gon2 < 0.01] = np.nan

        # Store the mesh data
        mesh_data_on2.append((longitude_bins, latitude_bins, average_gon2, times[0]))

    gitm_on2_outputs = {
        "mesh_data_on2": mesh_data_on2,
        "scatter_data_on2": scatter_data_on2,
        "xwind_data": xwind_data,
        "xwind_loc": xwind_loc,
        "ywind_data": ywind_data,
        "ywind_loc": ywind_loc,
        "avg_n2_lim": avg_n2_lim,
        "n2_lim_map": n2_lim_map,
        "thresh": thresh
    }
    return gitm_on2_outputs

def gitm_on2_processing_v2(points, time_to_file_map, cache, thresh, start_times, end_times):
    """
    Processes GITM ON2 data by matching each GOLD point to a GITM point to retrieve the relevant information
    Uses the original gitm_on2_processing function for scatter data used in comparison visualization
    """
    normalized_start_times = [
        datetime.fromisoformat(t) if isinstance(t, str) else t for t in start_times]
    
    normalized_end_times = [
        datetime.fromisoformat(t) if isinstance(t, str) else t for t in end_times]

    start_times = normalized_start_times
    end_times = normalized_end_times
    gitm_point_results = []
    mesh_data_on2 = [] 
    central_time_files = []

    points_by_window = {i: [] for i in range(len(start_times))}
    scan_windows = list(zip(start_times, end_times))

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
        file_times = sorted(file_times, key=lambda x: x[0])

        if file_times:
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
                # Load from cache or read new file
                on2_key = f"{gitm_file}_on2"
                if on2_key in cache:
                    headers, data, var_index_map, iO_, iN2_ = cache[on2_key]
                else:
                    if gitm_file.endswith(".nc"):
                        headers = read_gitm_headers_nc(gitm_file)
                        iO_ = 3
                        iN2_ = 4
                        iVars_ = [0, 1, 2, iO_, iN2_]
                        data = read_gitm_one_file_nc(gitm_file, iVars_)
                        var_index_map = {i: j for j, i in enumerate(iVars_)}
                    else:
                        headers = read_gitm_headers(files=[gitm_file])
                        vars = remap_variable_names(headers["vars"])
                        iO_ = vars.index('[O] (/m3)')
                        iN2_ = vars.index('[N2] (/m3)')
                        iVars_ = sorted(set([0, 1, 2, iO_, iN2_]))  # lon, lat, alt, O, N2
                        data = read_gitm_one_file(headers["filename"][0], iVars_)
                        var_index_map = {i: j for j, i in enumerate(iVars_)}
                    cache[on2_key] = (headers, data, var_index_map, iO_, iN2_)

                iLon = var_index_map[0]
                iLat = var_index_map[1]
                iAlt = var_index_map[2]
                Lons = data[iLon][:, 0, 0] * (180.0 / np.pi)
                Lons[Lons > 180] -= 360
                Lats = data[iLat][0, :, 0] * (180.0 / np.pi)
                nLons, nLats, nAlts = data[iLon].shape
                _, gon2, *_ = calculate_on2(headers, data, iO_, iN2_, data[iAlt], nLons, nLats, nAlts, thresh)

                # Process all points
                for point in file_points:
                    plat = point["lat"]
                    plon = point["lon"]
                    j = np.abs(Lats - plat).argmin()
                    k = np.abs(Lons - plon).argmin()
                    val = gon2[0, k, j] if 0 <= k < gon2.shape[1] and 0 <= j < gon2.shape[2] else np.nan
                    gitm_point_results.append({
                        "lat": Lats[j],
                        "lon": Lons[k],
                        "on2": val,
                        "gitm_time": start_time
                    })

        # Generate mesh
        if gitm_point_results:
            all_lats = np.array([p["lat"] for p in gitm_point_results])
            all_lons = np.array([p["lon"] for p in gitm_point_results])
            all_vals = np.array([p["on2"] for p in gitm_point_results])
            all_lons[all_lons > 180] -= 360

            latitude_range = (-90, 90)
            longitude_range = (-180, 180)
            latitude_bins, longitude_bins = generate_custom_bins(latitude_range, longitude_range)
            n_latitude_buckets = len(latitude_bins) - 1
            n_longitude_buckets = len(longitude_bins) - 1
            gon2_sum = np.zeros((n_latitude_buckets, n_longitude_buckets))
            npoints = np.zeros((n_latitude_buckets, n_longitude_buckets))

            for lat, lon, val in zip(all_lats, all_lons, all_vals):
                lon_idx = np.digitize(lon, longitude_bins) - 1
                lat_idx = np.digitize(lat, latitude_bins) - 1

                # Skip points outside the defined mesh
                if not (0 <= lon_idx < n_longitude_buckets and 0 <= lat_idx < n_latitude_buckets):
                    continue

                # Only accumulate valid numeric values
                if np.isfinite(val):
                    gon2_sum[lat_idx, lon_idx] += val
                    npoints[lat_idx, lon_idx] += 1

            with np.errstate(divide='ignore', invalid='ignore'):
                average_gon2 = np.divide(gon2_sum, npoints)
                average_gon2[npoints == 0] = np.nan

            # Mask out physically invalid values
            invalid_mask = (average_gon2 < 0.01) | (average_gon2 > 10) | np.isnan(average_gon2)
            average_gon2[invalid_mask] = np.nan

            # Fill NaN bins with nearest valid neighbor
            # Without the following segment some mesh cells miss data.
            # Mainly for the plotting representation
            mask = np.isnan(average_gon2)
            if np.any(mask):
                # Get indices of nearest valid values for every NaN
                nearest_idx = distance_transform_edt(mask,
                                                    return_distances=False,
                                                    return_indices=True)
                average_gon2 = average_gon2[tuple(nearest_idx)]
            
            if gitm_point_results:
                mesh_data_on2.append((longitude_bins, latitude_bins, average_gon2, start_time))
            else: 
                mesh_data_on2.append((longitude_bins, latitude_bins, np.full((n_latitude_buckets, n_longitude_buckets), np.nan), start_time))

    all_fixed_scatter_outputs = []
    gitm_on2_results = gitm_on2_processing(central_time_files, thresh, 1, cache)
    xwind_data = gitm_on2_results["xwind_data"]
    ywind_data = gitm_on2_results["ywind_data"]
    xwind_loc = gitm_on2_results["xwind_loc"]
    ywind_loc = gitm_on2_results["ywind_loc"]
    all_fixed_scatter_outputs.extend(gitm_on2_results["scatter_data_on2"])

    gitm_on2_outputs2 = {
        "mesh_data_on2": mesh_data_on2,
        "scatter_data_on2": all_fixed_scatter_outputs,
        "thresh": thresh, 
        "xwind_data": xwind_data, 
        "ywind_data": ywind_data,
        "xwind_loc": xwind_loc,
        "ywind_loc": ywind_loc
    }
    return gitm_on2_outputs2

def on2_data_comparison(gitm_on2_outputs, gold_on2_vars, start_times_on2, end_times_on2, plot_option, output_dir): 

    '''
    Main comparison function. Processes data for quantitative comparisons and can plot global contour plots if desired.
    Produces mean difference, RMS, and cross correlation data.
    If a variable has "left" in the name, it refers to GOLD data (i.e. left column in the global plots)
    If a variable has "right" in the name, it refers to the GITM data plotted on the GOLD mesh (i.e. central column in global plots)
    The "scatter" data is GITM data that does not go through the binning procedure (i.e. the right column in the global plots)
    '''

    latitude_on2 = gold_on2_vars["latitude_on2"]
    longitude_on2 = gold_on2_vars["longitude_on2"]
    on2 = gold_on2_vars["on2"]
    mesh_data_on2 = gitm_on2_outputs["mesh_data_on2"]
    scatter_data_on2 = gitm_on2_outputs["scatter_data_on2"]
    xwind_data = gitm_on2_outputs["xwind_data"]
    xwind_loc = gitm_on2_outputs["xwind_loc"]
    ywind_data = gitm_on2_outputs["ywind_data"]
    ywind_loc = gitm_on2_outputs["ywind_loc"]
    n_scans = gold_on2_vars["n_scans"]
    thresh = gitm_on2_outputs["thresh"]
    pairs = int(n_scans / 2)

    if start_times_on2 and isinstance(start_times_on2[0], datetime):
        date_str = start_times_on2[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')
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

    for i in range(pairs):
        if i % 3 == 0 and i > 0:
            fig = plt.figure(figsize=(12, 7.5))
            counter += 1

        # Indexing for scan data
        scan_index1 = 2 * i
        scan_index2 = 2 * i + 1

        # Extract mesh data and plot time
        longitude_bins_left, latitude_bins_left = longitude_bins, latitude_bins
        if len(mesh_data_on2) > 1:
            longitude_bins_right, latitude_bins_right, average_gon2, plot_time = mesh_data_on2[i]
        else:
            longitude_bins_right, latitude_bins_right, average_gon2, plot_time = mesh_data_on2[0]

        # Initialize grids for both datasets
        on2_sum1 = np.full((len(latitude_bins_left) - 1, len(longitude_bins_left) - 1), np.nan)
        on2_sum2 = np.full((len(latitude_bins_left) - 1, len(longitude_bins_left) - 1), np.nan)
        npoints1 = np.zeros((len(latitude_bins_left) - 1, len(longitude_bins_left) - 1))
        npoints2 = np.zeros((len(latitude_bins_left) - 1, len(longitude_bins_left) - 1))

        on2_sum_right1 = np.full((len(latitude_bins_right) - 1, len(longitude_bins_right) - 1), np.nan)
        on2_sum_right2 = np.full((len(latitude_bins_right) - 1, len(longitude_bins_right) - 1), np.nan)
        npoints_right1 = np.zeros((len(latitude_bins_right) - 1, len(longitude_bins_right) - 1))
        npoints_right2 = np.zeros((len(latitude_bins_right) - 1, len(longitude_bins_right) - 1))

        # Accumulate ON2 values into bins for both datasets
        for j in range(longitude_on2.shape[0]):
            for k in range(longitude_on2.shape[1]):
                lon, lat = longitude_on2[j, k], latitude_on2[j, k]

                if longitude_range[0] <= lon <= longitude_range[1] and latitude_range[0] <= lat <= latitude_range[1]:
                    # Left dataset
                    lon_idx = np.clip(np.digitize(lon, longitude_bins_left) - 1, 0, len(longitude_bins_left) - 2)
                    lat_idx = np.clip(np.digitize(lat, latitude_bins_left) - 1, 0, len(latitude_bins_left) - 2)

                    if not np.isnan(on2[scan_index1, j, k]):
                        on2_sum1[lat_idx, lon_idx] = np.nansum([on2_sum1[lat_idx, lon_idx], on2[scan_index1, j, k]])
                        npoints1[lat_idx, lon_idx] += 1

                    if not np.isnan(on2[scan_index2, j, k]):
                        on2_sum2[lat_idx, lon_idx] = np.nansum([on2_sum2[lat_idx, lon_idx], on2[scan_index2, j, k]])
                        npoints2[lat_idx, lon_idx] += 1

                    # Right dataset
                    lon_idx_right = np.clip(np.digitize(lon, longitude_bins_right) - 1, 0, len(longitude_bins_right) - 2)
                    lat_idx_right = np.clip(np.digitize(lat, latitude_bins_right) - 1, 0, len(latitude_bins_right) - 2)

                    if not np.isnan(on2[scan_index1, j, k]):
                        on2_sum_right1[lat_idx_right, lon_idx_right] = np.nansum([on2_sum_right1[lat_idx_right, lon_idx_right], on2[scan_index1, j, k]])
                        npoints_right1[lat_idx_right, lon_idx_right] += 1

                    if not np.isnan(on2[scan_index2, j, k]):
                        on2_sum_right2[lat_idx_right, lon_idx_right] = np.nansum([on2_sum_right2[lat_idx_right, lon_idx_right], on2[scan_index2, j, k]])
                        npoints_right2[lat_idx_right, lon_idx_right] += 1

        average_on2_1 = compute_average(on2_sum1, npoints1)
        average_on2_2 = compute_average(on2_sum2, npoints2)
        average_on2_right1 = compute_average(on2_sum_right1, npoints_right1)
        average_on2_right2 = compute_average(on2_sum_right2, npoints_right2)
        average_on2_1 = combine_averages(average_on2_1, average_on2_2, npoints1, npoints2)
        average_on2_right1 = combine_averages(average_on2_right1, average_on2_right2, npoints_right1, npoints_right2)

        longitude_mesh_left, latitude_mesh_left = create_meshgrid(longitude_bins_left, latitude_bins_left)
        longitude_mesh_right, latitude_mesh_right = create_meshgrid(longitude_bins_right, latitude_bins_right)

        # Plot the left dataset in the first column
        ax_left = fig.add_axes([0.06, 0.69 - ((0.29)*i) + (0.655*10)/7.5 * counter, (10 * 0.26)/12, (10 * 0.18)/7.5])
        c1 = ax_left.pcolormesh(longitude_mesh_left, latitude_mesh_left, average_on2_1, cmap='coolwarm', shading='auto', alpha=0.95, vmin=0.1, vmax=1.8)
        ax_left.set_ylabel("Latitude", fontsize = 10)
        ax_left.set_xlim(-120, 60)
        ax_left.set_ylim(-60, 65)
        ax_left.set_title(f"GOLD: {start_times_on2[i].strftime('%Y-%m-%d %H:%M')} to {end_times_on2[i].strftime('%H:%M')}", fontsize = 10)

        # Apply mask to second dataset and compute difference
        masked_gon2 = np.where(np.isnan(average_on2_1), np.nan, average_gon2)
        #valid_mask = ~np.isnan(average_on2_right1) & ~np.isnan(masked_gon2)
        valid_mask = ~np.isnan(average_gon2) & ~np.isnan(average_on2_right1)
        #diff = np.where(valid_mask, average_on2_right1 - masked_gon2, np.nan)
        diff = np.where(valid_mask, average_on2_right1 - average_gon2, np.nan)
        mean_diff = np.nanmean(diff)
        mean_diffs.append(mean_diff)
        rms_diff = np.sqrt(np.nanmean(diff**2))
        rms_vals.append(rms_diff)
        mean_percent_dif = (mean_diff / np.nanmean(average_on2_right1)) * 100
        mean_percent_diffs.append(mean_percent_dif)
        mean_percent_rms = (rms_diff / np.nanmean(average_on2_right1)) * 100
        mean_percent_rms_vals.append(mean_percent_rms)
        overlapping_bins = np.count_nonzero(valid_mask)

        print(f"Plot {i+1}: Mean Diff = {mean_diff:.4f}, RMS = {rms_diff:.4f}, Overlapping Bins = {overlapping_bins}")

        flat1 = average_on2_right1.flatten()
        flat2 = masked_gon2.flatten()

        #valid_mask = ~np.isnan(flat1) & ~np.isnan(flat2)
        #clean1 = flat1[valid_mask]
        #clean2 = flat2[valid_mask]
        clean1 = average_on2_right1[valid_mask]
        clean2 = masked_gon2[valid_mask]

        if len(clean1) > 1:
            corr_score, p_value = pearsonr(clean1, clean2)
            corr_scores.append(corr_score)
        else:
            print("Not enough valid data points for correlation.")
        
        # Plot the right dataset in the second column
        ax_right = fig.add_axes([0.295, 0.69 - ((0.29)*i) + (0.655*10)/7.5 * counter, (10 * 0.26)/12, (10 * 0.18)/7.5])
        c2 = ax_right.pcolormesh(longitude_mesh_right, latitude_mesh_right, masked_gon2, cmap='coolwarm', shading='auto', alpha=0.95, vmin=0.1, vmax=1.8)
        ax_right.set_xlim(-120, 60)
        ax_right.set_ylim(-60, 65)
        ax_right.set_title(f"GITM: {start_times_on2[i].strftime('%Y-%m-%d %H:%M')} to {end_times_on2[i].strftime('%H:%M')}", fontsize = 10)
        ax_right.set_yticks([])

        ax_right.text(0.05, 0.95, f'Diff: {mean_diff:.2f}, {mean_percent_dif:.0f}%', transform=ax_right.transAxes, fontsize=10, color='black', verticalalignment='top')
        ax_right.text(0.05, 0.85, f'RMS: {rms_diff:.2f}, {mean_percent_rms:.0f}%', transform=ax_right.transAxes, fontsize=10, color='black', verticalalignment='top')

        ############################ SCATTER PLOT ###############################
        # Plot the scatter data in the third column
        ax_scatter = fig.add_axes([0.53, 0.69 - ((0.29)*i) + (0.655*10)/7.5 * counter, (10*0.36)/12, (10*0.18)/7.5])
        longitudes_valid, latitudes_valid, gon2_valid, scatter_time = scatter_data_on2[i]
        sc = ax_scatter.scatter(longitudes_valid, latitudes_valid, c=gon2_valid, cmap='coolwarm', s=15, vmin=0.1, vmax=1.8)

        # Draw the red bounding box
        lon_min, lon_max = -120, 60
        lat_min, lat_max = -60, 65
        ax_scatter.plot([lon_min, lon_max], [lat_min, lat_min], 'r-', linewidth=2)
        ax_scatter.plot([lon_min, lon_max], [lat_max, lat_max], 'r-', linewidth=2)
        ax_scatter.plot([lon_min, lon_min], [lat_min, lat_max], 'r-', linewidth=2)
        ax_scatter.plot([lon_max, lon_max], [lat_min, lat_max], 'r-', linewidth=2)
        ax_scatter.set_xlim(-180, 180)
        ax_scatter.set_ylim(-90, 90)
        #if xwind_data is not None:
        if xwind_data and ywind_data:
            ax_scatter.quiver(xwind_loc[i], ywind_loc[i], xwind_data[i], ywind_data[i], scale=10000.0, color='black')
        ax_scatter.set_ylabel("Latitude", fontsize=10, rotation=270)
        ax_scatter.yaxis.tick_right() 
        ax_scatter.yaxis.set_label_position("right") 
        ax_scatter.set_title(f"GITM Global at {scatter_time.strftime('%Y-%m-%d %H:%M')}", fontsize = 10)

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
        fig.colorbar(c2, cax=cbar_ax, label=r"O/N$_2$ Intensity")

        if (i + 1) % 3 == 0:
            if plot_option == 1 and isinstance(thresh, float):
                plt.savefig(os.path.join(output_dir, f"on2_comp_plot_{date_str}_{thresh}_{((i+1)//3 - 1):d}.png"), dpi=300)
            elif plot_option == 1 and len(thresh) > 1:
                plt.savefig(os.path.join(output_dir, f"on2_comp_plot_{date_str}_multithresh_{thresh[0]}_to_{thresh[-1]}_{((i+1)//3 - 1):d}.png"), dpi=300)
            else:
                plt.close(fig)

    if (pairs % 3) != 0:
        if plot_option == 1 and isinstance(thresh, float) == 1:
            plt.savefig(os.path.join(output_dir, f"on2_comp_plot_{date_str}_{thresh}_{(pairs // 3):d}.png"), dpi=300)
        elif plot_option == 1 and len(thresh) > 1:
            plt.savefig(os.path.join(output_dir, f"on2_comp_plot_{date_str}_multithresh_{thresh[0]}_to_{thresh[-1]}_{(pairs // 3):d}.png"), dpi=300)
        else:
            plt.close(fig)

    print(f"Averages for the day {date_str}: Mean Diff: {np.mean(mean_diffs):.4f}, RMS: {np.mean(rms_vals):.4f}, Corr: {np.mean(corr_scores):.4f}")
    on2_results = {
        "mean_diffs" : mean_diffs,
        "mean_percent_diffs" : mean_percent_diffs,
        "rms_vals" : rms_vals,
        "mean_percent_rms_vals" : mean_percent_rms_vals,
        "corr_scores": corr_scores,
        "nscans": pairs
    }
    return on2_results

def on2_thresh_comp(nc_files, num_den_thresh, directory, output_dir):

    for file in nc_files:
        gold_on2_vars = extract_gold_on2(file)
        
        # Extract scan info
        valid_on2_results = extract_valid_on2_points(
            gold_on2_vars["on2"],
            gold_on2_vars["time_data_on2"],
            gold_on2_vars["latitude_on2"],
            gold_on2_vars["longitude_on2"])
        
        on2 = gold_on2_vars["on2"]
        start_times_on2, end_times_on2 = get_scan_pair_times(valid_on2_results, on2.shape[0])
        gold_on2_vars["valid_on2_results"] = valid_on2_results
        gold_times = [p["time"] for p in valid_on2_results]
        time_to_file_map = find_closest_filesv2(gold_times, directory)

        # Process GITM for each threshold and store separately
        gitm_results_per_thresh = []
        for thresh in num_den_thresh:
            cache = {}
            gitm_result = gitm_on2_processing_v2(valid_on2_results, time_to_file_map, cache, thresh, start_times_on2, end_times_on2)
            gitm_results_per_thresh.append(gitm_result)

        n_scans = len(gitm_results_per_thresh[0]["mesh_data_on2"])
        total_mesh = []
        total_scatter = []
        total_thresh = []
        total_xwind_data = []
        total_ywind_data = []
        total_xwind_loc = []
        total_ywind_loc = []

        for scan_idx in range(n_scans):
            for thresh_idx, thresh_result in enumerate(gitm_results_per_thresh):
                total_mesh.append(thresh_result["mesh_data_on2"][scan_idx])
                total_scatter.append(thresh_result["scatter_data_on2"][scan_idx])
                total_thresh.append(thresh_result["thresh"])
                xw = thresh_result.get("xwind_data")
                yw = thresh_result.get("ywind_data")
                xl = thresh_result.get("xwind_loc")
                yl = thresh_result.get("ywind_loc")

                if (
                    xw is None or yw is None or xl is None or yl is None
                    or scan_idx >= len(xw)
                    or scan_idx >= len(yw)
                    or scan_idx >= len(xl)
                    or scan_idx >= len(yl)):
                    continue

                total_xwind_data.append(xw[scan_idx])
                total_ywind_data.append(yw[scan_idx])
                total_xwind_loc.append(xl[scan_idx])
                total_ywind_loc.append(yl[scan_idx])

        pair_size = 2
        n_scans_gold = gold_on2_vars["n_scans"]
        n_pairs = n_scans_gold // pair_size
        n_thresh = len(num_den_thresh)
        total_scans = n_scans_gold * n_thresh 

        gold_vars_repeated = {"n_scans": total_scans}

        for k, v in gold_on2_vars.items():
            if isinstance(v, np.ndarray):
                if v.shape[0] == n_scans_gold:
                    # Repeat scan-dependent arrays like on2
                    v_pairs = v.reshape(n_pairs, pair_size, *v.shape[1:])
                    v_pairs_repeated = np.repeat(v_pairs[:, np.newaxis, ...], n_thresh, axis=1)
                    gold_vars_repeated[k] = v_pairs_repeated.reshape(total_scans, *v.shape[1:])
                else:
                    gold_vars_repeated[k] = v
            else:
                gold_vars_repeated[k] = v

        start_times_on2_expanded = list(np.repeat(start_times_on2, 3))
        end_times_on2_expanded   = list(np.repeat(end_times_on2, 3))

        start_times_on2_repeated = list(np.tile(start_times_on2_expanded, n_thresh))
        end_times_on2_repeated   = list(np.tile(end_times_on2_expanded, n_thresh))
        gold_vars_repeated["n_scans"] = total_scans

        total_gitm_results = {
            "mesh_data_on2": total_mesh,
            "scatter_data_on2": total_scatter,
            "thresh": total_thresh,
            "xwind_data": total_xwind_data,
            "ywind_data": total_ywind_data,
            "xwind_loc": total_xwind_loc,
            "ywind_loc": total_ywind_loc}

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        on2_outputs = on2_data_comparison(total_gitm_results, gold_vars_repeated, start_times_on2_repeated, end_times_on2_repeated, 1, output_dir)