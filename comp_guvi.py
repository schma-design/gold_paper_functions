import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd
from guvi_read import read_guvi_sav_file
from scipy.optimize import linear_sum_assignment
from comp_gold_funcs import extract_gold_on2, extract_valid_on2_points

def extract_date(filename):
    match = re.search(r"(\d{4})_(\d{3})", filename)
    if match:
        year = int(match.group(1))
        doy = int(match.group(2))
        return (year, doy)
    else:
        return (0, 0)

def plot_gold_guvi_scan_pairs(gold_points, guvi_lat, guvi_lon, guvi_on2, guvi_time):

    guvi_lon_360 = guvi_lon.ravel()
    guvi_lon_flat = np.where(guvi_lon_360 > 180, guvi_lon_360 - 360, guvi_lon_360)
    guvi_on2_flat = guvi_on2.ravel()
    guvi_time_flat = guvi_time.ravel().astype('datetime64[s]')
    
    nscans = max(p["i"] for p in gold_points) + 1
    
    for scan_index in range(0, nscans - 1, 2):
        print(f"\n--- Processing Scan Pair {scan_index} & {scan_index + 1} ---")
        
        scan_points = [p for p in gold_points if p["i"] in (scan_index, scan_index + 1)]
        if not scan_points:
            print("No GOLD points found for this pair, skipping...")
            continue
        
        times = [p["time"] for p in scan_points]
        min_time = min(times)
        max_time = max(times)
        
        window_start = min_time - timedelta(minutes=15)
        window_end   = max_time + timedelta(minutes=15)
        guvi_mask_day = np.array([np.datetime64(t, 'D') == np.datetime64(min_time.date()) for t in guvi_time_flat])
        guvi_mask_time = (guvi_time_flat >= np.datetime64(window_start)) & (guvi_time_flat <= np.datetime64(window_end))
        guvi_mask = guvi_mask_day & guvi_mask_time
        guvi_lats_hour = guvi_lat.ravel()[guvi_mask]
        guvi_lons_hour = guvi_lon_flat[guvi_mask]
        guvi_on2_hour = guvi_on2_flat[guvi_mask]
        scan0_points = [p for p in scan_points if p["i"] == scan_index]
        scan1_points = [p for p in scan_points if p["i"] == scan_index + 1]
        
        # Plotting
        fig, ax = plt.subplots(figsize=(8,6))
        vmin, vmax = 0.1, 1.8
        
        sc0 = ax.scatter([p["lon"] for p in scan0_points],
                        [p["lat"] for p in scan0_points],
                        c=[p["on2"] for p in scan0_points],
                        cmap="coolwarm", marker="o", vmin=vmin, vmax=vmax)
        
        sc1 = ax.scatter([p["lon"] for p in scan1_points],
                        [p["lat"] for p in scan1_points],
                        c=[p["on2"] for p in scan1_points],
                        cmap="coolwarm", marker="o", label = f"GOLD Points", vmin=vmin, vmax=vmax)
        
        if len(guvi_on2_hour) > 0:
            ax.scatter(guvi_lons_hour,
                    guvi_lats_hour,
                    c=guvi_on2_hour,
                    cmap="coolwarm",
                    marker="^", vmin = vmin, vmax = vmax,
                    label=f"GUVI points")
        
        ax.set_xlabel("Longitude", fontsize=14)
        ax.set_ylabel("Latitude", fontsize=14)
        ax.set_title(f"{window_start.strftime('%Y-%m-%d %H:%M')} to {window_end.strftime('%H:%M')}", fontsize=14)
        ax.legend(fontsize=14)
        cbar = fig.colorbar(sc0, ax=ax, orientation='vertical')
        cbar.set_label(r"O/N$_{2}$", fontsize=14)
        cbar.ax.tick_params(labelsize=14) 
        ax.tick_params(axis='both', labelsize=14)
        plt.tight_layout()
        fname = f"gold_guvi_{pd.to_datetime(min_time).date()}_scanpair_hour_{scan_index}_{scan_index+1}.png"
        plt.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"Saved figure: {fname}")

def bijective_parity(gold_files, guvi_files):

    max_dist_deg = 2
    scanid1 = [2, 0, 14]
    scanid2 = [3, 1, 15]
    print("The current scan selection is tailored to quiet days before 3 specific geomagnetic events. The procedure" \
        " for selecting scans from the terminal has not yet been completed.")

    all_data = []
    custom_colors = [
    "#00008b",  # blue
    "#d62728",  # red
    "#17becf"  # cyan
    ]

    for i, file in enumerate(gold_files):
        print(f"Processing file {i+1}")

        gold_vars = extract_gold_on2(file)
        time_data = gold_vars["time_data_on2"]
        latitude_gold = gold_vars["latitude_on2"]
        longitude_gold = gold_vars["longitude_on2"]
        on2_gold = gold_vars["on2"]

        guvi_data = read_guvi_sav_file(guvi_files[i])
        guvi_time = guvi_data["times"]
        guvi_on2 = guvi_data["on2"]
        guvi_lat = guvi_data["lats"]
        guvi_lon = guvi_data["lons"]
        gold_points = extract_valid_on2_points(on2_gold, time_data, latitude_gold, longitude_gold)

        guvi_lat_flat = guvi_lat.ravel()
        guvi_lon_360 = guvi_lon.ravel()
        guvi_lon_flat = np.where(guvi_lon_360 > 180, guvi_lon_360 - 360, guvi_lon_360)
        guvi_on2_flat = guvi_on2.ravel()
        guvi_time_flat = guvi_time.ravel()

        scan_id1 = scanid1[i]
        scan_id2 = scanid2[i]

        scan_points = [p for p in gold_points if p["i"] in (scan_id1, scan_id2)]
        if not scan_points:
            print("No GOLD points for scans")
            return
        
        gold_lats = np.array([p["lat"] for p in scan_points])
        gold_lons = np.array([p["lon"] for p in scan_points])
        gold_on2s = np.array([p["on2"] for p in scan_points])
        gold_times = np.array([p["time"] for p in scan_points])

        times = gold_times.tolist()
        window_start = min(times) - timedelta(minutes=15)
        window_end   = max(times) + timedelta(minutes=15)

        guvi_mask = (guvi_time_flat >= window_start) & (guvi_time_flat <= window_end)
        #print(f"GUVI time range (masked): {guvi_time_flat[guvi_mask].min()} → {guvi_time_flat[guvi_mask].max()}")
        guvi_lats = guvi_lat_flat[guvi_mask]
        guvi_lons = guvi_lon_flat[guvi_mask]
        guvi_on2s = guvi_on2_flat[guvi_mask]

        print(f"GUVI points in window: {len(guvi_on2s)}")

        if len(guvi_on2s) == 0 or len(gold_lats) == 0:
            print("No points to match.")
            return

        dlat = guvi_lats[:, None] - gold_lats[None, :]
        dlon = guvi_lons[:, None] - gold_lons[None, :]
        dist = np.sqrt(dlat**2 + dlon**2)
        guvi_idx, gold_idx = linear_sum_assignment(dist)

        matched_gold = []
        matched_guvi = []

        for gi, gj in zip(guvi_idx, gold_idx):
            if np.isnan(gold_on2s[gj]):
                continue
            if dist[gi, gj] <= max_dist_deg:
                matched_gold.append(gold_on2s[gj])
                matched_guvi.append(guvi_on2s[gi])

        print(f"Matched {len(matched_gold)} bijective pairs (within {max_dist_deg}°)")
        if matched_gold:
            label = min(gold_times).strftime("%Y-%m-%d")
            all_data.append((matched_gold, matched_guvi, label))

    if not all_data:
        print("No matched data across all days")
        return

    all_gold = np.concatenate([np.array(g) for g, _, _ in all_data])
    all_guvi = np.concatenate([np.array(g) for _, g, _ in all_data])
    offset_const = np.mean(all_gold - all_guvi)
    print(f"Opt Offset: {offset_const:.2f}")

    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(8,8))

    for (matched_gold, matched_guvi, label), color in zip(all_data, custom_colors):
        plt.scatter(matched_gold, matched_guvi, color=color, alpha = 0.5, label=label)
        plt.scatter(matched_gold, np.array(matched_guvi) + offset_const, marker = '^', color=color, label=f"{label} + {offset_const:.2f}")

    lims = [min(all_gold.min(), all_guvi.min()), max(all_gold.max(), all_guvi.max())]
    plt.plot(lims, lims, 'k--')
    plt.xlabel(r"GOLD O/N$_{2}$")
    plt.ylabel(r"GUVI O/N$_{2}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gold_guvi_all_parity.png", dpi=300)
    plt.close()