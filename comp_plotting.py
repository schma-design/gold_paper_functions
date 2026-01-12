from datetime import datetime
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import numpy as np

def plot_on2_results(csv_path, date_str, start_times, res_strings):
    # Load CSV
    plt.rcParams.update({'font.size': 14})
    df = pd.read_csv(csv_path)

    match = re.search(r'(\d{6})', csv_path)
    if match:
        date_str = match.group(1)
        plot_date = datetime.strptime(date_str, '%y%m%d').strftime('%B %d, %Y')
    else:
        plot_date = "Unknown Date"

    # Compute absolute values of ON2 Mean Differences
    df["Abs Mean Difference"] = df["ON2 Mean Difference"].abs()

    # Group by threshold and compute mean over scan sets
    grouped = df.groupby("Num Density Threshold * 1e21")["Abs Mean Difference"].mean().reset_index()

    if 'mad' in res_strings:
        plt.figure(figsize=(10, 6))
        plt.plot(grouped["Num Density Threshold * 1e21"], grouped["Abs Mean Difference"], marker="o", linestyle='-')
        #plt.title(f"Absolute Mean ON2 Difference Across all Scans for {plot_date}")
        plt.xlabel(r"Column Density Threshold (×1e21) /m$^2$")
        plt.ylabel(r"Mean Absolute O/N$_2$ Difference")
        plt.tight_layout()
        plt.savefig(f"on2_abs_mean_dif_{date_str}.png", dpi=300)
        plt.close()

    # Get all unique threshold values
    thresholds = df["Num Density Threshold * 1e21"].unique()

    if 'cor' in res_strings:
        plt.figure(figsize=(10, 6))

        # Loop over each threshold and plot the scan values
        for thresh in thresholds:
            subset = df[df["Num Density Threshold * 1e21"] == thresh]
            plt.plot(
                subset["Scan Set #"],
                subset["ON2 Cross Correlation"],
                marker='o',
                label=(f"{thresh:.1f}" r"× 1e21 /m$^2$"))

        time_labels = [dt.strftime('%H:%M:%S') for dt in start_times]
        scan_indices = list(range(1, len(time_labels) + 1))

        #plt.title(f"ON2 Cross Correlation Over Time on {plot_date}")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
        plt.ylabel("Cross Correlation")
        plt.legend(title=r"Column Density Threshold")
        plt.tight_layout()
        plt.savefig(f"on2_cross_corr_{date_str}.png", dpi=300)
        plt.close()

    if 'eot' in res_strings:
        plt.figure(figsize=(10,6))
        for thresh in thresholds:
            subset = df[df["Num Density Threshold * 1e21"] == thresh]
            error = subset["ON2 Mean Difference"]
            scans = subset["Scan Set #"]
            plt.plot(scans, error, marker="o", label=f"{thresh:.1f}" r"× 1e21 /m$^2$")

        #plt.title(f"ON2 Error over Time on {plot_date}")
        plt.ylabel(r"O/N$_2$ Error")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
        plt.legend(title="Column Density Threshold")
        plt.tight_layout()
        plt.savefig(f"on2_error_over_time_{date_str}.png", dpi=300)
        plt.close()

    if 'abseot' in res_strings:
        plt.figure(figsize=(10,6))
        for thresh in thresholds:
            subset = df[df["Num Density Threshold * 1e21"] == thresh]
            error = np.abs(subset["ON2 Mean Difference"])
            scans = subset["Scan Set #"]
            plt.plot(scans, error, marker="o", label=f"Column Density Threshold: {thresh:.1f} × 1e21 /m$^2$")

        #plt.title(f"ON2 Absolute Error over Time on {plot_date}")
        plt.ylabel("ON2 Error")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"on2_abs_error_over_time_{date_str}.png", dpi=300)
        plt.close()

    if 'rmsot' in res_strings:
        plt.figure(figsize=(10,6))
        for thresh in thresholds:
            subset = df[df["Num Density Threshold * 1e21"] == thresh]
            rms = subset["ON2 RMS"]
            scans = subset["Scan Set #"]
            plt.plot(scans, rms, marker="o", label=f"{thresh:.1f}" r"× 1e21 /m$^2$")

        #plt.title(f"ON2 RMS over Time on {plot_date}")
        plt.ylabel(r"O/N$_2$ RMS")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
        plt.legend(title="Column Density Threshold")
        plt.tight_layout()
        plt.savefig(f"on2_rms_over_time_{date_str}.png", dpi=300)
        plt.close()

def plot_combined_on2(csv_paths, plot_types, thresholds, output_dir="."):

    if isinstance(plot_types, str):
        plot_types = [pt.strip().lower() for pt in plot_types.split(",")]

    plt.rcParams.update({"font.size": 14})
    
    #Load all CSVs
    all_data = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if "Time" in df.columns:
            df["scan_time"] = pd.to_datetime(df["Time"], errors="coerce")
        else:
            basename = os.path.basename(path)
            try:
                date_part = basename.split("_")[-1].replace(".csv", "")
                file_date = datetime.strptime(date_part, "%y%m%d")
            except Exception:
                file_date = datetime.fromtimestamp(os.path.getmtime(path))
            df["scan_time"] = file_date
        df = df.dropna(subset=["scan_time"])
        all_data.append(df)

    if not all_data:
        print("No data loaded from csv_paths.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    if not combined_df.empty:
        first_date = combined_df["scan_time"].min()
        date_str = first_date.strftime("%b%Y").lower()
    else:
        date_str = "unknown"

    if "mad" in plot_types:
        plt.figure(figsize=(10, 6))
        for date, df_date in combined_df.groupby(combined_df["scan_time"].dt.date):
            tmp = df_date.copy()
            tmp["Abs Mean Difference"] = tmp["ON2 Mean Difference"].abs()
            grouped = tmp.groupby("Num Density Threshold * 1e21")["Abs Mean Difference"].mean().reset_index()
            plt.plot(
                grouped["Num Density Threshold * 1e21"],
                grouped["Abs Mean Difference"],
                marker="o",
                label=str(date))
        plt.xlabel(r"Column Density Threshold (×1e21) /m$^2$")
        plt.ylabel(r"Mean Absolute O/N$_2$ Difference")
        plt.legend(title="Date")
        plt.tight_layout()
        fig_name = f"on2_combined_abs_mean_dif_{date_str}.png"
        plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
        plt.close()
        print(f"Saved figure: {fig_name}")

    if "arms" in plot_types:
        plt.figure(figsize=(10,6))
        for date, df_date in combined_df.groupby(combined_df["scan_time"].dt.date):
            df_copy = df_date.copy()
            grouped = df_copy.groupby("Num Density Threshold * 1e21")["ON2 RMS"].mean().reset_index()
            plt.plot(
                grouped["Num Density Threshold * 1e21"],
                grouped["ON2 RMS"],
                marker='o',
                label=str(date))
        plt.xlabel(r"Column Density Threshold (×1e21) /m$^2$")
        plt.ylabel(r"Mean O/N$_2$ RMS")
        plt.legend(title="Date")
        plt.tight_layout()
        fig_name = f"on2_total_combined_arms_{date_str}.png"
        plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
        plt.close()
        print(f"Saved figure: {fig_name}")

    if "acor" in plot_types:
        plt.figure(figsize=(10,6))
        for date, df_date in combined_df.groupby(combined_df["scan_time"].dt.date):
            df_copy = df_date.copy()
            grouped = df_copy.groupby("Num Density Threshold * 1e21")["ON2 Cross Correlation"].mean().reset_index()
            plt.plot(
                grouped["Num Density Threshold * 1e21"],
                grouped["ON2 Cross Correlation"],
                marker = 'o',
                label=str(date))
        plt.xlabel(r"Column Density Threshold (×1e21) /m$^2$")
        plt.ylabel(r"Mean O/N$_2$ Cross Correlation")
        plt.legend(title="Date")
        plt.tight_layout()
        fig_name = f"on2_total_combined_acor_{date_str}.png"
        plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
        plt.close()
        print(f"Saved figure: {fig_name}")

    # Time-series plots: 
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    time_series_types = ["cor", "rmsot", "abseot", "eot"]

    for plot_type in plot_types:
        if plot_type not in time_series_types:
            continue
        if thresholds is None:
            raise ValueError(f"Thresholds must be provided for {plot_type} plot.")

        plt.figure(figsize=(12, 6))
        for i, thresh in enumerate(thresholds):
            df_thresh = combined_df[np.isclose(combined_df["Num Density Threshold * 1e21"], thresh, atol=1e-3)].copy()
            if df_thresh.empty:
                continue
            df_thresh = df_thresh.sort_values("scan_time")

            if plot_type == "cor":
                yvals = df_thresh["ON2 Cross Correlation"]
                ylabel = r"O/N$_2$ Cross Correlation"
                #title = "ON2 Cross Correlation Across Storm"
            elif plot_type == "rmsot":
                yvals = df_thresh["ON2 RMS"]
                ylabel = r"O/N$_2$ RMS"
                #title = "ON2 RMS Over Time Across Thresholds"
            elif plot_type == "abseot":
                yvals = np.abs(df_thresh["ON2 Mean Difference"])
                ylabel = r"Absolute O/N$_2$ Error"
                #title = "ON2 Abs Error Over Time Across Thresholds"
            elif plot_type == "eot":
                yvals = df_thresh["ON2 Mean Difference"]
                ylabel = r"O/N$_2$ Error"
                #title = "ON2 Error Over Time Across Thresholds"

            plt.plot(
                df_thresh["scan_time"],
                yvals,
                marker="o",
                linestyle="-",
                #label=f"{legend_label_prefix}: {thresh:.1f}",
                label=f"{thresh:.1f}" r"× 1e21 /m$^2$",
                color=color_cycle[i % len(color_cycle)]
                )

        plt.xlabel("Date and Time")
        plt.ylabel(ylabel)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        plt.gca().tick_params(axis='x', which='major', pad=18) 
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        plt.legend(title = "Column Density Threshold")
        plt.tight_layout()
        fig_name = f"on2_combined_{plot_type}_{date_str}.png"
        plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
        plt.close()
        print(f"Saved figure: {fig_name}")

####################### PLOTTING TDISK DATA FROM METHODS 0 AND 1 #############################
def plot_rawtdisk_results(csv_path, date_str, start_times, res_strings):
    # Load CSV
    plt.rcParams.update({'font.size': 14})
    df = pd.read_csv(csv_path)

    match = re.search(r'(\d{6})', csv_path)
    if match:
        date_str = match.group(1)
        plot_date = datetime.strptime(date_str, '%y%m%d').strftime('%B %d, %Y')
    else:
        plot_date = "Unknown Date"
    
    # Compute absolute values of ON2 Mean Differences
    df["Abs Mean Difference"] = df["TDISK Mean Difference"].abs()

    # Group by threshold and compute mean over 6 scan sets
    grouped = df.groupby("Altitude")["Abs Mean Difference"].mean().reset_index()

    if 'mad' in res_strings:
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(grouped["Altitude"], grouped["Abs Mean Difference"], marker="o", linestyle='-')
        #plt.title(f"Absolute Mean Temperature Difference Across all Scans for {plot_date}")
        plt.xlabel("Altitude (km)")
        plt.ylabel("Mean Absolute TDISK Difference (K)")
        plt.tight_layout()
        plt.savefig(f"tdisk_abs_mean_dif_{date_str}.png")
        plt.close()

    # Get all unique threshold values
    altitudes = df["Altitude"].unique()

    time_labels = [dt.strftime('%H:%M:%S') for dt in start_times]
    scan_indices = list(range(1, len(time_labels) + 1))

    if 'cor' in res_strings:
        plt.figure(figsize=(10, 6))
        # Loop over each threshold and plot the scan values
        for altitude in altitudes:
            subset = df[df["Altitude"] == altitude]
            plt.plot(
                subset["Scan Set #"],
                subset["TDISK Cross Correlation"],
                marker='o',
                label=f"{altitude:.1f} km")

        #plt.title(f"TDISK Cross Correlation over Time for Each Altitude on {plot_date}")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
        plt.ylabel("Cross Correlation")
        plt.legend(title="Altitude")
        plt.tight_layout()
        plt.savefig(f"tdisk_cross_corr_{date_str}.png", dpi=300)
        plt.close()

    if 'eot' in res_strings:
        plt.figure(figsize=(10,6))
        for altitude in altitudes:
            subset = df[df["Altitude"] == altitude]
            error = subset["TDISK Mean Difference"]
            scans = subset["Scan Set #"]
            plt.plot(scans, error, marker="o", label=f"{altitude:.1f} km")

        #plt.title(f"TDISK Error over Time on {plot_date}")
        plt.ylabel("TDISK Error (K)")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
        plt.legend(title="Altitude")
        plt.tight_layout()
        plt.savefig(f"tdisk_error_over_time_{date_str}.png", dpi=300)
        plt.close()

    if 'abseot' in res_strings:
        plt.figure(figsize=(10,6))
        for altitude in altitudes:
            subset = df[df["Altitude"] == altitude]
            abs = np.abs(subset["TDISK Mean Difference"])
            scans = subset["Scan Set #"]
            plt.plot(scans, abs, marker="o", label=f"{altitude:.1f} km")

        plt.title(f"TDISK Absolute Error over Time on {plot_date}")
        plt.ylabel("TDISK Error (K)")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=45)
        plt.legend(title="Altitude")
        plt.tight_layout()
        plt.savefig(f"tdisk_abs_error_over_time_{date_str}.png", dpi=300)
        plt.close()

    if 'rmsot' in res_strings:
        plt.figure(figsize=(10,6))
        for altitude in altitudes:
            subset = df[df["Altitude"] == altitude]
            rms = subset["TDISK RMS"]
            scans = subset["Scan Set #"]
            plt.plot(scans, rms, marker="o", label=f"{altitude:.1f} km")

        #plt.title(f"TDISK RMS over Time on {plot_date}")
        plt.ylabel("RMS (K)")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
        plt.legend(title="Altitude")
        plt.tight_layout()
        plt.savefig(f"tdisk_rms_over_time_{date_str}.png", dpi=300)
        plt.close()

def plot_fptdisk_results(csv_path, date_str, start_times, res_strings):
    # Load CSV
    plt.rcParams.update({'font.size': 14})
    df = pd.read_csv(csv_path)

    match = re.search(r'(\d{6})', csv_path)
    if match:
        date_str = match.group(1)
        plot_date = datetime.strptime(date_str, '%y%m%d').strftime('%B %d, %Y')
    else:
        plot_date = "Unknown Date"

    # Compute absolute values of ON2 Mean Differences
    df["Abs Mean Difference"] = df["TDISK Mean Difference"].abs()

    # Group by threshold and compute mean over 6 scan sets
    grouped = df.groupby("Altitude")["Abs Mean Difference"].mean().reset_index()

    if 'mad' in res_strings:
        plt.figure(figsize=(10, 6))
        plt.plot(grouped["Altitude"], grouped["Abs Mean Difference"], marker="o", linestyle='-')
        #plt.title(f"Absolute Mean Temperature Difference Across all Scans for {plot_date}")
        plt.xlabel("Altitude (km)")
        plt.ylabel("Mean Absolute Temperature Difference (K)")
        plt.tight_layout()
        plt.savefig(f"tdisk_fp_abs_mean_dif_{date_str}.png", dpi=300)
        plt.close()

    # Get all unique threshold values
    altitudes = df["Altitude"].unique()

    time_labels = [dt.strftime('%H:%M:%S') for dt in start_times]
    scan_indices = list(range(1, len(time_labels) + 1))

    if 'cor' in res_strings:
        plt.figure(figsize=(10, 6))

        # Loop over each threshold and plot the 6 scan values
        for altitude in altitudes:
            subset = df[df["Altitude"] == altitude]
            plt.plot(
                subset["Scan Set #"],
                subset["TDISK Cross Correlation"],
                marker='o',
                label=f"{altitude:.1f} km"
            )

        #plt.title(f"TDISK Cross Correlation Across Scans for Each \nFixed Peak Altitude on {plot_date}")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
        plt.ylabel("Cross Correlation")
        plt.legend(title="Altitude")
        plt.tight_layout()
        plt.savefig(f"tdisk_fp_cross_corr_{date_str}.png", dpi=300)
        plt.close()

    if 'eot' in res_strings:
        plt.figure(figsize=(10,6))
        for altitude in altitudes:
            subset = df[df["Altitude"] == altitude]
            error = subset["TDISK Mean Difference"]
            scans = subset["Scan Set #"]
            plt.plot(scans, error, marker="o", label=f"{altitude:.1f} km")

       #plt.title("TDISK Fixed Peak Error as a Function of Time")
        plt.ylabel("Temperature Error (K)")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
        plt.legend(title="Altitude")
        plt.tight_layout()
        plt.savefig(f"tdisk_fp_error_over_time_{date_str}.png", dpi=300)
        plt.close()

    if 'abseot' in res_strings:
        plt.figure(figsize=(10,6))
        for altitude in altitudes:
            subset = df[df["Altitude"] == altitude]
            abs = np.abs(subset["TDISK Mean Difference"])
            scans = subset["Scan Set #"]
            plt.plot(scans, abs, marker="o", label=f"Altitude: {altitude:.0f} km")

        #plt.title(f"TDISK Fixed Peak Absolute Error over Time on {plot_date}")
        plt.ylabel("TDISK Error (K)")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"tdisk_fp_abs_error_over_time_{date_str}.png", dpi=300)
        plt.close()

    if 'rmsot' in res_strings:
        plt.figure(figsize=(10,6))
        for altitude in altitudes:
            subset = df[df["Altitude"] == altitude]
            rms = subset["TDISK RMS"]
            scans = subset["Scan Set #"]
            plt.plot(scans, rms, marker="o", label=f"{altitude:.1f} km")

        #plt.title(f"TDISK Fixed Peak RMS over Time on {plot_date}")
        plt.ylabel("Temperature RMS")
        plt.xlabel("Time")
        plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
        plt.legend(title="Altitude")
        plt.tight_layout()
        plt.savefig(f"tdisk_fp_rms_over_time_{date_str}.png", dpi=300)
        plt.close()

def plot_combined_tdisk(csv_paths, altitudes, method, plot_types, output_dir="."):
    """
    Plots multiple days from storms on one plot
    """
    plt.rcParams.update({"font.size": 14})

    # normalize plot_types input and map aliases
    if isinstance(plot_types, str):
        plot_types = [plot_types]
    plot_types = [pt.lower() for pt in plot_types]

    all_data = []
    for path in csv_paths:
        df = pd.read_csv(path)

        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        elif "scan_time" in df.columns:
            df["Time"] = pd.to_datetime(df["scan_time"], errors="coerce")
        else:
            basename = os.path.basename(path)
            try:
                date_part = basename.split("_")[-1].replace(".csv", "")
                file_date = datetime.strptime(date_part, "%y%m%d")
            except Exception:
                file_date = datetime.fromtimestamp(os.path.getmtime(path))
            df["Time"] = file_date

        df = df.dropna(subset=["Time"])
        all_data.append(df)

    if not all_data:
        print("No data loaded from csv_paths.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    if not combined_df.empty:
        first_date = combined_df["Time"].min()
        date_str = first_date.strftime("%b%Y").lower()
    else:
        date_str = "unknown"

    # ------------------------
    # MAD: mean absolute diff vs Altitude (per date)
    # ------------------------
    if "mad" in plot_types:
        plt.figure(figsize=(10, 6))
        # group by calendar date of Time, replicate original behavior
        for date, df_date in combined_df.groupby(combined_df["Time"].dt.date):
            tmp = df_date.copy()
            tmp["Abs Mean Difference"] = tmp["TDISK Mean Difference"].abs()
            grouped = tmp.groupby("Altitude")["Abs Mean Difference"].mean().reset_index()
            # original plotted Altitude vs grouped mean for each date
            plt.plot(grouped["Altitude"], grouped["Abs Mean Difference"], marker="o", label=str(date))

        plt.xlabel("Altitude (km)")
        plt.ylabel("Mean Absolute Temperature Difference (K)")
        plt.legend(title="Date")
        '''
        plt.axvline(x=195, color='red', linestyle='--', linewidth=1.5)
        plt.text(
            195,
            plt.ylim()[1] * 0.5,
            "Quiet Days",
            rotation=90,
            color='black',
            verticalalignment='center',
            horizontalalignment='right')
        '''
        plt.tight_layout()

        fig_name = f"tdisk_combined_abs_mean_dif_{method}_{date_str}.png"
        plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
        plt.close()
        print(f"Saved figure: {fig_name}")

    # ------------------------
    # Time-series plots (cross_corr, rms, abs_error, error)
    # ------------------------
    # Prepare color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # helper to filter by altitude tolerance
    def _filter_by_altitude(df, alt):
        return df[np.isclose(df["Altitude"], alt, atol=0.5)].copy()
    
    if ("arms" in plot_types) or ("acor" in plot_types):
        print("At least one of the inputted plot types currently has no plotting procedure.")

    if any(pt in ("cor", "rmsot", "abseot", "eot") for pt in plot_types):
        # CROSS CORRELATION
        if "cor" in plot_types:
            plt.figure(figsize=(12, 6))
            for i, altitude in enumerate(altitudes):
                df_alt = _filter_by_altitude(combined_df, altitude)
                if df_alt.empty:
                    continue
                df_alt = df_alt.sort_values("Time")
                plt.plot(
                    df_alt["Time"],
                    df_alt["TDISK Cross Correlation"],
                    marker="o",
                    linestyle="-",
                    label=f"Altitude: {altitude:.1f} km",
                    color=color_cycle[i % len(color_cycle)],
                )
            plt.xlabel("Date and Time")
            plt.ylabel("Temperature Cross Correlation")
            plt.title("Temperature Cross Correlation Across Altitudes")
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            fig_name = f"tdisk_combined_cross_corr_{method}_{date_str}.png"
            plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
            plt.close()
            print(f"Saved figure: {fig_name}")

        # RMS
        if "rmsot" in plot_types:
            plt.figure(figsize=(12, 6))
            for i, altitude in enumerate(altitudes):
                df_alt = _filter_by_altitude(combined_df, altitude)
                if df_alt.empty:
                    continue
                df_alt = df_alt.sort_values("Time")
                plt.plot(
                    df_alt["Time"],
                    df_alt["TDISK RMS"],
                    marker="o",
                    linestyle="-",
                    label=f"Altitude: {altitude:.1f} km",
                    color=color_cycle[i % len(color_cycle)],
                )
            plt.xlabel("Date and Time")
            plt.ylabel("Temperature RMS (K)")
            plt.title("TDISK RMS Over Time Across Altitudes")
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            fig_name = f"tdisk_combined_rms_{method}_{date_str}.png"
            plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
            plt.close()
            print(f"Saved figure: {fig_name}")

        # ABS ERROR (abs of mean difference)
        if "abseot" in plot_types:
            plt.figure(figsize=(12, 6))
            for i, altitude in enumerate(altitudes):
                df_alt = _filter_by_altitude(combined_df, altitude)
                if df_alt.empty:
                    continue
                df_alt = df_alt.sort_values("Time")
                plt.plot(
                    df_alt["Time"],
                    np.abs(df_alt["TDISK Mean Difference"]),
                    marker="o",
                    linestyle="-",
                    label=f"Altitude: {altitude:.1f} km",
                    color=color_cycle[i % len(color_cycle)],
                )
            plt.xlabel("Date and Time")
            plt.ylabel("Absolute Temp Error (K)")
            plt.title("TDISK Abs Error Over Time Across Altitudes")
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            fig_name = f"tdisk_combined_abserror_{method}_{date_str}.png"
            plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
            plt.close()
            print(f"Saved figure: {fig_name}")

        # ERROR (signed mean difference)
        if "eot" in plot_types:
            plt.figure(figsize=(12, 6))
            for i, altitude in enumerate(altitudes):
                df_alt = _filter_by_altitude(combined_df, altitude)
                if df_alt.empty:
                    continue
                df_alt = df_alt.sort_values("Time")
                plt.plot(
                    df_alt["Time"],
                    df_alt["TDISK Mean Difference"],
                    marker="o",
                    linestyle="-",
                    label=f"Altitude: {altitude:.1f} km",
                    color=color_cycle[i % len(color_cycle)],
                )
            plt.xlabel("Date and Time")
            plt.ylabel("Temp Error (K)")
            plt.title("TDISK Error Over Time Across Altitudes")
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            fig_name = f"tdisk_combined_error_{method}_{date_str}.png"
            plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
            plt.close()
            print(f"Saved figure: {fig_name}")

############################### PLOTTING TDISK DATA RELIANT ON N2 THRESHOLD #############################
def plot_globaln2_tdisk_results(csv_path, date_str, start_times, res_strings):
    # Load CSV
    plt.rcParams.update({'font.size': 14})
    df = pd.read_csv(csv_path)

    match = re.search(r'(\d{6})', csv_path)
    if match:
        date_str = match.group(1)
        plot_date = datetime.strptime(date_str, '%y%m%d').strftime('%B %d, %Y')
    else:
        plot_date = "Unknown Date"
    
    # Compute absolute values of ON2 Mean Differences
    df["Abs Mean Difference"] = df["TDISK Mean Difference"].abs()

    thresholds = df["Num Density Threshhold * 1e21"].unique()
    time_labels = [dt.strftime('%H:%M:%S') for dt in start_times]
    scan_indices = list(range(1, len(time_labels) + 1))

    for thresh in thresholds:
        df_thresh = df[df["Num Density Threshhold * 1e21"] == thresh]
        altitudes = df_thresh["Altitude"].unique()

        # === Plot 1: Mean Abs Diff vs Altitude (grouped by Offset) ===
        #grouped = df_thresh.groupby(["Alt Offset", "Altitude"])["Abs Mean Difference"].mean().reset_index()
        offset_groups = df_thresh.groupby("Alt Offset", sort=False)
        offsets = df_thresh["Alt Offset"].unique()

        if 'mad' in res_strings:
            plt.figure(figsize=(10, 6))

            # Group by offset, average across *all altitudes and scans*
            mean_diff_by_offset = df_thresh.groupby("Alt Offset")["Abs Mean Difference"].mean()

            # Plot offsets on x-axis
            plt.plot(
                mean_diff_by_offset.index,
                mean_diff_by_offset.values,
                marker="o",
                linestyle="-"
            )

            plt.xlabel("Altitude Offset (km)")
            plt.ylabel("Mean Absolute Temperature Difference (K)")
            #plt.title(f"Mean Abs Temp Diff vs Offset\nThreshold {thresh} ×1e21 on {plot_date}")
            #plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"tdisk_globaln2_abs_mean_diff_vs_offset_{thresh}_{date_str}.png", dpi=300)
            plt.close()

        if 'cor' in res_strings:
            # === Plot 2: Cross Correlation vs Scan ===
            plt.figure(figsize=(10, 6))
            for offset in offsets:
                subset = df_thresh[df_thresh["Alt Offset"] == offset]
                plt.plot(
                    subset["Scan Set #"],
                    subset["TDISK Cross Correlation"],
                    marker='o',
                    label=f"{offset} km"
                )
            #plt.title(f"TDISK Cross Correlation for Global N2 Threshold {thresh} ×1e21 on {plot_date}")
            plt.xlabel("Time")
            plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
            plt.ylabel("Cross Correlation")
            plt.legend(title="Altitude Offset")
            plt.tight_layout()
            plt.savefig(f"tdisk_globaln2_cross_corr_{thresh}_{date_str}.png", dpi=300)
            plt.close()

        if 'eot' in res_strings:
            # === Plot 3: Mean Diff vs Time ===
            plt.figure(figsize=(10,6))
            for offset in offsets:
                subset = df_thresh[df_thresh["Alt Offset"] == offset]
                error = subset["TDISK Mean Difference"]
                scans = subset["Scan Set #"]
                plt.plot(scans, error, marker="o", label=f"{offset} km")
            #plt.title(f"TDISK Error vs Time for Global N2 Threshold {thresh} ×1e21 on {plot_date}")
            plt.ylabel("Temperature Error (K)")
            plt.xlabel("Time")
            plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
            plt.legend(title="Altitude Offset")
            plt.tight_layout()
            plt.savefig(f"tdisk_globaln2_error_over_time_{thresh}_{date_str}.png", dpi=300)
            plt.close()

        if 'abseot' in res_strings:
            plt.figure(figsize=(10,6))
            for offset in offsets:
                subset = df_thresh[df_thresh["Alt Offset"] == offset]
                abs = np.abs(subset["TDISK Mean Difference"])
                scans = subset["Scan Set #"]
                plt.plot(scans, abs, marker="o", label=f"{offset} km")

            #plt.title(f"TDISK Global N2 Absolute Error over Time on {plot_date}")
            plt.ylabel("Temperature Error (K)")
            plt.xlabel("Time")
            plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
            plt.legend(title="Altitude Offset")
            plt.tight_layout()
            plt.savefig(f"tdisk_globaln2_abs_error_over_time_{date_str}.png", dpi=300)
            plt.close()

        if 'rmsot' in res_strings:
            plt.figure(figsize=(10,6))
            for offset in offsets:
                subset = df_thresh[df_thresh["Alt Offset"] == offset]
                rms = subset["TDISK RMS"]
                scans = subset["Scan Set #"]
                plt.plot(scans, rms, marker="o", label=f"{offset} km")

            #plt.title(f"TDISK Global N2 RMS over Time on {plot_date}")
            plt.ylabel("Temperature RMS")
            plt.xlabel("Time")
            plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
            plt.legend(title="Altitude Offset")
            plt.tight_layout()
            plt.savefig(f"tdisk_globaln2_rms_over_time_{date_str}.png", dpi=300)
            plt.close()

def plot_n2pixeltdisk_results(csv_path, date_str, start_times, res_strings):
    # Load CSV
    df = pd.read_csv(csv_path)
    plt.rcParams.update({'font.size': 14})

    match = re.search(r'(\d{6})', csv_path)
    if match:
        date_str = match.group(1)
        plot_date = datetime.strptime(date_str, '%y%m%d').strftime('%B %d, %Y')
    else:
        plot_date = "Unknown Date"

    # Compute absolute values of ON2 Mean Differences
    df["Abs Mean Difference"] = df["TDISK Mean Difference"].abs()
    thresholds = df["Num Density Threshhold * 1e21"].unique()
    time_labels = [dt.strftime('%H:%M:%S') for dt in start_times]
    scan_indices = list(range(1, len(time_labels) + 1))

    for thresh in thresholds:
        df_thresh = df[df["Num Density Threshhold * 1e21"] == thresh]

        # === Plot 1: Mean Abs Diff vs Altitude (grouped by Offset) ===
        offset_means = df_thresh.groupby("Alt Offset", sort=False)["Abs Mean Difference"].mean().reset_index()
        offset_means = offset_means.sort_values("Alt Offset")
        offsets = df_thresh["Alt Offset"].unique()

        if 'mad' in res_strings:
            plt.figure(figsize=(10, 6))
            plt.plot(offset_means["Alt Offset"], offset_means["Abs Mean Difference"], marker="o", linestyle='-')
            #plt.title(f"Abs Mean Temp Diff for N2 Threshold by Pixel vs Altitude Offset\n at Threshold {thresh} ×1e21 on {plot_date}")
            plt.xlabel("Offset Altitude (km)")
            plt.ylabel("Mean Absolute Temperature Difference (K)")
            #plt.legend(title="Alt Offset (km)")
            plt.tight_layout()
            plt.savefig(f"tdisk_n2pixel_abs_mean_diff_{thresh}_{date_str}.png", dpi=300)
            plt.close()

        if 'cor' in res_strings:
            # === Plot 2: Cross Correlation vs Scan ===
            plt.figure(figsize=(10, 6))
            for offset in offsets:
                subset = df_thresh[df_thresh["Alt Offset"] == offset]
                plt.plot(
                    subset["Scan Set #"],
                    subset["TDISK Cross Correlation"],
                    marker='o',
                    label=f"{offset:.0f} km")
                
            #plt.title(f"TDISK Cross Correlation for N2 Pixel Weighting at \n N2 Threshold {thresh} ×1e21 on {plot_date}")
            plt.xlabel("Time")
            plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
            plt.ylabel("Cross Correlation")
            plt.legend(title="Altitude Offset")
            plt.tight_layout()
            plt.savefig(f"tdisk_n2pixel_cross_corr_{thresh}_{date_str}.png", dpi=300)
            plt.close()

        if 'eot' in res_strings:
            # === Plot 3: Mean Diff vs Time ===
            plt.figure(figsize=(10,6))
            for offset in offsets:
                subset = df_thresh[df_thresh["Alt Offset"] == offset]
                error = subset["TDISK Mean Difference"]
                scans = subset["Scan Set #"]
                plt.plot(scans, error, marker="o", label=f"{offset:.0f} km")

            #plt.title(f"TDISK Error vs Time for Peak by Pixel for \n N2 Threshold {thresh} ×1e21 on {plot_date}")
            plt.ylabel("Temperature Error (K)")
            plt.xlabel("Time")
            plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
            plt.legend(title="Altitude Offset")
            plt.tight_layout()
            plt.savefig(f"tdisk_n2pixel_error_over_time_{thresh}_{date_str}.png", dpi=300)
            plt.close()

        if 'abseot' in res_strings:
            plt.figure(figsize=(10,6))
            for offset in offsets:
                subset = df_thresh[df_thresh["Alt Offset"] == offset]
                abs = np.abs(subset["TDISK Mean Difference"])
                scans = subset["Scan Set #"]
                plt.plot(scans, abs, marker="o", label=f"{offset:.0f} km")

            #plt.title(f"TDISK N2 Pixel Absolute Error over Time on {plot_date}")
            plt.ylabel("Temperature Error (K)")
            plt.xlabel("Time")
            plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
            plt.legend(title="Altitude Offset")
            plt.tight_layout()
            plt.savefig(f"tdisk_n2pixel_abs_error_over_time_{date_str}.png", dpi=300)
            plt.close()

        if 'rmsot' in res_strings:
            plt.figure(figsize=(10,6))
            for offset in offsets:
                subset = df_thresh[df_thresh["Alt Offset"] == offset]
                rms = subset["TDISK RMS"]
                scans = subset["Scan Set #"]
                plt.plot(scans, rms, marker="o", label=f"{offset:.0f} km")

            #plt.title(f"TDISK N2 Pixel RMS over Time on {plot_date}")
            plt.ylabel("Temperature RMS (K)")
            plt.xlabel("Time")
            plt.xticks(ticks=scan_indices, labels=time_labels, rotation=0)
            plt.legend(title="Altitude Offset")
            plt.tight_layout()
            plt.savefig(f"tdisk_n2pixel_rms_over_time_{date_str}.png", dpi=300)
            plt.close()

def plot_weighted_cf(csv_paths, res_strings, output_dir):
    """
    Plot mean absolute difference and mean RMS difference vs altitude offset
    for multiple CSV files (multiple dates) on the same plot.
    """

    plt.rcParams.update({"font.size": 14})

    # normalize inputs
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    if isinstance(res_strings, str):
        res_strings = [res_strings]
    res_strings = [r.lower() for r in res_strings]

    all_data = []

    # -------------------------
    # Load CSVs + assign dates
    # -------------------------
    for path in csv_paths:
        df = pd.read_csv(path)

        # Try extracting date from filename (YYMMDD)
        match = re.search(r"(\d{6})", os.path.basename(path))
        if match:
            file_date = datetime.strptime(match.group(1), "%y%m%d").date()
        else:
            # fallback: file modification time
            file_date = datetime.fromtimestamp(os.path.getmtime(path)).date()

        df["Date"] = file_date
        all_data.append(df)

    if not all_data:
        print("No CSV data loaded.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # absolute difference column
    combined_df["TDISK Abs Difference"] = combined_df["TDISK Mean Difference"].abs()

    # month-year tag for filenames
    date_tag = min(combined_df["Date"]).strftime("%Y%m")

    # -------------------------
    # MAD vs Offset
    # -------------------------
    if "mad" in res_strings:
        plt.figure(figsize=(10, 6))

        for date, df_date in combined_df.groupby("Date"):
            grouped = (
                df_date
                .groupby("Alt Offset", sort=True)["TDISK Abs Difference"]
                .mean()
                .reset_index()
            )

            plt.plot(
                grouped["Alt Offset"],
                grouped["TDISK Abs Difference"],
                marker="o",
                linestyle="-",
                label=date.strftime("%Y-%m-%d"),
            )

        plt.xlabel("Offset Altitude (km)")
        plt.ylabel("Mean Absolute Temperature Difference (K)")
        plt.legend(title="Date")
        plt.tight_layout()

        fname = f"tdisk_weightedcf_mad_vs_offset_{date_tag}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()
        print(f"Saved: {fname}")

    # -------------------------
    # RMS vs Offset
    # -------------------------
    if "arms" in res_strings:
        plt.figure(figsize=(10, 6))

        for date, df_date in combined_df.groupby("Date"):
            grouped = (
                df_date
                .groupby("Alt Offset", sort=True)["TDISK RMS"]
                .mean()
                .reset_index()
            )

            plt.plot(
                grouped["Alt Offset"],
                grouped["TDISK RMS"],
                marker="o",
                linestyle="-",
                label=date.strftime("%Y-%m-%d"),
            )

        plt.xlabel("Offset Altitude (km)")
        plt.ylabel("Mean RMS Temperature Difference (K)")
        plt.legend(title="Date")
        plt.tight_layout()

        fname = f"tdisk_weightedcf_rms_vs_offset_{date_tag}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()
        print(f"Saved: {fname}")

def plot_method_g(csv_paths, res_strings, output_dir):
    """
    Plot mean absolute difference and mean RMS difference vs altitude offset
    for multiple CSV files (multiple dates) on the same plot.
    """

    plt.rcParams.update({"font.size": 14})

    # normalize inputs
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    if isinstance(res_strings, str):
        res_strings = [res_strings]
    res_strings = [r.lower() for r in res_strings]

    all_data = []

    # -------------------------
    # Load CSVs + assign dates
    # -------------------------
    for path in csv_paths:
        df = pd.read_csv(path)

        # Try extracting date from filename (YYMMDD)
        match = re.search(r"(\d{6})", os.path.basename(path))
        if match:
            file_date = datetime.strptime(match.group(1), "%y%m%d").date()
        else:
            # fallback: file modification time
            file_date = datetime.fromtimestamp(os.path.getmtime(path)).date()

        df["Date"] = file_date
        all_data.append(df)

    if not all_data:
        print("No CSV data loaded.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # absolute difference column
    combined_df["TDISK Abs Difference"] = combined_df["TDISK Mean Difference"].abs()

    # month-year tag for filenames
    date_tag = min(combined_df["Date"]).strftime("%Y%m")

    # -------------------------
    # MAD vs Offset
    # -------------------------
    if "mad" in res_strings:
        plt.figure(figsize=(10, 6))

        for date, df_date in combined_df.groupby("Date"):
            grouped = (
                df_date
                .groupby("Alt Offset", sort=True)["TDISK Abs Difference"]
                .mean()
                .reset_index()
            )

            plt.plot(
                grouped["Alt Offset"],
                grouped["TDISK Abs Difference"],
                marker="o",
                linestyle="-",
                label=date.strftime("%Y-%m-%d"),
            )

        plt.xlabel("Offset Altitude (km)")
        plt.ylabel("Mean Absolute Temperature Difference (K)")
        plt.legend(title="Date")
        plt.tight_layout()

        fname = f"tdisk_method_G_mad_vs_offset_{date_tag}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()
        print(f"Saved: {fname}")

    # -------------------------
    # RMS vs Offset
    # -------------------------
    if "arms" in res_strings:
        plt.figure(figsize=(10, 6))

        for date, df_date in combined_df.groupby("Date"):
            grouped = (df_date.groupby("Alt Offset", sort=True)["TDISK RMS"].mean().reset_index())

            plt.plot(
                grouped["Alt Offset"],
                grouped["TDISK RMS"],
                marker="o",
                linestyle="-",
                label=date.strftime("%Y-%m-%d"),
            )

        plt.xlabel("Offset Altitude (km)")
        plt.ylabel("Mean RMS Temperature Difference (K)")
        plt.legend(title="Date")
        plt.tight_layout()

        fname = f"tdisk_method_G_rms_vs_offset_{date_tag}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()
        print(f"Saved: {fname}")

def plot_tdisk_by_thresh(csv_paths, num_den_thresh, method, plot_types, output_dir="."):
    """
    Generalized plotting for TDISK metrics by threshold and offset.
    """

    if isinstance(plot_types, str):
        plot_types = [plot_types]
    plot_types = [pt.lower() for pt in plot_types]

    plt.rcParams.update({"font.size": 14})
    all_data = []
    for path in csv_paths:
        df = pd.read_csv(path)

        if "Time" in df.columns:
            df["scan_time"] = pd.to_datetime(df["Time"], errors="coerce")
        else:
            basename = os.path.basename(path)
            try:
                date_part = basename.split("_")[-1].replace(".csv", "")
                file_date = datetime.strptime(date_part, "%y%m%d")
            except Exception:
                file_date = datetime.fromtimestamp(os.path.getmtime(path))
            df["scan_time"] = file_date

        df = df.dropna(subset=["scan_time"])
        all_data.append(df)

    if not all_data:
        print("No data loaded.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Loop through requested plot types
    for plot_type in plot_types:
        for thresh in num_den_thresh:
            df_thresh = combined_df[
                abs(combined_df["Num Density Threshhold * 1e21"] - thresh) < 1e-6
            ].copy()

            if df_thresh.empty:
                print(f"No data for threshold {thresh}")
                continue

            # === Special case: abs_mean_diff_by_offset ("mad") ===
            if plot_type == "mad":
                plt.figure(figsize=(10, 6))

                for date, df_date in df_thresh.groupby(df_thresh["scan_time"].dt.date):
                    tmp = df_date.copy()
                    tmp["Abs Mean Difference"] = tmp["TDISK Mean Difference"].abs()
                    grouped = (tmp.groupby("Alt Offset")["Abs Mean Difference"].mean().reset_index().sort_values("Alt Offset"))
                    plt.plot(grouped["Alt Offset"], grouped["Abs Mean Difference"], marker="o", label=str(date))

                plt.xlabel("Altitude Offset (km)")
                plt.ylabel("Mean Absolute Temperature Difference (K)")
                plt.legend(title="Date")
                plt.tight_layout()
                date_str = df_thresh["scan_time"].min().strftime("%b%Y").lower()
                fig_name = (f"tdisk_combined_abs_mean_dif_{method}_thresh_{thresh:.1f}_{date_str}.png")
                plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
                plt.close()
                print(f"Saved figure: {fig_name}")
                continue

            # === All other plot types ===
            plt.figure(figsize=(12, 6))
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            offset_vals = sorted(df_thresh["Alt Offset"].unique())
            for i, offset in enumerate(offset_vals):
                df_offset = df_thresh[df_thresh["Alt Offset"] == offset].sort_values("scan_time")

                if plot_type == "cor":
                    yvals = df_offset["TDISK Cross Correlation"]
                    ylabel = "TDISK Cross Correlation"
                elif plot_type == "rmsot":
                    yvals = df_offset["TDISK RMS"]
                    ylabel = "TDISK RMS (K)"
                elif plot_type == "abseot":
                    yvals = np.abs(df_offset["TDISK Mean Difference"])
                    ylabel = "Abs Temp Error (K)"
                elif plot_type == "eot":
                    yvals = df_offset["TDISK Mean Difference"]
                    ylabel = "Temp Error (K)"
                else:
                    raise ValueError(f"Unknown plot_type: {plot_type}")

                plt.plot(
                    df_offset["scan_time"],
                    yvals,
                    marker="o",
                    linestyle="-",
                    label=f"Offset: {offset} km",
                    color=color_cycle[i % len(color_cycle)],
                )

            # Labels & title
            plt.xlabel("Date and Time")
            plt.ylabel(ylabel)

            title_map = {
                "cor": "Cross Correlation Over Time",
                "rmsot": "RMS Over Time",
                "abseot": "Abs Temp Error Over Time",
                "eot": "Temp Error Over Time",
            }
            method_prefix = "Global N₂" if method == 2 else "N₂ Pixel"
            plt.title(f"{method_prefix} {title_map[plot_type]}\nThreshold: {thresh:.1f} × 1e21")

            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            plt.grid(True)
            plt.legend(title="Alt Offset", loc="best")
            plt.tight_layout()

            # Filename
            date_str = df_thresh["scan_time"].min().strftime("%b%Y").lower()
            fig_name = f"tdisk_{plot_type}_{method}_thresh_{thresh:.1f}_{date_str}.png"
            plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
            plt.close()
            print(f"Saved figure: {fig_name}")

def compare_tdisk_metrics(
    csv_paths, labels, start_times, output_filename,
    globaln2_path=None, n2pixel_path=None,
    num_thresh=None, offset=None):

    time_labels = [dt.strftime('%H:%M') for dt in start_times]
    scan_indices = list(range(1, len(time_labels) + 1))

    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    titles = ["TDISK Error over Time", "TDISK Cross Correlation over Time", "TDISK RMS over Time"]
    ylabels = ["TDISK Error (K)", "Cross Correlation", "TDISK RMS"]

    # --- Plot raw/fixed peak data ---
    for csv_path, label in zip(csv_paths, labels):
        df = pd.read_csv(csv_path)
        altitudes = sorted(df["Altitude"].unique())
        for altitude in altitudes:
            subset = df[df["Altitude"] == altitude]
            scans = subset["Scan Set #"]

            axs[0].plot(scans, subset["TDISK Mean Difference"], marker='o', label=f"{label} @ {altitude:.0f} km")
            axs[1].plot(scans, subset["TDISK Cross Correlation"], marker='o', label=f"{label} @ {altitude:.0f} km")
            axs[2].plot(scans, subset["TDISK RMS"], marker='o', label=f"{label} @ {altitude:.0f} km")

    # --- Optional: Plot Global N2 data ---
    if globaln2_path and num_thresh is not None and offset is not None:
        df_n2 = pd.read_csv(globaln2_path)
        df_filt = df_n2[
            (df_n2["Num Density Threshhold * 1e21"] == num_thresh) &
            (df_n2["Alt Offset"] == offset)
        ]
        scans = df_filt["Scan Set #"]
        axs[0].plot(scans, df_filt["TDISK Mean Difference"], marker='x', linestyle='--', label=f"Global N2 @ offset {offset} km")
        axs[1].plot(scans, df_filt["TDISK Cross Correlation"], marker='x', linestyle='--', label=f"Global N2 @ offset {offset} km")
        axs[2].plot(scans, df_filt["TDISK RMS"], marker='x', linestyle='--', label=f"Global N2 @ offset {offset} km")

    # --- Optional: Plot N2 Pixel data ---
    if n2pixel_path and num_thresh is not None and offset is not None:
        df_pix = pd.read_csv(n2pixel_path)
        df_filt = df_pix[
            (df_pix["Num Density Threshhold * 1e21"] == num_thresh) &
            (df_pix["Alt Offset"] == offset)
        ]
        scans = df_filt["Scan Set #"]
        axs[0].plot(scans, df_filt["TDISK Mean Difference"], marker='s', linestyle=':', label=f"N2 Pixel @ offset {offset} km")
        axs[1].plot(scans, df_filt["TDISK Cross Correlation"], marker='s', linestyle=':', label=f"N2 Pixel @ offset {offset} km")
        axs[2].plot(scans, df_filt["TDISK RMS"], marker='s', linestyle=':', label=f"N2 Pixel @ offset {offset} km")

    # --- Final formatting ---
    for ax, title, ylabel in zip(axs, titles, ylabels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    axs[2].set_xlabel("Time")
    axs[2].set_xticks(scan_indices)
    axs[2].set_xticklabels(time_labels, rotation=45)

    plt.suptitle("TDISK Comparison across Datasets", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    plt.close()