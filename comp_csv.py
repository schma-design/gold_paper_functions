import pandas as pd
from datetime import datetime
import os
from comp_on2_funcs import gitm_on2_processing, on2_data_comparison, gitm_on2_processing_v2
from comp_temp_funcs import gitm_temp_data_processing, gitm_tdisk_processing_v2, temp_data_comparison

def basic_on2_comparison_csv(gold_on2_vars, file_paths_on2, num_den_thresh, visualization, fixed_var, output_dir, start_times_on2, end_times_on2):
    on2_rows = []
    cache = {}
    for thresh in num_den_thresh:
        adjusted_time = []
        for start, end in zip(start_times_on2, end_times_on2):
            if start is not None and end is not None:
                midpoint = start + (end - start) / 2
            else:
                midpoint = None
            adjusted_time.append(midpoint)

        if fixed_var == 0:
        # GITM ON2 processing
            gitm_on2_outputs = gitm_on2_processing(file_paths_on2, thresh, 1, cache)

            # ON2 data comparison
            on2_outputs = on2_data_comparison(gitm_on2_outputs, gold_on2_vars, start_times_on2, end_times_on2, visualization, output_dir)
        else:
            cache = {}
            valid_on2_results = gold_on2_vars["valid_on2_results"]
            gitm_on2_results2 = gitm_on2_processing_v2(valid_on2_results, file_paths_on2, cache, thresh, start_times_on2, end_times_on2)
            on2_outputs = on2_data_comparison(gitm_on2_results2, gold_on2_vars, start_times_on2, end_times_on2, visualization, output_dir)

        mean_dif = on2_outputs["mean_diffs"]
        mean_percent_dif = on2_outputs["mean_percent_diffs"]
        rms = on2_outputs["rms_vals"]
        mean_percent_rms_vals = on2_outputs["mean_percent_rms_vals"]
        corr_scores = on2_outputs["corr_scores"]
        nplots = on2_outputs["nscans"]

        for i in range(nplots):
            on2_row = {
                "Num Density Threshold * 1e21": thresh,
                "Scan Set #": i + 1,
                "Time": adjusted_time[i],
                "ON2 Mean Difference": mean_dif[i],
                "ON2 Mean Percent Difference": mean_percent_dif[i],
                "ON2 RMS": rms[i],
                "ON2 Mean Percent RMS": mean_percent_rms_vals[i],
                "ON2 Cross Correlation": corr_scores[i]
            }
            on2_rows.append(on2_row)

    #Generate filename from first start time
    if start_times_on2 and isinstance(start_times_on2[0], datetime):
        date_str = start_times_on2[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')

    os.makedirs(output_dir, exist_ok=True)

    csv_filename = f"on2_rawdata_comparison_{date_str}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    pd.DataFrame(on2_rows).to_csv(csv_path, index=False)
    return csv_path, date_str

def basic_tdisk_comparison_csv(gold_tdisk_vars, file_paths_tdisk, alt_list, visualization, fixed_var, output_dir, start_times_tdisk, end_times_tdisk):

    tdisk_rows = []
    cache = {}
    for altitude in alt_list:
        adjusted_time = []
        for start, end in zip(start_times_tdisk, end_times_tdisk):
            if start is not None and end is not None:
                midpoint = start + (end - start) / 2
            else:
                midpoint = None
            adjusted_time.append(midpoint)
        
        if fixed_var == 0:
            gitm_tdisk_outputs = gitm_temp_data_processing(file_paths_tdisk, 0, altitude, None, cache, None)
        else:
            valid_tdisk_results = gold_tdisk_vars["valid_tdisk_results"]
            gitm_tdisk_outputs = gitm_tdisk_processing_v2(valid_tdisk_results, file_paths_tdisk, 0, altitude, None, cache, start_times_tdisk, end_times_tdisk, None)
            
        true_alt = gitm_tdisk_outputs["true_alt"]
        tdisk_results = temp_data_comparison(gitm_tdisk_outputs, gold_tdisk_vars, start_times_tdisk, end_times_tdisk, visualization, output_dir)

        nplots = tdisk_results["nscans"]
        mean_dif_t = tdisk_results["mean_diffs"]
        mean_percent_dif_t = tdisk_results["mean_percent_diffs"]
        rms_t = tdisk_results["rms_vals"]
        mean_percent_rms_vals_t = tdisk_results["mean_percent_rms_vals"]
        corr_scores = tdisk_results["corr_scores"]

        for i in range(nplots):
            tdisk_row = {
                "Altitude": true_alt,
                "Scan Set #": i + 1,
                "Time": adjusted_time[i],
                "TDISK Mean Difference": mean_dif_t[i],
                "TDISK Mean Percent Difference": mean_percent_dif_t[i],
                "TDISK RMS": rms_t[i],
                "TDISK Mean Percent RMS": mean_percent_rms_vals_t[i],
                "TDISK Cross Correlation": corr_scores[i]
            }
            tdisk_rows.append(tdisk_row)

    #Generate filename from first start time
    if start_times_tdisk and isinstance(start_times_tdisk[0], datetime):
        date_str = start_times_tdisk[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d') 

    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"tdisk_rawdata_comparison_{date_str}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    pd.DataFrame(tdisk_rows).to_csv(csv_path, index=False)
    return csv_path, date_str

def fixed_peak_tdisk_comparison(gold_tdisk_vars, file_paths_tdisk, alt_list, visualization, fixed_var, output_dir, start_times_tdisk, end_times_tdisk):
    tdisk_rows = []
    cache = {}
    for altitude in alt_list:
        adjusted_time = []

        for start, end in zip(start_times_tdisk, end_times_tdisk):
            if start is not None and end is not None:
                midpoint = start + (end - start) / 2
            else:
                midpoint = None
            adjusted_time.append(midpoint)

        if fixed_var == 0:
            gitm_tdisk_outputs = gitm_temp_data_processing(file_paths_tdisk, 1, altitude, None, cache, None)
        else:
            valid_tdisk_results = gold_tdisk_vars["valid_tdisk_results"]
            gitm_tdisk_outputs = gitm_tdisk_processing_v2(valid_tdisk_results, file_paths_tdisk, 1, altitude, None, cache, start_times_tdisk, end_times_tdisk, None)

        tdisk_results = temp_data_comparison(gitm_tdisk_outputs, gold_tdisk_vars, start_times_tdisk, end_times_tdisk, visualization, output_dir)
        
        mean_dif_t = tdisk_results["mean_diffs"]
        mean_percent_dif_t = tdisk_results["mean_percent_diffs"]
        rms_t = tdisk_results["rms_vals"]
        mean_percent_rms_vals_t = tdisk_results["mean_percent_rms_vals"]
        corr_scores = tdisk_results["corr_scores"]
        nplots = tdisk_results["nscans"]

        for i in range(nplots):
            tdisk_row = {
                "Altitude": altitude,
                "Scan Set #": i + 1,
                "Time": adjusted_time[i],
                "TDISK Mean Difference": mean_dif_t[i],
                "TDISK Mean Percent Difference": mean_percent_dif_t[i],
                "TDISK RMS": rms_t[i],
                "TDISK Mean Percent RMS": mean_percent_rms_vals_t[i],
                "TDISK Cross Correlation": corr_scores[i]
            }
            tdisk_rows.append(tdisk_row)

            #Generate filename from first start time
    if start_times_tdisk and isinstance(start_times_tdisk[0], datetime):
        date_str = start_times_tdisk[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')

    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"tdisk_fixedpeak_comparison_{date_str}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    pd.DataFrame(tdisk_rows).to_csv(csv_path, index=False)
    return csv_path, date_str

def globaln2_temp_comparison(gold_tdisk_vars, file_paths, offset_alt, num_den_thresh, visualization, fixed_var, output_dir, start_times_tdisk, end_times_tdisk):

    rows = []
    cache = {}

    for thresh in num_den_thresh:      
        adjusted_time = []
        for start, end in zip(start_times_tdisk, end_times_tdisk):
            if start is not None and end is not None:
                midpoint = start + (end - start) / 2
            else:
                midpoint = None
            adjusted_time.append(midpoint)

        for offset in offset_alt:

            if fixed_var == 0:
                gitm_on2_outputs = gitm_on2_processing(file_paths, thresh, 0, cache)
                gitm_tdisk_outputs = gitm_temp_data_processing(file_paths, 2, offset, gitm_on2_outputs, cache, None)
            else: 
                valid_tdisk_results = gold_tdisk_vars["valid_tdisk_results"]
                gitm_tdisk_outputs = gitm_tdisk_processing_v2(valid_tdisk_results, file_paths, 2, offset, thresh, cache, start_times_tdisk, end_times_tdisk, None)

            altitude = gitm_tdisk_outputs["scan_alts"]
            tdisk_results = temp_data_comparison(gitm_tdisk_outputs, gold_tdisk_vars, start_times_tdisk, end_times_tdisk, visualization, output_dir)

            mean_dif_tdisk = tdisk_results["mean_diffs"]
            mean_percent_dif_tdisk = tdisk_results["mean_percent_diffs"]
            rms_tdisk = tdisk_results["rms_vals"]
            mean_percent_rms_vals_tdisk = tdisk_results["mean_percent_rms_vals"]
            corr_scores = tdisk_results["corr_scores"]
            nplots = tdisk_results["nscans"]

            for i in range(nplots):
                row = {
                    "Num Density Threshhold * 1e21" : thresh,
                    "Scan Set #": i + 1,
                    "Time": adjusted_time[i],
                    "Altitude": altitude[i],
                    "Alt Offset": offset,
                    "TDISK Mean Difference": mean_dif_tdisk[i],
                    "TDISK Mean Percent Difference": mean_percent_dif_tdisk[i],
                    "TDISK RMS": rms_tdisk[i],
                    "TDISK Mean Percent RMS": mean_percent_rms_vals_tdisk[i],
                    "TDISK Cross Correlation": corr_scores[i]
                }
                rows.append(row)

    #Generate filename from first start time
    if start_times_tdisk and isinstance(start_times_tdisk[0], datetime):
        date_str = start_times_tdisk[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')

    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"tdisk_globaln2_comparison_{date_str}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, date_str

def temp_by_n2pixel_comparison(gold_tdisk_vars, file_paths, offset_alt, num_den_thresh, visualization, fixed_var, output_dir, start_times_tdisk, end_times_tdisk):

    rows = []
    cache = {}

    for thresh in num_den_thresh:
        for offset in offset_alt:
            adjusted_time = []
            for start, end in zip(start_times_tdisk, end_times_tdisk):
                if start is not None and end is not None:
                    midpoint = start + (end - start) / 2
                else:
                    midpoint = None
                adjusted_time.append(midpoint)

            if fixed_var == 0:
                gitm_on2_outputs = gitm_on2_processing(file_paths, thresh, 0, cache)
                gitm_tdisk_outputs = gitm_temp_data_processing(file_paths, 3, offset, gitm_on2_outputs, cache, None)
            elif fixed_var == 1:
                valid_tdisk_results = gold_tdisk_vars["valid_tdisk_results"]
                gitm_tdisk_outputs = gitm_tdisk_processing_v2(valid_tdisk_results, file_paths, 3, offset, thresh, cache, start_times_tdisk, end_times_tdisk, None)

            tdisk_results = temp_data_comparison(gitm_tdisk_outputs, gold_tdisk_vars, start_times_tdisk, end_times_tdisk, visualization, output_dir)
            nplots = tdisk_results["nscans"]
            mean_dif_tdisk = tdisk_results["mean_diffs"]
            mean_percent_dif_tdisk = tdisk_results["mean_percent_diffs"]
            rms_tdisk = tdisk_results["rms_vals"]
            mean_percent_rms_vals_tdisk = tdisk_results["mean_percent_rms_vals"]
            corr_scores = tdisk_results["corr_scores"]

            for i in range(nplots):
                row = {
                    "Num Density Threshhold * 1e21" : thresh,
                    "Scan Set #": i + 1,
                    "Time": adjusted_time[i],
                    "Alt Offset": offset,
                    "TDISK Mean Difference": mean_dif_tdisk[i],
                    "TDISK Mean Percent Difference": mean_percent_dif_tdisk[i],
                    "TDISK RMS": rms_tdisk[i],
                    "TDISK Mean Percent RMS": mean_percent_rms_vals_tdisk[i],
                    "TDISK Cross Correlation": corr_scores[i]
                }
                rows.append(row)

            # Generate filename from first start time
    if start_times_tdisk and isinstance(start_times_tdisk[0], datetime):
        date_str = start_times_tdisk[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')

    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"tdisk_n2_pixel_comparison_{date_str}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, date_str

def sza_assignment_csv(gold_tdisk_vars, file_paths_tdisk, visualization, fixed_var, output_dir, start_times_tdisk, end_times_tdisk):

    tdisk_rows = []
    cache = {}

    adjusted_time = []
    for start, end in zip(start_times_tdisk, end_times_tdisk):
        if start is not None and end is not None:
            midpoint = start + (end - start) / 2
        else:
            midpoint = None
        adjusted_time.append(midpoint)
    
    if fixed_var == 0:
        print("This method was intended for use with the pbp selection for -time_method")
        return None, None
    else:
        valid_tdisk_results = gold_tdisk_vars["valid_tdisk_results"]
        gitm_tdisk_outputs = gitm_tdisk_processing_v2(valid_tdisk_results, file_paths_tdisk, 5, None, None, cache, start_times_tdisk, end_times_tdisk, None)
        
    true_alt = gitm_tdisk_outputs["true_alt"]
    tdisk_results = temp_data_comparison(gitm_tdisk_outputs, gold_tdisk_vars, start_times_tdisk, end_times_tdisk, visualization, output_dir)

    nplots = tdisk_results["nscans"]
    mean_dif_t = tdisk_results["mean_diffs"]
    mean_percent_dif_t = tdisk_results["mean_percent_diffs"]
    rms_t = tdisk_results["rms_vals"]
    mean_percent_rms_vals_t = tdisk_results["mean_percent_rms_vals"]
    corr_scores = tdisk_results["corr_scores"]

    for i in range(nplots):
        tdisk_row = {
            "Altitude": true_alt,
            "Scan Set #": i + 1,
            "Time": adjusted_time[i],
            "TDISK Mean Difference": mean_dif_t[i],
            "TDISK Mean Percent Difference": mean_percent_dif_t[i],
            "TDISK RMS": rms_t[i],
            "TDISK Mean Percent RMS": mean_percent_rms_vals_t[i],
            "TDISK Cross Correlation": corr_scores[i]
        }
        tdisk_rows.append(tdisk_row)

    #mGenerate filename from first start time
    if start_times_tdisk and isinstance(start_times_tdisk[0], datetime):
        date_str = start_times_tdisk[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')

    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"tdisk_cf_assignment_comparison_{date_str}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    pd.DataFrame(tdisk_rows).to_csv(csv_path, index=False)
    return csv_path, date_str

def weighted_cf_csv(gold_tdisk_vars, file_paths, offset_alt, visualization, fixed_var, output_dir, start_times_tdisk, end_times_tdisk, cf_data):

    rows = []
    cache = {}

    for offset in offset_alt:
        adjusted_time = []
        for start, end in zip(start_times_tdisk, end_times_tdisk):
            if start is not None and end is not None:
                midpoint = start + (end - start) / 2
            else:
                midpoint = None
            adjusted_time.append(midpoint)

        if fixed_var == 0:
            print("This method was intended for use with the pbp selection for -time_method")
            return None, None
        else: 
            valid_tdisk_results = gold_tdisk_vars["valid_tdisk_results"]
            gitm_tdisk_outputs = gitm_tdisk_processing_v2(valid_tdisk_results, file_paths, 6, offset, None, cache, start_times_tdisk, end_times_tdisk, cf_data)

        altitude = gitm_tdisk_outputs["scan_alts"]
        tdisk_results = temp_data_comparison(gitm_tdisk_outputs, gold_tdisk_vars, start_times_tdisk, end_times_tdisk, visualization, output_dir)

        mean_dif_tdisk = tdisk_results["mean_diffs"]
        mean_percent_dif_tdisk = tdisk_results["mean_percent_diffs"]
        rms_tdisk = tdisk_results["rms_vals"]
        mean_percent_rms_vals_tdisk = tdisk_results["mean_percent_rms_vals"]
        corr_scores = tdisk_results["corr_scores"]
        nplots = tdisk_results["nscans"]

        for i in range(nplots):
            row = {
                "Scan Set #": i + 1,
                "Time": adjusted_time[i],
                "Estimated Altitude": altitude[i],
                "Alt Offset": offset,
                "TDISK Mean Difference": mean_dif_tdisk[i],
                "TDISK Mean Percent Difference": mean_percent_dif_tdisk[i],
                "TDISK RMS": rms_tdisk[i],
                "TDISK Mean Percent RMS": mean_percent_rms_vals_tdisk[i],
                "TDISK Cross Correlation": corr_scores[i]
            }
            rows.append(row)

            # Generate filename from first start time
    if start_times_tdisk and isinstance(start_times_tdisk[0], datetime):
        date_str = start_times_tdisk[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')

    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"tdisk_weighted_cf_{date_str}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, date_str

def sza_peak_tdisk_comparison(gold_tdisk_vars, file_paths_tdisk, offsets_list, visualization, fixed_var, output_dir, start_times_tdisk, end_times_tdisk):
    tdisk_rows = []
    cache = {}
    for offset in offsets_list:
        adjusted_time = []
        for start, end in zip(start_times_tdisk, end_times_tdisk):
            if start is not None and end is not None:
                midpoint = start + (end - start) / 2
            else:
                midpoint = None
            adjusted_time.append(midpoint)

        if fixed_var == 0:
            print("This method was intended for use with the pbp selection for -time_method")
            return None, None
        else:
            valid_tdisk_results = gold_tdisk_vars["valid_tdisk_results"]
            gitm_tdisk_outputs = gitm_tdisk_processing_v2(valid_tdisk_results, file_paths_tdisk, 7, offset, None, cache, start_times_tdisk, end_times_tdisk, None)

        scan_alt = gitm_tdisk_outputs["scan_alts"]
        tdisk_results = temp_data_comparison(gitm_tdisk_outputs, gold_tdisk_vars, start_times_tdisk, end_times_tdisk, visualization, output_dir)
        
        mean_dif_t = tdisk_results["mean_diffs"]
        mean_percent_dif_t = tdisk_results["mean_percent_diffs"]
        rms_t = tdisk_results["rms_vals"]
        mean_percent_rms_vals_t = tdisk_results["mean_percent_rms_vals"]
        corr_scores = tdisk_results["corr_scores"]
        nplots = tdisk_results["nscans"]

        for i in range(nplots):
            tdisk_row = {
                "Approximated Altitude": scan_alt[i],
                "Alt Offset": offset,
                "Scan Set #": i + 1,
                "Time": adjusted_time[i],
                "TDISK Mean Difference": mean_dif_t[i],
                "TDISK Mean Percent Difference": mean_percent_dif_t[i],
                "TDISK RMS": rms_t[i],
                "TDISK Mean Percent RMS": mean_percent_rms_vals_t[i],
                "TDISK Cross Correlation": corr_scores[i]
            }
            tdisk_rows.append(tdisk_row)

    # Generate filename from first start time
    if start_times_tdisk and isinstance(start_times_tdisk[0], datetime):
        date_str = start_times_tdisk[0].strftime('%y%m%d')
    else:
        date_str = datetime.now().strftime('%y%m%d')

    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"tdisk_sza_peak_comparison_{date_str}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    pd.DataFrame(tdisk_rows).to_csv(csv_path, index=False)
    return csv_path, date_str