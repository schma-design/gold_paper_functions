import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from thermo_plot_tmpv0 import get_args
from netCDF4 import Dataset
import argparse
from guvi_read import read_guvi_sav_file
from comp_guvi import extract_date, plot_gold_guvi_scan_pairs, bijective_parity
import comp_gold_funcs as gold
import comp_csv as csv
import comp_plotting as plot
from comp_on2_funcs import on2_thresh_comp
from comp_temp_funcs import tdisk_alt_comp
from comp_gen_misc import gather_nc_files, find_closest_files, extract_nc_from_tar, find_closest_filesv2

def get_args():

    parser = argparse.ArgumentParser(
        description = 'Create GOLD/GITM Data Comparisons')
    
    parser.add_argument('-type', required=True, choices=['on2', 'tdisk', 'both', 'guvi'], \
                        help = "Type of data being processed")
    
    parser.add_argument('-directory',  \
                        type=str, default = None, \
                        help = 'File path to GITM data (only plots GOLD data if not provided)')
    
    parser.add_argument('filelist', nargs='+', \
                        help = "File path for GOLD data")
    
    parser.add_argument('-method',  \
                        default = 0, type = int,
                        help = 'Select method for determining altitude for temp data. 0 for raw comparison, 1 for fixed peak weighted temp avg, ' \
                        '2 for global avg based on N2 integration lim as a function of time, 3 for weighted temp avg based on N2 int limit per pixel' \
                        '4 - LEGACY used to generate a comparison for the prior methods as a function of time. NOT a new comparison metric' \
                        '5 for altitude assignment based on contribution function data, 6 for weighted temperature average from contribution functions with an altitude offset' \
                        '7 for the weighted temperature average with peak altitude assigned from SZA via contribution functions')
    
    parser.add_argument('-visualization', 
                        action='store_const', const=1, default = 0,
                        help = 'Enable visualization (1 if present, 0 if absent)')
    
    parser.add_argument('-numthresh', nargs='*', type = float, default = [0.7], help = 'List of number density thresholds to test (*e21)')

    parser.add_argument('-alts', nargs='*', type = float, default = [224], help='List of altitudes to test (in km)')

    parser.add_argument('-fpalt', nargs='*', type = float, default = None, help='Fixed peak altitude used for final combined results plot')

    parser.add_argument('-offsets', nargs='*', type = float, default = [0], help = "List of offsets to test (in km)")

    parser.add_argument('-timescale', choices=['dbd', 'sw'], default = None, help = "Selection for day-by-day result plots or stormwise result plots. Default is both")

    parser.add_argument('-results', nargs='*', choices=['mad', 'cor', 'eot', 'rmsot', 'abseot', 'arms', 'acor'], default = None, 
                        help = "Desired result plots to display. Default is all, but if you only want specific plots, these are " \
                        "the short codes:" \
                        "mad = Mean (for the day) Absolute Difference" \
                        "cor = Cross Correlation" \
                        "eot = Error Over Time" \
                        "abseot = Absolute Error Over Time" \
                        "rmsot = RMS Over Time" \
                        "arms = Average RMS (implemented for N2 Threshold Investigations)" \
                        "acor = Average Cross Correlation (implemented for N2 Threshold Investigations)")
    
    parser.add_argument('-time_method', choices=['pbp', 'fixed'], default = 'pbp', 
                        help = "Selection for point-by-point time comparisons or fixed time comparisons")
    
    parser.add_argument('-output_dir', type=str, default=None, help="File Path for Output Plot Destination")

    parser.add_argument('-comp_var', action='store_const', const=1, default=0, help="Variable for comparing a single time to multiple thresholds/altitudes")

    parser.add_argument('-cf', type = str, default = None, help = "File path for contribution function dataset (used in TDISK method 6)")

    args = parser.parse_args()

    return args

args = get_args()

################################### FULL PROCEDURE ##################
if args.type != 'guvi':
    nc_files = gather_nc_files(args.filelist)
directory = args.directory
visualization = args.visualization
method = args.method
num_den_thresh = args.numthresh
alt_list = args.alts
offsets_list = args.offsets
timescale = args.timescale
res_strings = args.results
output_dir = args.output_dir
if output_dir == None:
    output_dir == "."
if res_strings is None:
    res_strings = ['mad', 'cor', 'eot', 'abseot', 'rmsot']
fp_alt = args.fpalt
file_list = args.filelist
cf_file_path = args.cf
csv_paths = []

if args.type == 'on2':

    # GOLD only procedure
    if args.directory == None:
        for file in nc_files:
            gold_on2_vars = gold.extract_gold_on2(file)
            gold_on2_data = gold.gold_on2_processing(gold_on2_vars)
            start_times_on2 = gold_on2_data["start_times_on2"]
            end_times_on2 = gold_on2_data["end_times_on2"]
            gold.plot_gold_on2(gold_on2_vars, start_times_on2, end_times_on2)

    if args.comp_var == 1:
        on2_thresh_comp(nc_files, num_den_thresh, directory, output_dir)

    # Regular comparisons
    if (args.time_method == 'pbp') or (args.time_method == 'fixed'):
        for file in nc_files:
            gold_on2_vars = gold.extract_gold_on2(file)
            gold_on2_data = gold.gold_on2_processing(gold_on2_vars)
            on2 = gold_on2_vars["on2"]
            time_data = gold_on2_vars["time_data_on2"]
            latitude = gold_on2_vars["latitude_on2"]
            longitude = gold_on2_vars["longitude_on2"]
            valid_on2_results = gold.extract_valid_on2_points(on2, time_data, latitude, longitude)
            start_times_on2, end_times_on2 = gold.get_scan_pair_times(valid_on2_results, on2.shape[0])

            if args.time_method == 'pbp':
                valid_on2_results = gold.extract_valid_on2_points(on2, time_data, latitude, longitude)
                gold_times = [p["time"] for p in valid_on2_results]
                time_to_file_map = find_closest_filesv2(gold_times, directory)
                gold_on2_vars["valid_on2_results"] = valid_on2_results
                csv_path, date_str = csv.basic_on2_comparison_csv(gold_on2_vars, time_to_file_map, num_den_thresh, visualization, 1, output_dir, start_times_on2, end_times_on2)
            else:
                file_paths = find_closest_files(start_times_on2, directory)
                csv_path, date_str = csv.basic_on2_comparison_csv(gold_on2_vars, file_paths, num_den_thresh, visualization, 0, output_dir, start_times_on2, end_times_on2)
            csv_paths.append(csv_path)
            if timescale == 'dbd' or timescale == None:
                plot.plot_on2_results(csv_path, date_str, start_times_on2, res_strings)

            if timescale == 'sw' or timescale == None:
                plot.plot_combined_on2(csv_paths, res_strings, num_den_thresh, output_dir)

if args.type == 'tdisk':

    if args.directory == None:
        for file in nc_files:
            gold_tdisk_vars = gold.extract_gold_tdisk(file)
            gold_tdisk_data = gold.gold_tdisk_processing(gold_tdisk_vars)
            start_times_tdisk = gold_tdisk_data["start_times_tdisk"]
            end_times_tdisk = gold_tdisk_data["end_times_tdisk"]
            gold.plot_gold_tdisk(gold_tdisk_vars, start_times_tdisk, end_times_tdisk)

    if args.comp_var == 1:
        tdisk_alt_comp(file_list, alt_list, directory, output_dir)

    if (args.time_method == 'pbp') or (args.time_method == 'fixed'):
        for file in nc_files:
            start_times = []
            cache = {}
            gitm_on2_outputs = None
            gold_tdisk_vars = gold.extract_gold_tdisk(file)
            gold_tdisk_data = gold.gold_tdisk_processing(gold_tdisk_vars)
            temperature = gold_tdisk_vars["temperature"]
            time_data = gold_tdisk_vars["time_data_tdisk"]
            latitude = gold_tdisk_vars["latitude_tdisk"]
            longitude = gold_tdisk_vars["longitude_tdisk"]
            sza = gold_tdisk_vars["sza"]
            valid_tdisk_results = gold.extract_valid_tdisk_points(temperature, time_data, latitude, longitude, sza)
            start_times_tdisk, end_times_tdisk = gold.get_scan_pair_times(valid_tdisk_results, temperature.shape[0])

            if args.time_method == 'pbp':
                pbp_var = 1
                valid_tdisk_results = gold.extract_valid_tdisk_points(temperature, time_data, latitude, longitude, sza)
                gold_times = [p["time"] for p in valid_tdisk_results]
                time_to_file_map = find_closest_filesv2(gold_times, directory)
                gold_tdisk_vars["valid_tdisk_results"] = valid_tdisk_results
            else: 
                pbp_var = 0
                file_paths = find_closest_files(start_times, directory)
                time_to_file_map = file_paths

            if method == 0:
                csv_path, date_str = csv.basic_tdisk_comparison_csv(gold_tdisk_vars, time_to_file_map, alt_list, visualization, pbp_var, output_dir, start_times_tdisk, end_times_tdisk)
                csv_paths.append(csv_path)
                if timescale == 'dbd' or timescale == None:
                    plot.plot_rawtdisk_results(csv_path, date_str, start_times_tdisk, res_strings)

            if method == 1:
                csv_path, date_str = csv.fixed_peak_tdisk_comparison(gold_tdisk_vars, time_to_file_map, alt_list, visualization, pbp_var, output_dir, start_times_tdisk, end_times_tdisk)
                csv_paths.append(csv_path)
                if timescale == 'dbd' or timescale == None:
                    plot.plot_fptdisk_results(csv_path, date_str, start_times_tdisk, res_strings)

            if method == 2:
                csv_path, date_str = csv.globaln2_temp_comparison(gold_tdisk_vars, time_to_file_map, offsets_list, num_den_thresh, visualization, pbp_var, output_dir, start_times_tdisk, end_times_tdisk)
                csv_paths.append(csv_path)
                if timescale == 'dbd' or timescale == None:
                    plot.plot_globaln2_tdisk_results(csv_path, date_str, start_times_tdisk, res_strings)

            if method == 3:
                csv_path, date_str = csv.temp_by_n2pixel_comparison(gold_tdisk_vars, time_to_file_map, offsets_list, num_den_thresh, visualization, pbp_var, output_dir, start_times_tdisk, end_times_tdisk)
                csv_paths.append(csv_path)
                if timescale == 'dbd' or timescale == None:
                    plot.plot_n2pixeltdisk_results(csv_path, date_str, start_times_tdisk, res_strings)

            if method == 4:
                csv_path0, date_str0 = csv.basic_tdisk_comparison_csv(gold_tdisk_vars, time_to_file_map, alt_list, visualization, pbp_var, output_dir, start_times_tdisk, end_times_tdisk)
                csv_path1, date_str1 = csv.fixed_peak_tdisk_comparison(gold_tdisk_vars, time_to_file_map, fp_alt, visualization, pbp_var, output_dir, start_times_tdisk, end_times_tdisk)
                csv_path2, date_str2 = csv.globaln2_temp_comparison(gold_tdisk_vars, time_to_file_map, offsets_list, num_den_thresh, visualization, 1, output_dir, start_times_tdisk, end_times_tdisk)
                csv_path3, date_str3 = csv.temp_by_n2pixel_comparison(gold_tdisk_vars, time_to_file_map, offsets_list, num_den_thresh, visualization, 1, output_dir, start_times_tdisk, end_times_tdisk)
                if isinstance(num_den_thresh, list) and len(num_den_thresh) == 1:
                    num_thresh = num_den_thresh[0]
                if isinstance(offsets_list, list) and len(offsets_list) == 1:
                    offset_val = offsets_list[0]
                plot.compare_tdisk_metrics(
                csv_paths = [csv_path0, csv_path1],
                labels = [f"Raw TDISK", f"Fixed Peak"],
                start_times = start_times_tdisk,
                output_filename = f"tdisk_all_comparison_4_{date_str0}.png",
                globaln2_path = csv_path2,
                n2pixel_path = csv_path3,
                num_thresh = num_thresh,
                offset = offset_val)

            if method == 5: 
                csv_path, date_str = csv.sza_assignment_csv(gold_tdisk_vars, time_to_file_map, visualization, pbp_var, output_dir, start_times_tdisk, end_times_tdisk)
                csv_paths.append(csv_path)

            if method == 6: 
                cf_data = Dataset(cf_file_path, 'r')
                csv_path, date_str = csv.weighted_cf_csv(gold_tdisk_vars, time_to_file_map, offsets_list, visualization, pbp_var, output_dir, start_times_tdisk, end_times_tdisk, cf_data)
                csv_paths.append(csv_path)

            if method == 7: 
                csv_path, date_str = csv.sza_peak_tdisk_comparison(gold_tdisk_vars, time_to_file_map, offsets_list, visualization, pbp_var, output_dir, start_times_tdisk, end_times_tdisk)
                csv_paths.append(csv_path)
        
        if (timescale == 'sw' or timescale == None):
            if (method == 0 or method == 1):
                plot.plot_combined_tdisk(csv_paths, alt_list, method, res_strings, output_dir)
            if (method == 2 or method == 3):
                plot.plot_tdisk_by_thresh(csv_paths, num_den_thresh, method, res_strings, output_dir)

        if method == 6:
            plot.plot_weighted_cf(csv_paths, res_strings, output_dir)

        if method == 7: 
            plot.plot_method_g(csv_paths, res_strings, output_dir)

if args.type == 'both':
    filelist = args.filelist
    on2_nc_files = []
    tdisk_nc_files = []

    for file in filelist:
        if file.lower().endswith('.tar'):
            extracted_files = extract_nc_from_tar(file)
            for extracted in extracted_files:
                if "ON2" in extracted.upper():
                    on2_nc_files.append(extracted)
                elif "TDISK" in extracted.upper():
                    tdisk_nc_files.append(extracted)
        elif file.lower().endswith('.nc'):
            if "ON2" in file.upper():
                on2_nc_files.append(file)
            elif "TDISK" in file.upper():
                tdisk_nc_files.append(file)
        else:
            print(f"Skipping unsupported file: {file}")
    on2_nc_files.sort()
    tdisk_nc_files.sort()
    
    for on2_file, tdisk_file in zip(on2_nc_files, tdisk_nc_files):
        print(f"Processing ON2: {on2_file}, TDISK: {tdisk_file}")

        gold_on2_vars = gold.extract_gold_on2(on2_file)
        gold_tdisk_vars = gold.extract_gold_tdisk(tdisk_file)

        gold_on2_data = gold.gold_on2_processing(gold_on2_vars)
        gold_tdisk_data = gold.gold_tdisk_processing(gold_tdisk_vars)

        start_times_on2 = gold_on2_data["start_times_on2"]
        end_times_on2 = gold_on2_data["end_times_on2"]

        start_times_tdisk = gold_tdisk_data["start_times_tdisk"]
        end_times_tdisk = gold_tdisk_data["end_times_tdisk"]

        gold.plot_gold_combined(gold_on2_vars, start_times_on2, end_times_on2,
                        gold_tdisk_vars, start_times_tdisk, end_times_tdisk)

if args.type == 'guvi':
    guvi_files = []
    gold_files = []
    for file in file_list:
        if file.lower().endswith(".sav"):
            guvi_files.append(file)
        elif file.lower().endswith(".nc"):
            gold_files.append(file)

    guvi_files.sort(key=extract_date)
    gold_files.sort(key=extract_date)

    if visualization == 1:
        for i in range(len(gold_files)):
            guvi = guvi_files[i]
            gold = gold_files[i]
            gold_vars = gold.extract_gold_on2(gold)
            gold_data = gold.gold_on2_processing(gold_vars)
            start_times = gold_data["start_times_on2"]
            end_times = gold_data["end_times_on2"]
            time_data = gold_vars["time_data_on2"]
            latitude_gold = gold_vars["latitude_on2"]
            longitude_gold = gold_vars["longitude_on2"]
            on2_gold = gold_vars["on2"]
            guvi_data = read_guvi_sav_file(guvi)
            guvi_time = guvi_data["times"]
            guvi_on2 = guvi_data["on2"]
            guvi_lat = guvi_data["lats"]
            guvi_lon = guvi_data["lons"]
            gold_points = gold.extract_valid_on2_points(on2_gold, time_data, latitude_gold, longitude_gold)
            plot_gold_guvi_scan_pairs(gold_points, guvi_lat, guvi_lon, guvi_on2, guvi_time)

    bijective_parity(gold_files, guvi_files)  