import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from netCDF4 import Dataset, num2date

def read_gitm_headers_nc(file_path):
    with Dataset(file_path) as ds:
        time_var = ds.variables["time"]
        nc_time = time_var[0]
        time_units = time_var.units 
        calendar = getattr(time_var, "calendar", "standard")
        py_time = num2date(nc_time, units=time_units, calendar=calendar)

        header = {
            "nFiles": 1,
            "version": ds.getncattr("version"),
            "nLons": len(ds.dimensions["lon"]),
            "nLats": len(ds.dimensions["lat"]),
            "nAlts": len(ds.dimensions["z"]),
            "filename": [file_path],
            "time": [py_time],
            "vars": [
                "Longitude", "Latitude", "Altitude",
                "[O] (/m3)", "[N2] (/m3)", "Tn (K)"]
        }

    return header

def read_gitm_one_file_nc(file_path, vars_to_read):
    with Dataset(file_path) as ds:
        # Map the header "vars" list to actual NetCDF variable names
        var_map = {
            0: "Longitude",
            1: "Latitude",
            2: "Altitude",
            3: "O",
            4: "N2",
            5: "Tn"}

        data = {"nLons": len(ds.dimensions["lon"]), "nLats": len(ds.dimensions["lat"]), "nAlts": len(ds.dimensions["z"]),}

        for i in vars_to_read:
            nc_var = var_map[i]
            arr = ds.variables[nc_var][0, ...]  # squeeze time dimension if size=1
            data[i] = np.array(arr)

        # Include time as a datetime object
        time_var = ds.variables["time"]
        nc_time = time_var[0]
        time_units = time_var.units
        calendar = getattr(time_var, "calendar", "standard")
        data["time"] = num2date(nc_time, units=time_units, calendar=calendar)

    return data

