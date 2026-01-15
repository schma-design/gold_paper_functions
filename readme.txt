To generate a comarison for GOLD, run the comp_routine.py script. This script takes in a variety of arguments, 
but a required argument for all comparisons is the file path to the desired GOLD data, whether that is a 
.nc or .tar file. 

The user can then choose the type of comparison or plot they would like (between on2, tdisk, or guvi). If no 
GITM data directory is provided, the script will plot the raw GOLD data. Depending on the type of comparison and
method/methods desired, there are various arguments that can be inputted (use the "help" feature in get_args 
to learn more). 

An example input for generating a GOLD/GITM temperature comparison for a method that implements an offset range 
is included below.

python .comp_routine -type tdisk .\gold\TDISK2024051020240512.tar -directory .\gitm_data -visualization 
-method 7 -offsets 0 10 20 30 40 50 -output_dir .\output_directory -results mad arms -timescale sw

Though these scripts were intended to be called from comp_routine.py, all GITM global comparison plots were 
produced from the functions within comp_on2_funcs.py and comp_temp_funcs.py. The GUVI plots were produced from 
the functions within comp_guvi.py. All line plots were produced via the functions in comp_plotting.py. 