from __future__ import print_function
import glob

def extract_data(string):
    """
    extracts and stores information from string
    """
    print(a, b)
    pass

log_files_path = "logs/*.txt"
log_files = glob.glob(log_files_path)

for file in log_files:
    # extract data from log_file "file"
    with open(file, "r") as f:
        f_line = f.readline()       # data is in first line
        print(f_line)
        # mess with data
        extract_data(f_line)
