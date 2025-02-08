#!/usr/bin/env python

# Load libraries.
import argparse
import numpy as np
import os
import os.path
import pandas as pd
import shutil
import sys
import wfdb

from helper_code import find_records, get_signal_files, is_integer

# Parse arguments.
def get_parser():
    description = 'Prepare the PTB-XL database for use in the Challenge.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True) # records100 or records500
    parser.add_argument('-d', '--ptbxl_database_file', type=str, required=True) # ptbxl_database.csv
    parser.add_argument('-f', '--signal_format', type=str, required=False, default='dat', choices=['dat', 'mat'])
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Suppress stdout for noisy commands.
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = stdout

# Convert .dat files to .mat files (optional).
def convert_dat_to_mat(record, write_dir=None):
    import wfdb.io.convert

    # Change the current working directory; wfdb.io.convert.matlab.wfdb_to_matlab places files in the current working directory.
    cwd = os.getcwd()
    if write_dir:
        os.chdir(write_dir)

    # Convert the .dat file to a .mat file.
    with suppress_stdout():
        wfdb.io.convert.matlab.wfdb_to_mat(record)

    # Remove the .dat file.
    os.remove(record + '.hea')
    os.remove(record + '.dat')

    # Rename the .mat file.
    os.rename(record + 'm' + '.hea', record + '.hea')
    os.rename(record + 'm' + '.mat', record + '.mat')

    # Update the header file with the renamed record and .mat file.
    with open(record + '.hea', 'r') as f:
        output_string = ''
        for l in f:
            if l.startswith('#Creator') or l.startswith('#Source'):
                pass
            else:
                l = l.replace(record + 'm', record)
                output_string += l

    with open(record + '.hea', 'w') as f:
        f.write(output_string)

    # Change the current working directory back to the previous current working directory.
    if write_dir:
        os.chdir(cwd)

# Fix the checksums from the Python WFDB library.
def fix_checksums(record, checksums=None):
    if checksums is None:
        x = wfdb.rdrecord(record, physical=False)
        signals = np.asarray(x.d_signal)
        checksums = np.sum(signals, axis=0, dtype=np.int16)

    header_filename = os.path.join(record + '.hea')
    string = ''
    with open(header_filename, 'r') as f:
        for i, l in enumerate(f):
            if i == 0:
                arrs = l.split(' ')
                num_leads = int(arrs[1])
            if 0 < i <= num_leads and not l.startswith('#'):
                arrs = l.split(' ')
                arrs[6] = str(checksums[i-1])
                l = ' '.join(arrs)
            string += l

    with open(header_filename, 'w') as f:
        f.write(string)

# Run script.
def run(args):
    # Load the demographic information.
    df = pd.read_csv(args.ptbxl_database_file, index_col='ecg_id')

    # Identify the header files.
    records = find_records(args.input_folder)

    # Update the header files to include demographics data and copy the signal files unchanged.
    for record in records:

        # Extract the demographics data.
        record_path, record_basename = os.path.split(record)
        ecg_id = int(record_basename.split('_')[0])
        row = df.loc[ecg_id]

        recording_date_string = row['recording_date']
        date_string, time_string = recording_date_string.split(' ')
        yyyy, mm, dd = date_string.split('-')
        date_string = f'{dd}/{mm}/{yyyy}'

        age = row['age']
        age = int(age) if is_integer(age) else float(age)

        sex = row['sex']
        if sex == 0:
            sex = 'Male'
        elif sex == 1:
            sex = 'Female'
        else:
            sex = 'Unknown'

        height = row['height']
        height = int(height) if is_integer(height) else float(height)

        weight = row['weight']
        weight = int(weight) if is_integer(weight) else float(weight)

        # Assume that all of the patients are negative for Chagas, which is likely to be the case for every or almost every patient
        # in the PTB-XL dataset.
        label = False

        # Update the header file.
        input_header_file = os.path.join(args.input_folder, record + '.hea')
        output_header_file = os.path.join(args.output_folder, record + '.hea')

        input_path = os.path.join(args.input_folder, record_path)
        output_path = os.path.join(args.output_folder, record_path)

        os.makedirs(output_path, exist_ok=True)

        with open(input_header_file, 'r') as f:
            input_header = f.read()

        lines = input_header.split('\n')
        record_line = ' '.join(lines[0].strip().split(' ')[:4]) + '\n'
        signal_lines = '\n'.join(l.strip() for l in lines[1:] \
            if l.strip() and not l.startswith('#')) + '\n'
        comment_lines = '\n'.join(l.strip() for l in lines[1:] \
            if l.startswith('#') and not any((l.startswith(x) for x in ('# Age:', '# Sex:', '# Height:', '# Weight:', '# Chagas label:')))) + '\n'

        record_line = record_line.strip() + f' {time_string} {date_string} ' + '\n'
        signal_lines = signal_lines.strip() + '\n'
        comment_lines = comment_lines.strip() + f'# Age: {age}\n# Sex: {sex}\n# Height: {height}\n# Weight: {weight}\n# Chagas label: {label}\n'

        output_header = record_line + signal_lines + comment_lines

        with open(output_header_file, 'w') as f:
            f.write(output_header)

        # Copy the signal files if the input and output folders are different.
        if os.path.normpath(args.input_folder) != os.path.normpath(args.output_folder):
            relative_path = os.path.split(record)[0]

            signal_files = get_signal_files(input_header_file)
            for signal_file in signal_files:
                input_signal_file = os.path.join(args.input_folder, relative_path, signal_file)
                output_signal_file = os.path.join(args.output_folder, relative_path, signal_file)
                if os.path.isfile(input_signal_file):
                    shutil.copy2(input_signal_file, output_signal_file)

        # Convert data from .dat files to .mat files as requested.
        if args.signal_format in ('mat', '.mat'):
            convert_dat_to_mat(record, write_dir=args.output_folder)

        # Recompute the checksums as needed.
        fix_checksums(os.path.join(args.output_folder, record))

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))