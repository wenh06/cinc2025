#!/usr/bin/env python

# Load libraries.
import argparse
import h5py
import numpy as np
import os
import os.path
import pandas as pd
import sys
import wfdb

from helper_code import is_integer, is_boolean, sanitize_boolean_value

# Parse arguments.
def get_parser():
    description = 'Prepare the CODE-15% dataset for the Challenge.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--signal_files', type=str, required=True, nargs='*') # exams_part0.hdf5, exams_part1.hdf5, ...
    parser.add_argument('-d', '--demographics_file', type=str, required=True) # exams.csv
    parser.add_argument('-l', '--labels_file', type=str, required=True) # code15_chagas_labels.csv
    parser.add_argument('-f', '--signal_format', type=str, required=False, default='dat', choices=['dat', 'mat'])
    parser.add_argument('-o', '--output_paths', type=str, required=True, nargs='*')
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
    if write_dir:
        cwd = os.getcwd()
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
    # Load the patient demographic data.
    exam_id_to_age = dict()
    exam_id_to_sex = dict()

    df = pd.read_csv(args.demographics_file)
    for idx, row in df.iterrows():
        exam_id = row['exam_id']
        assert(is_integer(exam_id))
        exam_id = int(exam_id)

        age = row['age']
        assert(is_integer(age))
        age = int(age)
        exam_id_to_age[exam_id] = age

        is_male = row['is_male']
        assert(is_boolean(is_male))
        is_male = sanitize_boolean_value(is_male)
        sex = 'Male' if is_male else 'Female' # This variable was encoding as a binary value.
        exam_id_to_sex[exam_id] = sex

    # Load the Chagas labels.
    exam_id_to_chagas = dict()

    df = pd.read_csv(args.labels_file)
    for idx, row in df.iterrows():
        exam_id = row['exam_id']
        assert(is_integer(exam_id))
        exam_id = int(exam_id)

        chagas = row['chagas']
        assert(is_boolean(chagas))
        chagas = sanitize_boolean_value(chagas)
        exam_id_to_chagas[exam_id] = bool(chagas)

    # Load and convert the signal data.

    # See https://zenodo.org/records/4916206 for more information about these values.
    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    sampling_frequency = 400
    units = 'mV'

    # Define the paramaters for the WFDB files.
    gain = 1000
    baseline = 0
    num_bits = 16
    fmt = str(num_bits)

    # Define the output paths for the signal files, i.e., you can use a different path for each signal file or the same path for
    # every signal file.
    if len(args.output_paths) == len(args.signal_files):
        signal_files = args.signal_files
        output_paths = args.output_paths
    elif len(args.output_paths) == 1:
        signal_files = args.signal_files
        output_paths = [args.output_paths[0]]*len(args.signal_files)
    else:
        raise Exception('The number of signal files must match the number of output paths.')

    num_groups = len(signal_files)

    # Iterate over the input signal files.
    for k in range(num_groups):
        signal_file = signal_files[k]
        output_path = output_paths[k]
        os.makedirs(output_path, exist_ok=True)

        with h5py.File(signal_file, 'r') as f:
            exam_ids = list(f['exam_id'])
            num_exam_ids = len(exam_ids)

            # Iterate over the exam IDs in each signal file.
            for i in range(num_exam_ids):
                exam_id = exam_ids[i]

                # Skip exam IDs without Chagas labels.
                if not exam_id in exam_id_to_chagas:
                    continue
                else:
                    pass

                physical_signals = np.array(f['tracings'][i], dtype=np.float32)

                # Perform basic error checking on the signal.
                num_samples, num_leads = np.shape(physical_signals)
                assert(num_leads == 12)

                # Remove zero padding at the start and end of the signals..
                r = 0
                while r < num_samples and np.all(physical_signals[r, :] == 0):
                    r += 1

                s = num_samples
                while s > r and np.all(physical_signals[s-1, :] == 0):
                    s -= 1

                if r >= s:
                    continue
                else:
                    physical_signals = physical_signals[r:s, :]

                # Convert the signal to digital units; saturate the signal and represent NaNs as the lowest representable integer.
                digital_signals = gain * physical_signals
                digital_signals = np.round(digital_signals)
                digital_signals = np.clip(digital_signals, -2**(num_bits-1)+1, 2**(num_bits-1)-1)
                digital_signals[~np.isfinite(digital_signals)] = -2**(num_bits-1)
                digital_signals = np.asarray(digital_signals, dtype=np.int32) # We need to promote from 16-bit integers due to an error in the Python WFDB library.

                # Add the exam ID, the patient ID, age, sex, Chagas label, and data source.
                age = exam_id_to_age[exam_id]
                sex = exam_id_to_sex[exam_id]
                chagas = exam_id_to_chagas[exam_id]
                source = 'CODE-15%'
                comments = [f'Age: {age}', f'Sex: {sex}', f'Chagas label: {chagas}', f'Source: {source}']

                # Save the signal.
                record = str(exam_id)
                wfdb.wrsamp(record, fs=sampling_frequency, units=[units]*num_leads, sig_name=lead_names,
                            d_signal=digital_signals, fmt=[fmt]*num_leads, adc_gain=[gain]*num_leads, baseline=[baseline]*num_leads,
                            write_dir=output_path, comments=comments)

                # Convert data from .dat files to .mat files, if requested.
                if args.signal_format in ('mat', '.mat'):
                    convert_dat_to_mat(record, write_dir=output_path)

                # Recompute the checksums as needed.
                checksums = np.sum(digital_signals, axis=0, dtype=np.int16)
                fix_checksums(os.path.join(output_path, record), checksums)

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))