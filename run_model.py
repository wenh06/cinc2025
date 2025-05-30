#!/usr/bin/env python

# Please do *not* edit this script. Changes will be discarded so that we can run the trained models consistently.

# This file contains functions for running your model for the Challenge. You can run it as follows:
#
#   python run_model.py -d data -m model -o outputs -v
#
# where 'data' is a folder containing the Challenge data, 'model' is a folder containing the your trained model, 'outputs' is a
# folder for saving your model's outputs, and -v is an optional verbosity flag.

import argparse
import os
import sys

from helper_code import *
from team_code import load_model, run_model

# Parse arguments.
def get_parser():
    description = 'Run the trained Challenge models.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-p', '--output_file', type=str, required=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-f', '--allow_failures', action='store_true')
    return parser

# Run the code.
def run(args):
    # Load the models.
    if args.verbose:
        print('Loading the Challenge model...')

    # You can use these functions to perform tasks, such as loading your model, that you only need to perform once.
    model = load_model(args.model_folder, args.verbose) ### Teams: Implement this function!!!

    # Find the Challenge data.
    if args.verbose:
        print('Finding the Challenge data...')

    records = find_records(args.data_folder)
    num_records = len(records)

    if num_records == 0:
        raise Exception('No data were provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(args.output_folder, exist_ok=True)

    # Run the team's model on the Challenge data.
    if args.verbose:
        print('Running the Challenge model on the Challenge data...')

    # Iterate over the records.
    for i, record in enumerate(records):
        if args.verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {record}...')

        # Allow or disallow the model to fail on parts of the data; this can be helpful for debugging.
        try:
            binary_output, probability_output = run_model(os.path.join(args.data_folder, record), model, args.verbose) ### Teams: Implement this function!!!
        except:
            if args.allow_failures:
                if args.verbose:
                    print('... failed.')
                binary_output, probability_output = float('nan'), float('nan')
            else:
                raise

        # Save Challenge outputs.
        head, tail = os.path.split(record)
        output_path = os.path.join(args.output_folder, head)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(args.output_folder, record + '.txt')
        save_outputs(output_file, tail, binary_output, probability_output)

    if args.output_file:
        collect_outputs(args.output_file, records, args.output_folder)

    if args.verbose:
        print('Done.')

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))