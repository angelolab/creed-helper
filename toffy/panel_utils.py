import os
import csv
import pandas as pd

from ark.utils import io_utils


def convert_panel(panel_path):
    panel_name = os.path.basename(panel_path).split('.')[0]
    panel_dir = os.path.dirname(panel_path)
    example_panel = pd.read_csv(os.path.join('..', 'files', 'example_panel_file.csv'))

    with open(panel_path) as r:
        headers = [next(r) for i in list(range(11))]
        cols = headers[10].split(',')
        panel = pd.read_csv(r, sep=',', names=cols)

    # retrieve original mass and target values
    panel.columns = [panel.columns.str.replace('"', '')]
    panel.columns = panel.columns.get_level_values(0)
    toffy_panel = panel[['Mass', 'Target']].copy()

    # edit panel
    mass_start =[]
    mass_stop = []

    for i, row in toffy_panel.iterrows():
        mass = row['Mass']

        mass_start.append(mass-0.3)
        mass_stop.append(mass)

    toffy_panel['Start'] = mass_start
    toffy_panel['Stop'] = mass_stop

    toffy_panel = pd.concat([toffy_panel, example_panel], ignore_index=True)
    toffy_panel.to_csv(os.path.join(panel_dir, panel_name + '-toffy.csv'), index=False)


def load_panel(panel_path):
    panel_name = os.path.basename(panel_path).split('.')[0]
    panel_dir = os.path.dirname(panel_path)

    # if panel path points to toffy panel, read in
    if '-toffy' in panel_name:
        toffy_panel = pd.read_csv(panel_path)
    else:
        files = io_utils.remove_file_extensions(io_utils.list_files(panel_dir, substrs=panel_name))
        # check if toffy panel exists in panel_dir
        if panel_name + '-toffy.csv' in files:
            converted_panel_path = os.path.join(panel_dir, panel_name + '-toffy.csv')
            toffy_panel = pd.read_csv(converted_panel_path)
        # if no toffy panel, create one
        else:
            toffy_panel = convert_panel(panel_path)

    return toffy_panel





