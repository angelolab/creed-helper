import json
import os
import warnings
import copy
from distutils.dir_util import copy_tree

from ark.utils import io_utils
from ark.utils.misc_utils import verify_in_list
from toffy.tiling_utils import rename_duplicate_fovs


def rename_missing_fovs(fov_data):
    """Identify FOVs that are missing the 'name' key and create one with value placeholder_{n}
    Args:
        fov_data (dict): the FOV run JSON data

    Returns:
        dict: a copy of the run JSON data with placeholder names for FOVs that lack one
       """

    copy_fov_data = copy.deepcopy(fov_data)

    # count of FOVs that are missing the 'name' key
    missing_count = 0

    # iterate over each FOV and add a placeholder name if necessary
    for fov in copy_fov_data['fovs']:
        if 'name' not in fov.keys():
            missing_count += 1
            fov['name'] = f'placeholder_{missing_count}'

    return copy_fov_data


def rename_fov_dirs(run_path, fov_dir, new_dir=None):
    """Renames FOV directories with default_name to have custom_name sourced from the run JSON file

    Args:
        run_path (str): path to the JSON run file which contains custom name values
        fov_dir (str): directory where the FOV default named subdirectories are stored
        new_dir (str): path to new directory to output renamed folders to, defaults to None

    Raises:
        KeyError: issue reading keys from the JSON file
        ValueError: there exist default named directories that are not described in the run file
        UserWarning: not all custom names from the run file have an existing directory

        """

    io_utils.validate_paths(run_path)
    io_utils.validate_paths(fov_dir)

    # check that new_dir doesn't already exist
    if new_dir is not None:
        if os.path.exists(new_dir):
            raise ValueError(f"The new directory supplied already exists: {new_dir}")

    with open(run_path) as file:
        run_metadata = json.load(file)

    # check for missing or duplicate fov names
    run_metadata = rename_missing_fovs(run_metadata)
    run_metadata = rename_duplicate_fovs(run_metadata)

    # retrieve custom names and number of scans for each fov, construct matching default names
    fov_scan = dict()
    for fov in run_metadata.get('fovs', ()):
        custom_name = fov.get('name')
        run_order = fov.get('runOrder', -1)
        scans = fov.get('scanCount', -1)
        if run_order * scans < 0:
            raise KeyError(f"Could not locate keys in {run_path}")

        # fovs with multiple scans have scan number specified
        if scans > 1:
            for scan in range(1, scans + 1):
                default_name = f'fov-{run_order}-scan-{scan}'
                fov_scan[default_name] = f'{custom_name}-{scan}'
        else:
            default_name = f'fov-{run_order}-scan-{scans}'
            fov_scan[default_name] = custom_name

    # retrieve current default directory names, check if already renamed
    old_dirs = io_utils.list_folders(fov_dir)
    if set(old_dirs) == set(fov_scan.values()):
        raise ValueError(f"All FOV folders in {fov_dir} have already been renamed")

    # check if custom fov names & scan counts match the number of existing default directories
    try:
        verify_in_list(default=list(fov_scan.keys()), existing_folders=old_dirs)
    except ValueError:
        warnings.warn(f"Not all FOVs specified in {run_path} have an existing directory")

    verify_in_list(existing_folders_not_found_in_run_file=old_dirs, default=list(fov_scan.keys()))

    # validate new_dir and copy contents of fov_dir
    if new_dir is not None:
        copy_tree(fov_dir, new_dir)
        change_dir = new_dir
    else:
        change_dir = fov_dir

    # change the default directory names to custom names
    for folder in fov_scan:
        fov_subdir = os.path.join(change_dir, folder)
        if os.path.isdir(fov_subdir):
            new_name = os.path.join(change_dir, fov_scan[folder])
            os.rename(fov_subdir, new_name)