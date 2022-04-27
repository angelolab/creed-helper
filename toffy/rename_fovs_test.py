import tempfile
import json
import os
import pytest

from ark.utils import io_utils
from toffy import rename_fovs as rf


def create_sample_run(name_list, run_order_list, scan_count_list, create_json=False, bad=False):
    fov_list = []
    sample_run = {"fovs": fov_list}

    # set up dictionary
    for name, run_order, scan_count in zip(name_list, run_order_list, scan_count_list):
        ex_fov = {
            "scanCount": scan_count,
            "runOrder": run_order,
            "name": name
        }
        fov_list.append(ex_fov)

    # delete name key if one is not provided
    for fov in sample_run.get('fovs', ()):
        if fov.get('name') is None:
            del fov['name']

    # create bad dictionary
    if bad:
        sample_run['bad key'] = sample_run['fovs']
        del sample_run['fovs']

    # create json file for the data
    if create_json:
        temp = tempfile.NamedTemporaryFile(mode="w", delete=False)
        json.dump(sample_run, temp)
        return temp.name

    return sample_run


def create_sample_fov_dirs(fovs, base_dir):
    # make fov subdirectories
    for fov in fovs:
        os.mkdir(os.path.join(base_dir, fov))


def remove_fov_dirs(base_dir):
    # delete fov subdirectories
    fovs = io_utils.list_folders(base_dir)
    for fov in fovs:
        os.rmdir(os.path.join(base_dir, fov))


def test_rename_missing_fovs():
    # data with missing names
    ex_name = ['custom_1', None, 'custom_2', 'custom_3', None]
    ex_run_order = list(range(1, 6))
    ex_scan_count = list(range(1, 6))

    # create a dict with the sample data
    ex_run = create_sample_run(ex_name, ex_run_order, ex_scan_count)

    # test that missing names are given a placeholder
    rf.rename_missing_fovs(ex_run)
    for fov in ex_run.get('fovs', ()):
        assert fov.get('name') is not None


def test_rename_fov_dirs():
    with tempfile.TemporaryDirectory() as base_dir:
        # create run file and fov folder directories
        dirs = ['run_folder', 'fov_folder']
        for directory in dirs:
            os.mkdir(os.path.join(base_dir, directory))
        run_dir = os.path.join(base_dir, 'run_folder')
        fov_dir = os.path.join(base_dir, 'fov_folder')

        # create existing new directory
        not_new_dir = os.path.join(base_dir, 'not_new_directory')
        os.mkdir(not_new_dir)

        # existing directory for new_dir should raise an error
        with pytest.raises(ValueError, match="already exists"):
            rf.rename_fov_dirs(run_dir, fov_dir, not_new_dir)

        # regular sample run data
        ex_name = ['custom_1', 'custom_2', 'custom_3']
        ex_run_order = list(range(1, 4))
        ex_scan_count = [1, 2, 1]

        # create a json path with the sample data
        ex_run_path = create_sample_run(ex_name, ex_run_order, ex_scan_count, True)

        # bad sample run data
        bad_run = create_sample_run(ex_name, ex_run_order, ex_scan_count, True, bad=True)

        # bad run file data should raise an error
        with pytest.raises(KeyError):
            rf.rename_fov_dirs(bad_run, fov_dir)

        # create already renamed fov folders
        renamed_fovs = ['custom_1', 'custom_2-1', 'custom_2-2', 'custom_3']
        create_sample_fov_dirs(renamed_fovs, fov_dir)

        # fov folders already renamed should raise an error
        with pytest.raises(ValueError, match=r"already been renamed"):
            rf.rename_fov_dirs(ex_run_path, fov_dir)
        remove_fov_dirs(fov_dir)

        # create correct fov folders
        correct_fovs = ['fov-1-scan-1', 'fov-2-scan-1', 'fov-2-scan-2', 'fov-3-scan-1']
        create_sample_fov_dirs(correct_fovs, fov_dir)

        # test successful renaming to new dir
        rf.rename_fov_dirs(ex_run_path, fov_dir, os.path.join(base_dir, 'new_directory'))
        new_dir = os.path.join(base_dir, 'new_directory')
        assert set(io_utils.list_folders(new_dir)) == set(renamed_fovs)
        remove_fov_dirs(new_dir)

        # test successful renaming
        rf.rename_fov_dirs(ex_run_path, fov_dir)
        assert set(io_utils.list_folders(fov_dir)) == set(renamed_fovs)
        remove_fov_dirs(fov_dir)

        # create not enough fov folders
        less_fovs = ['fov-1-scan-1', 'fov-2-scan-1', 'fov-3-scan-1']
        create_sample_fov_dirs(less_fovs, fov_dir)

        # fovs in rule file without an existing dir should raise an error
        with pytest.warns(UserWarning, match="Not all FOVs"):
            rf.rename_fov_dirs(ex_run_path, fov_dir)
        remove_fov_dirs(fov_dir)

        # create extra fov folders
        extra_fovs = ['fov-1-scan-1', 'fov-2-scan-1', 'fov-2-scan-2',
                      'fov-3-scan-1', 'fov-3-scan-3']
        create_sample_fov_dirs(extra_fovs, fov_dir)

        # extra dirs not listed in run file should raise an error
        with pytest.raises(ValueError, match="not found in run file"):
            rf.rename_fov_dirs(ex_run_path, fov_dir)
        remove_fov_dirs(fov_dir)

        # delete sample run json
        os.remove(ex_run_path)
