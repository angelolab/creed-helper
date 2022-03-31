import os
import shutil
import time
import tempfile
import json
from pathlib import Path
from multiprocessing.pool import ThreadPool as Pool

from pytest_cases import parametrize_with_cases

from mibi_bin_tools import io_utils

from toffy.test_utils import WatcherCases
from toffy.fov_watcher import start_watcher

TISSUE_DATA_PATH = os.path.join(Path(__file__).parent, 'data', 'tissue')


def _slow_copy_sample_tissue_data(dest: str, delta: int = 10):
    """slowly copies files from ./data/tissue/

    Args:
        dest (str):
            Where to copy tissue files to
        delta (int):
            Time (in seconds) between each file copy
    """

    print('copytime')

    for tissue_file in os.listdir(TISSUE_DATA_PATH):
        shutil.copy(os.path.join(TISSUE_DATA_PATH, tissue_file), dest)
        time.sleep(delta)
    print('copies done!')


TISSUE_RUN_JSON_SPOOF = {
    'fovs': [
        {'runOrder': 1, 'scanCount': 1},
        {'runOrder': 2, 'scanCount': 1},
    ],
}


# TODO: add tests for per_run when per_run callbacks are created
@parametrize_with_cases('per_fov_partial, per_run, validators', cases=WatcherCases)
def test_watcher(per_fov_partial, per_run, validators):
    with tempfile.TemporaryDirectory() as tmpdir:

        RUN_DIR_NAME = 'run_XXX'

        per_fov = []
        for i, func in enumerate(per_fov_partial):
            cb_dir = os.path.join(tmpdir, f'cb_{i}', RUN_DIR_NAME)
            os.makedirs(cb_dir)
            per_fov.append(func(cb_dir))

        run_data = os.path.join(tmpdir, 'test_run')
        log_out = os.path.join(tmpdir, 'log_output')
        os.makedirs(run_data)
        os.makedirs(log_out)

        with open(os.path.join(run_data, 'test_run.json'), 'w') as f:
            json.dump(TISSUE_RUN_JSON_SPOOF, f)

        with Pool(processes=4) as pool:
            pool.apply_async(_slow_copy_sample_tissue_data, (run_data, 6))
            res_scan = pool.apply_async(start_watcher, (run_data, log_out, per_fov, per_run, 2))

            res_scan.get()

        with open(os.path.join(log_out, 'test_run_log.txt')) as f:
            print(f.read())

        fovs = [
            bin_file.split('.')[0]
            for bin_file in io_utils.list_files(TISSUE_DATA_PATH, substrs=['.bin'])
        ]

        for i, validator in enumerate(validators):
            validator(os.path.join(tmpdir, f'cb_{i}', RUN_DIR_NAME), fovs)

    pass
