import logging
import os
import shutil
import tempfile
import time
import warnings
from multiprocessing import TimeoutError
from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from alpineer import io_utils
from pytest_cases import parametrize_with_cases
from skimage.io import imsave

from toffy.fov_watcher import start_watcher
from toffy.json_utils import write_json_file
from toffy.settings import QC_COLUMNS, QC_SUFFIXES
from toffy.watcher_callbacks import build_callbacks

from .utils.test_utils import (
    TEST_CHANNELS,
    RunStructureCases,
    RunStructureTestContext,
    WatcherCases,
    mock_visualize_mph,
    mock_visualize_qc_metrics,
)

COMBINED_DATA_PATH = os.path.join(Path(__file__).parents[1], "data", "combined")
RUN_DIR_NAME = "run_XXX"

SLOW_COPY_INTERVAL_S = 1


def _reset_logging_file():
    with open(os.path.join(Path(__file__).parents[1], "pytest.txt"), "w"):
        pass


def _slow_copy_sample_tissue_data(
    dest: str, delta: int = 10, one_blank: bool = False, temp_bin: bool = False
):
    """Slowly copies files from ./data/tissue/.

    Args:
        dest (str):
            Where to copy tissue files to
        delta (int):
            Time (in seconds) between each file copy
        one_blank (bool):
            Add a blank .bin file or not
        temp_bin (bool):
            Use initial temp bin file paths or not
    """
    for tissue_file in sorted(os.listdir(COMBINED_DATA_PATH)):
        time.sleep(delta)
        if one_blank and ".bin" in tissue_file and tissue_file[0] != ".":
            # create blank (0 size) file
            open(os.path.join(dest, tissue_file), "w").close()
            one_blank = False
        else:
            tissue_path = os.path.join(COMBINED_DATA_PATH, tissue_file)
            if temp_bin and ".bin" in tissue_file:
                # copy to a temporary file with hash extension, then move to dest folder
                new_tissue_path = os.path.join(COMBINED_DATA_PATH, "." + tissue_file + ".aBcDeF")
                shutil.copy(tissue_path, new_tissue_path)
                shutil.copy(new_tissue_path, dest)
                os.remove(new_tissue_path)

                # simulate a renaming event in dest
                time.sleep(delta)
                copied_tissue_path = os.path.join(dest, "." + tissue_file + ".aBcDeF")
                os.rename(copied_tissue_path, os.path.join(dest, tissue_file))
            else:
                shutil.copy(tissue_path, dest)
                tissue_data = os.path.splitext(tissue_file)

                # simulate a .bin file update to test _check_bin_updates
                if tissue_data[1] == ".json" and "_processing" not in tissue_data[0]:
                    # ensure a sleep so the update doesn't interfere with an existing extraction
                    time.sleep(3)
                    bin_file_name = tissue_data[0] + ".bin"

                    # make sure only to update non-blank bin files
                    if os.path.getsize(os.path.join(dest, bin_file_name)) != 0:
                        shutil.copy(
                            os.path.join(COMBINED_DATA_PATH, bin_file_name),
                            os.path.join(dest, bin_file_name + ".temp"),
                        )
                        os.remove(os.path.join(dest, bin_file_name))
                        os.rename(
                            os.path.join(dest, bin_file_name + ".temp"),
                            os.path.join(dest, bin_file_name),
                        )


COMBINED_RUN_JSON_SPOOF = {
    "rois": [
        {
            "fovs": [
                {
                    "runOrder": 1,
                    "scanCount": 1,
                    "name": "R1C1",
                    "frameSizePixels": {"width": 32, "height": 32},
                },
                {
                    "runOrder": 2,
                    "scanCount": 1,
                    "name": "R2C1",
                    "frameSizePixels": {"width": 32, "height": 32},
                },
                {
                    "runOrder": 3,
                    "scanCount": 1,
                    "name": "R1C2",
                    "frameSizePixels": {"width": 32, "height": 32},
                    "standardTarget": "Molybdenum Foil",
                },
                {
                    "runOrder": 4,
                    "scanCount": 1,
                    "name": "R2C2",
                    "frameSizePixels": {"width": 32, "height": 32},
                },
            ]
        }
    ]
}


@parametrize_with_cases("run_json, expected_files", cases=RunStructureCases)
def test_run_structure(run_json, expected_files, recwarn):
    with RunStructureTestContext(run_json, files=expected_files) as (
        tmpdir,
        run_structure,
    ):
        for file in expected_files:
            run_structure.check_run_condition(os.path.join(tmpdir, file))
        assert all(run_structure.check_fov_progress().values())

        # hidden files should not throw an invalid FOV file warning but still be skipped
        exist, name = run_structure.check_run_condition(os.path.join(tmpdir, ".fake_file.txt"))
        for warn_data in recwarn.list:
            assert "not a valid FOV file and will be skipped" not in str(warn_data.message)
        assert not exist and name == ""

        # check for invalid file format
        exist, name = run_structure.check_run_condition(os.path.join(tmpdir, "fov.bin.txt"))
        assert not exist and name == ""

        # check for fake files
        with pytest.warns(Warning, match="This should be unreachable..."):
            exist, name = run_structure.check_run_condition(os.path.join(tmpdir, "fake_file.txt"))
        assert not exist and name == ""


def _slow_create_run_folder(run_folder_path: str, lag_time: int):
    time.sleep(lag_time)
    os.makedirs(run_folder_path)
    write_json_file(
        json_path=os.path.join(run_folder_path, "test_run.json"),
        json_object=COMBINED_RUN_JSON_SPOOF,
        encoding="utf-8",
    )


@patch("toffy.watcher_callbacks.visualize_qc_metrics", side_effect=mock_visualize_qc_metrics)
@patch("toffy.watcher_callbacks.visualize_mph", side_effect=mock_visualize_mph)
@pytest.mark.parametrize("run_folder_lag", [0, 5, 15])
@parametrize_with_cases(
    "run_cbs,int_cbs,fov_cbs,kwargs,validators,watcher_start_lag,existing_data",
    cases=WatcherCases.case_default,
)
def test_watcher_run_timeout(
    mock_viz_qc,
    mock_viz_mph,
    run_cbs,
    int_cbs,
    fov_cbs,
    kwargs,
    validators,
    watcher_start_lag,
    existing_data,
    run_folder_lag,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        tiff_out_dir = os.path.join(tmpdir, "cb_0", RUN_DIR_NAME)
        qc_out_dir = os.path.join(tmpdir, "cb_1", RUN_DIR_NAME)
        mph_out_dir = os.path.join(tmpdir, "cb_2", RUN_DIR_NAME)
        plot_dir = os.path.join(tmpdir, "cb_2_plots", RUN_DIR_NAME)
        pulse_out_dir = os.path.join(tmpdir, "cb_3", RUN_DIR_NAME)

        # add directories to kwargs
        kwargs["tiff_out_dir"] = tiff_out_dir
        kwargs["qc_out_dir"] = qc_out_dir
        kwargs["mph_out_dir"] = mph_out_dir
        kwargs["pulse_out_dir"] = pulse_out_dir
        kwargs["plot_dir"] = plot_dir

        # ensure warn_overwrite set to False if intermediate callbacks set, otherwise True
        kwargs["warn_overwrite"] = True if int_cbs else False

        run_folder = os.path.join(tmpdir, "test_run")
        log_out = os.path.join(tmpdir, "log_output")
        fov_callback, run_callback, intermediate_callback = build_callbacks(
            run_cbs, int_cbs, fov_cbs, **kwargs
        )

        with Pool(processes=4) as pool:
            pool.apply_async(_slow_create_run_folder, (run_folder, run_folder_lag))

            if run_folder_lag > 10:
                with pytest.raises(FileNotFoundError, match=f"Timed out waiting for {run_folder}"):
                    res_scan = pool.apply_async(
                        start_watcher,
                        (
                            run_folder,
                            log_out,
                            fov_callback,
                            run_callback,
                            intermediate_callback,
                            10,
                            1,
                            SLOW_COPY_INTERVAL_S,
                        ),
                    )

                    res_scan.get()
            else:
                res_scan = pool.apply_async(
                    start_watcher,
                    (
                        run_folder,
                        log_out,
                        fov_callback,
                        run_callback,
                        intermediate_callback,
                        10,
                        1,
                        SLOW_COPY_INTERVAL_S,
                    ),
                )

                try:
                    res_scan.get(timeout=7)
                except TimeoutError:
                    return


@patch("toffy.watcher_callbacks.visualize_qc_metrics", side_effect=mock_visualize_qc_metrics)
@patch("toffy.watcher_callbacks.visualize_mph", side_effect=mock_visualize_mph)
@pytest.mark.parametrize("add_blank", [False, True])
@pytest.mark.parametrize("temp_bin", [False, True])
@parametrize_with_cases(
    "run_cbs,int_cbs,fov_cbs,kwargs,validators,watcher_start_lag,existing_data", cases=WatcherCases
)
def test_watcher(
    mock_viz_qc,
    mock_viz_mph,
    run_cbs,
    int_cbs,
    fov_cbs,
    kwargs,
    validators,
    watcher_start_lag,
    existing_data,
    add_blank,
    temp_bin,
):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiff_out_dir = os.path.join(tmpdir, "cb_0", RUN_DIR_NAME)
            qc_out_dir = os.path.join(tmpdir, "cb_1", RUN_DIR_NAME)
            mph_out_dir = os.path.join(tmpdir, "cb_2", RUN_DIR_NAME)
            plot_dir = os.path.join(tmpdir, "cb_2_plots", RUN_DIR_NAME)
            pulse_out_dir = os.path.join(tmpdir, "cb_3", RUN_DIR_NAME)

            # add directories to kwargs
            kwargs["tiff_out_dir"] = tiff_out_dir
            kwargs["qc_out_dir"] = qc_out_dir
            kwargs["mph_out_dir"] = mph_out_dir
            kwargs["pulse_out_dir"] = pulse_out_dir
            kwargs["plot_dir"] = plot_dir

            # ensure warn_overwrite set to False if intermediate callbacks set, otherwise True
            kwargs["warn_overwrite"] = True if int_cbs else False

            run_data = os.path.join(tmpdir, "test_run")
            log_out = os.path.join(tmpdir, "log_output")
            os.makedirs(run_data)
            fov_callback, run_callback, intermediate_callback = build_callbacks(
                run_cbs, int_cbs, fov_cbs, **kwargs
            )
            write_json_file(
                json_path=os.path.join(run_data, "test_run.json"),
                json_object=COMBINED_RUN_JSON_SPOOF,
                encoding="utf-8",
            )

            # if existing_data set to True, test case where a FOV has already been extracted
            if existing_data[0]:
                os.makedirs(os.path.join(tiff_out_dir, "fov-2-scan-1"))
                channels_write = TEST_CHANNELS if existing_data[1] == "Full" else [TEST_CHANNELS[1]]
                for channel in channels_write:
                    random_img = np.random.rand(32, 32)
                    imsave(
                        os.path.join(tiff_out_dir, "fov-2-scan-1", f"{channel}.tiff"), random_img
                    )

                os.makedirs(qc_out_dir)
                for qcs, qcc in zip(QC_SUFFIXES, QC_COLUMNS):
                    df_qc = pd.DataFrame(
                        np.random.rand(len(TEST_CHANNELS), 3), columns=["fov", "channel", qcc]
                    )
                    df_qc["fov"] = "fov-2-scan-1"
                    df_qc["channel"] = TEST_CHANNELS
                    df_qc.to_csv(os.path.join(qc_out_dir, f"fov-2-scan-1_{qcs}.csv"), index=False)

                os.makedirs(mph_out_dir)
                df_mph = pd.DataFrame(
                    np.random.rand(1, 4), columns=["fov", "MPH", "total_count", "time"]
                )
                df_mph["fov"] = "fov-2-scan-1"
                df_mph.to_csv(os.path.join(mph_out_dir, "fov-2-scan-1-mph_pulse.csv"), index=False)

                os.makedirs(pulse_out_dir)
                df_ph = pd.DataFrame(np.random.rand(10, 3), columns=["mass", "fov", "pulse_height"])
                df_ph["fov"] = "fov-2-scan-1"
                df_ph.to_csv(
                    os.path.join(pulse_out_dir, "fov-2-scan-1_pulse_heights.csv"), index=False
                )

            # `_slow_copy_sample_tissue_data` mimics the instrument computer uploading data to the
            # client access computer.  `start_watcher` is made async here since these processes
            # wouldn't block each other in normal use
            with Pool(processes=4) as pool:
                pool.apply_async(
                    _slow_copy_sample_tissue_data,
                    (run_data, SLOW_COPY_INTERVAL_S, add_blank, temp_bin),
                )
                time.sleep(watcher_start_lag)

                watcher_warnings = []
                if not add_blank:
                    watcher_warnings.append(
                        r"Re-extracting incompletely extracted FOV fov-1-scan-1"
                    )
                if existing_data[0] and existing_data[1] == "Full":
                    watcher_warnings.append(r"already extracted for FOV fov-2-scan-1")

                if len(watcher_warnings) > 0:
                    with pytest.warns(UserWarning, match="|".join(watcher_warnings)):
                        res_scan = pool.apply_async(
                            start_watcher,
                            (
                                run_data,
                                log_out,
                                fov_callback,
                                run_callback,
                                intermediate_callback,
                                2700,
                                1,
                                SLOW_COPY_INTERVAL_S,
                            ),
                        )

                        res_scan.get()
                else:
                    res_scan = pool.apply_async(
                        start_watcher,
                        (
                            run_data,
                            log_out,
                            fov_callback,
                            run_callback,
                            intermediate_callback,
                            2700,
                            1,
                            SLOW_COPY_INTERVAL_S,
                        ),
                    )

                    res_scan.get()

            with open(os.path.join(Path(__file__).parents[1], "pytest.txt")) as f:
                logtxt = f.read()
                try:
                    assert add_blank == ("non-zero file size..." in logtxt)
                except AssertionError as e:
                    _reset_logging_file()
                    raise AssertionError(e)

            fovs = [
                bin_file.split(".")[0]
                for bin_file in sorted(io_utils.list_files(COMBINED_DATA_PATH, substrs=[".bin"]))
            ]

            # callbacks are not performed on moly points, remove fov-3-scan-1
            bad_fovs = [fovs[-2]]
            fovs = fovs[:-2] + [fovs[-1]]

            # callbacks are not performed for skipped fovs, remove blank fov (fov-1-scan-1)
            if add_blank:
                bad_fovs.append(fovs[0])
                fovs = fovs[1:]

            # extract tiffs check
            print("TIFF validator check")
            validators[0](os.path.join(tmpdir, "cb_0", RUN_DIR_NAME), fovs, bad_fovs)
            if kwargs["extract_prof"]:
                try:
                    validators[0](
                        os.path.join(tmpdir, "cb_0", RUN_DIR_NAME + "_proficient"), fovs, bad_fovs
                    )
                except AssertionError as e:
                    _reset_logging_file()
                    raise AssertionError(e)
            else:
                try:
                    assert not os.path.exists(
                        os.path.join(tmpdir, "cb_0", RUN_DIR_NAME) + "_proficient"
                    )
                except AssertionError as e:
                    _reset_logging_file()
                    raise AssertionError(e)

            # qc check
            print("QC check")
            try:
                validators[1](os.path.join(tmpdir, "cb_1", RUN_DIR_NAME), fovs, bad_fovs)
            except AssertionError as e:
                _reset_logging_file()
                raise AssertionError(e)

            # mph check
            print("MPH check")
            try:
                validators[2](
                    os.path.join(tmpdir, "cb_2", RUN_DIR_NAME),
                    os.path.join(tmpdir, "cb_2_plots", RUN_DIR_NAME),
                    fovs,
                    bad_fovs,
                )
            except AssertionError as e:
                _reset_logging_file()
                raise AssertionError(e)

            # stitch images check
            print("Stitch images check")
            try:
                validators[3](
                    os.path.join(tmpdir, "cb_0", RUN_DIR_NAME, f"{RUN_DIR_NAME}_stitched")
                )
            except AssertionError as e:
                _reset_logging_file()
                raise AssertionError(e)

            # pulse heights check
            print("Pulse heights check")
            try:
                validators[4](os.path.join(tmpdir, "cb_3", RUN_DIR_NAME), fovs, bad_fovs)
            except AssertionError as e:
                _reset_logging_file()
                raise AssertionError(e)

            _reset_logging_file()

    except OSError:
        warnings.warn("Temporary file cleanup was incomplete.")


def test_watcher_missing_fovs():
    with tempfile.TemporaryDirectory() as tmpdir:
        # add extra fov to run file
        large_run_json_spoof = COMBINED_RUN_JSON_SPOOF.copy()
        large_run_json_spoof["fovs"] = COMBINED_RUN_JSON_SPOOF["fovs"] + [
            {
                "runOrder": 5,
                "scanCount": 1,
                "frameSizePixels": {"width": 32, "height": 32},
                "name": "missing_fov",
            }
        ]

        run_data = os.path.join(tmpdir, "test_run")
        os.makedirs(run_data)
        for file in io_utils.list_files(COMBINED_DATA_PATH, substrs=[".bin", ".json"]):
            shutil.copy(os.path.join(COMBINED_DATA_PATH, file), os.path.join(run_data, file))
        log_out = os.path.join(tmpdir, "log_output")
        fov_callback, run_callback, intermediate_callback = build_callbacks(
            run_callbacks=["check_missing_fovs"],
            intermediate_callbacks=[],
            fov_callbacks=[],
        )

        write_json_file(
            json_path=os.path.join(run_data, "test_run.json"),
            json_object=large_run_json_spoof,
            encoding="utf-8",
        )

        # watcher should raise warning for missing fov data (and not hang waiting for new file)
        with pytest.warns(
            warnings.warn("The following FOVs were not processed due to missing/empty/late files:"),
        ):
            start_watcher(
                run_data,
                log_out,
                fov_callback,
                run_callback,
                intermediate_callback,
                completion_check_time=1,
                zero_size_timeout=1,
                watcher_timeout=1,
            )
