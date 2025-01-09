import logging
import os
import threading
import time
import warnings
from datetime import datetime
from multiprocessing import Lock
from pathlib import Path
from typing import Callable, Tuple, Union

import natsort as ns
import numpy as np
from matplotlib import pyplot as plt
from watchdog.events import (
    DirCreatedEvent,
    FileCreatedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from toffy.json_utils import read_json_file


class RunStructure:
    """Expected bin and json files.

    Attributes:
        fov_progress (dict): Whether or not an expected file has been created
    """

    def __init__(self, run_folder: str, fov_timeout: int = 7800):
        """Initializes RunStructure by parsing run json within provided run folder.

        Args:
            run_folder (str):
                path to run folder
            fov_timeout (int):
                number of seconds to wait for non-null filesize before raising an error
        """
        self.timeout = fov_timeout
        self.fov_progress = {}
        self.processed_fovs = []
        self.moly_points = []

        # find run .json and get parameters
        run_name = Path(run_folder).parts[-1]
        run_metadata = read_json_file(
            os.path.join(run_folder, f"{run_name}.json"), encoding="utf-8"
        )

        # parse run_metadata and populate expected structure
        for fov in run_metadata["rois"][0].get("fovs", ()):
            run_order = fov.get("runOrder", -1)
            scan = fov.get("scanCount", -1)
            if run_order * scan < 0:
                raise KeyError(f"Could not locate keys in {run_folder}.json")

            # scan 2's don't contain significant imaging data per new MIBI specs
            fov_name = f"fov-0{run_order}-scan-1" if run_order < 10 else f"fov-0{run_order}-scan-1"
            if fov.get("standardTarget", "") == "Molybdenum Foil":
                self.moly_points.append(fov_name)

            self.fov_progress[fov_name] = {"json": False, "bin": False}

        # get the highest FOV number, needed for checking if final FOV processed
        # NOTE: only scan-1 files considered, so len is good
        self.highest_fov = len(self.fov_progress)

    def check_run_condition(self, path: str) -> Tuple[bool, str]:
        """Checks if all requisite files exist and are complete.

        Args:
            path (str):
                path to expected file

        Raises:
            TimeoutError

        Returns:
            (bool, str):
                whether or not both json and bin files exist, as well as the name of the point
        """
        filename = Path(path).parts[-1]

        # if filename starts with a '.' (temp file), it should be ignored
        if filename[0] == ".":
            return False, ""

        # filename is not corrct format of fov.bin or fov.json
        if len(filename.split(".")) != 2:
            return False, ""

        fov_name, extension = filename.split(".")

        # path no longer valid
        if not os.path.exists(path):
            warnings.warn(
                f"{path} doesn't exist but was recently created. This should be unreachable...",
                Warning,
            )
            return False, ""

        # avoids repeated processing in case of duplicated events
        if fov_name in self.processed_fovs:
            return False, fov_name

        # does not process moly points, but add to process list to ensure proper incrementing
        if fov_name in self.moly_points:
            self.processed(fov_name)
            self.fov_progress[fov_name]["json"] = True
            self.fov_progress[fov_name]["bin"] = True
            return False, fov_name

        wait_time = 0
        if fov_name in self.fov_progress:
            if extension in self.fov_progress[fov_name]:
                while os.path.getsize(path) == 0:
                    # consider timed out fovs complete
                    if wait_time >= self.timeout:
                        del self.fov_progress[fov_name]
                        raise TimeoutError(f"timed out waiting for {path}...")

                    time.sleep(self.timeout / 10)
                    wait_time += self.timeout / 10

                self.fov_progress[fov_name][extension] = True

            if all(self.fov_progress[fov_name].values()):
                return True, fov_name

        elif extension == "bin":
            warnings.warn(f"Found unexpected bin file, {path}...", Warning)
            return False, ""

        return False, fov_name

    def processed(self, fov_name: str):
        """Notifies run structure that fov has been processed.

        Args:
            fov_name (str):
                Name of FoV
        """
        self.processed_fovs.append(fov_name)

    def check_fov_progress(self) -> dict:
        """Condenses internal dictionary to show which fovs have finished.

        Returns:
            dict
        """
        all_fovs = self.fov_progress.keys()
        moly_fovs = self.moly_points
        necessary_fovs = list(set(all_fovs).difference(moly_fovs))

        return {k: all(self.fov_progress[k].values()) for k in necessary_fovs}


class FOV_EventHandler(FileSystemEventHandler):
    """File event handler for FOV files.

    Attributes:
        run_folder (str):
            path to run folder
        watcher_out (str):
            folder to save all callback results + log file
        run_structure (RunStructure):
            expected run file structure + fov_progress status
        fov_callback (Callable[[str, str], None]):
            callback to run on each fov
        run_callback (Callable[[None], None]):
            callback to run over the entire run
    """

    instance_count = 0

    def __init__(
        self,
        run_folder: str,
        log_folder: str,
        fov_callback: Callable[[str, str], None],
        run_callback: Callable[[str], None],
        intermediate_callback: Callable[[str], None] = None,
        fov_timeout: int = 7800,
        watcher_timeout: int = 3 * 7800,
    ):
        """Initializes FOV_EventHandler.

        Args:
            run_folder (str):
                path to run folder
            log_folder (str):
                path to save outputs to
            fov_callback (Callable[[str, str], None]):
                callback to run on each fov
            run_callback (Callable[[None], None]):
                callback to run over the entire run
            intermediate_callback (Callable[[None], None]):
                run callback overriden to run on each fov
            fov_timeout (int):
                number of seconds to wait for non-null filesize before raising an error
            watcher_timeout (int):
                length to wait for new file generation before timing out
        """
        super().__init__()
        self.run_folder = run_folder

        self.last_event_time = datetime.now()
        self.timer_thread = threading.Thread(
            target=self.file_timer, args=(fov_timeout, watcher_timeout)
        )
        self.timer_thread.daemon = True
        self.timer_thread.start()

        self.log_path = os.path.join(log_folder, f"{Path(run_folder).parts[-1]}_log.txt")
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logging.basicConfig(
            level=logging.INFO,
            filename=self.log_path,
            filemode="a",
            format="%(name)s - %(levelname)s - %(message)s",
        )

        # create run structure
        self.run_structure = RunStructure(run_folder, fov_timeout=fov_timeout)

        self.fov_func = fov_callback
        self.run_func = run_callback
        self.inter_func = intermediate_callback
        self.inter_return_vals = None
        self.lock = Lock()
        self.last_fov_num_processed = 0
        self.all_fovs_complete = False

        for root, dirs, files in os.walk(run_folder):
            for name in ns.natsorted(files):
                # NOTE: don't call with check_last_fov to prevent duplicate processing
                self.on_created(FileCreatedEvent(os.path.join(root, name)), check_last_fov=False)

        # edge case if the last FOV gets written during the preprocessing stage
        # simulate a trigger using the first FOV file
        self._check_last_fov(
            os.path.join(root, list(self.run_structure.fov_progress.keys())[0] + ".bin")
        )

    def _check_fov_status(self, path: str):
        """Verifies the status of the file written at `path`.

        Args:
            path (str):
                The path to check the status of
        Returns:
            Tuple[Optional[str], Optional[str]]:
                The status of `path`, as well as the corresponding FOV name
        """
        try:
            fov_ready, point_name = self.run_structure.check_run_condition(path)
            return fov_ready, point_name
        except TimeoutError as timeout_error:
            print(f"Encountered TimeoutError error: {timeout_error}")
            logging.warning(
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                f"{path} never reached non-zero file size...\n"
            )

            # these count as processed FOVs, so increment
            self.last_fov_num_processed += 1

            # these count as processed FOVs, so mark as processed
            self.run_structure.processed(Path(path).parts[-1].split(".")[0])
            self.check_complete()

            return None, None

    def _generate_callback_data(self, point_name: str, overwrite: bool):
        """Runs the `fov_func` and `inter_func` if applicable for a FOV.

        Args:
            point_name (str):
                The name of the FOV to run FOV (and intermediate if applicable) callbacks on
            overwrite (bool):
                Forces an overwrite of already existing data, needed if a FOV needs re-extraction
        """
        print(f"Discovered {point_name}, beginning per-fov callbacks...")
        logging.info(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- Extracting {point_name}\n')
        logging.info(
            f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
            f"Running {self.fov_func.__name__} on {point_name}\n"
        )

        self.fov_func(self.run_folder, point_name, overwrite)
        self.run_structure.processed(point_name)

        if self.inter_func:
            # clear plots contained in intermediate return values if set
            if self.inter_return_vals:
                qc_plots = self.inter_return_vals.get("plot_qc_metrics", None)
                mph_plot = self.inter_return_vals.get("plot_mph_metrics", None)

                if qc_plots or mph_plot:
                    plt.cla()
                    plt.clf()
                    plt.close("all")

            self.inter_return_vals = self.inter_func(self.run_folder)

        self.check_complete()

    def _process_missed_fovs(self, path: str):
        """Given a `path`, check if there are any missing FOVs to process before it.

        Args:
            path (str):
                The path to check for missing FOVs prior
        """
        # verify the path provided is correct .bin type, if not skip
        filename = Path(path).parts[-1]
        name_ext = filename.split(".")
        if len(name_ext) != 2 or name_ext[1] != "bin":
            return

        # NOTE: MIBI now only stores relevant data in scan 1, ignore any scans > 1
        scan_num = int(name_ext[0].split("-")[3])
        if scan_num > 1:
            return

        # retrieve the FOV number
        fov_num = int(name_ext[0].split("-")[1])

        # a difference of 1 from last_fov_num_processed means there are no in-between FOVs
        # set to <= for safety (this should never happen in theory)
        if fov_num - 1 <= self.last_fov_num_processed:
            return

        # NOTE: from observation, only the most recent FOV will ever be in danger of timing out
        # so all the FOVs processed in this function should already be fully processed
        start_index = self.last_fov_num_processed + 1 if self.last_fov_num_processed else 1
        for i in np.arange(start_index, fov_num):
            fov_name = f"fov-{i}-scan-1"
            fov_bin_file = os.path.join(self.run_folder, fov_name + ".bin")
            fov_json_file = os.path.join(self.run_folder, fov_name + ".json")

            # this can happen if there's a lag copying files over
            while not os.path.exists(fov_bin_file) and not os.path.exists(fov_json_file):
                time.sleep(60)

            self._fov_callback_driver(os.path.join(self.run_folder, fov_name + ".bin"))
            self._fov_callback_driver(os.path.join(self.run_folder, fov_name + ".json"))

    def _check_last_fov(self, path: str):
        """Checks if the last FOV's data has been written.

        Needed because there won't be any more file triggers after this happens.

        Args:
            path (str):
                The path that triggers this call. Used only for formatting purposes.
        """
        # define the name of the last FOV
        last_fov = f"fov-{self.run_structure.highest_fov}-scan-1"
        last_fov_bin = f"{last_fov}.bin"
        last_fov_json = f"{last_fov}.json"

        # if the last FOV has been written, then process everything up to that if necessary
        # NOTE: don't process if it has already been written
        bin_dir = str(Path(path).parents[0])
        last_fov_is_processed = self.last_fov_num_processed == self.run_structure.highest_fov
        last_fov_data_exists = os.path.exists(
            os.path.join(bin_dir, last_fov_bin)
        ) and os.path.exists(os.path.join(bin_dir, last_fov_json))

        if not last_fov_is_processed and last_fov_data_exists:
            start_index = self.last_fov_num_processed + 1 if self.last_fov_num_processed else 1
            for i in np.arange(start_index, self.run_structure.highest_fov):
                fov_name = f"fov-{i}-scan-1"
                fov_bin_file = os.path.join(self.run_folder, fov_name + ".bin")
                fov_json_file = os.path.join(self.run_folder, fov_name + ".json")

                # this can happen if there's a lag copying files over
                while not os.path.exists(fov_bin_file) and not os.path.exists(fov_json_file):
                    time.sleep(60)

                self._fov_callback_driver(os.path.join(self.run_folder, fov_name + ".bin"))
                self._fov_callback_driver(os.path.join(self.run_folder, fov_name + ".json"))

            # process the final bin file
            self._fov_callback_driver(os.path.join(self.run_folder, last_fov_bin))
            self._fov_callback_driver(os.path.join(self.run_folder, last_fov_json))

            # explicitly call check_complete to start run callbacks, since all FOVs are done
            self.check_complete()

    def _check_bin_updates(self):
        """Checks for, and re-runs if necessary, any incompletely extracted FOVs."""
        for fov in self.run_structure.fov_progress:
            # skip moly points
            if fov in self.run_structure.moly_points:
                continue

            fov_bin_path = os.path.join(self.run_folder, fov + ".bin")
            fov_json_path = os.path.join(self.run_folder, fov + ".json")

            # if .bin file ctime > .json file ctime, incomplete extraction, need to re-extract
            fov_bin_create = Path(fov_bin_path).stat().st_ctime
            fov_json_create = Path(fov_json_path).stat().st_ctime

            if fov_bin_create > fov_json_create:
                warnings.warn(f"Re-extracting incompletely extracted FOV {fov}")
                logging.info(
                    f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- Re-extracting {fov}\n'
                )
                logging.info(
                    f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                    f"Running {self.fov_func.__name__} on {fov}\n"
                )

                # since reprocessing needed, remove from self.processed_fovs
                self.run_structure.processed_fovs.remove(fov)

                # re-extract the .bin file
                # NOTE: since no more FOVs are being written, last_fov_num_processed is irrelevant
                self._fov_callback_driver(fov_bin_path, overwrite=True)

    def _fov_callback_driver(self, file_trigger: str, overwrite: bool = False):
        """The FOV and intermediate-level callback motherbase for a single .bin file.

        Args:
            file_trigger (str):
                The file that gets caught by the watcher to throw into the pipeline
            overwrite (bool):
                Forces an overwrite of already existing data, needed if a FOV needs re-extraction
        """
        # check if what's created is in the run structure
        fov_ready, point_name = self._check_fov_status(file_trigger)

        if fov_ready:
            self._generate_callback_data(point_name, overwrite=overwrite)

        # needs to update if .bin file processed OR new moly point detected
        is_moly = point_name in self.run_structure.moly_points
        is_processed = point_name in self.run_structure.processed_fovs
        if fov_ready or (is_moly and not is_processed):
            self.last_fov_num_processed += 1

    def _run_callbacks(
        self, event: Union[DirCreatedEvent, FileCreatedEvent, FileMovedEvent], check_last_fov: bool
    ):
        """The pipeline runner, invoked when a new event is seen.

        Args:
            event (Union[DirCreatedEvent, FileCreatedEvent, FileMovedEvent]):
                The type of event seen. File/directory creation and file renaming are supported.
            check_last_fov (bool):
                Whether to invoke `_check_last_fov` on the event
        """
        if type(event) in [DirCreatedEvent, FileCreatedEvent]:
            file_trigger = event.src_path
        else:
            file_trigger = event.dest_path

        # process any FOVs that got missed on the previous iteration of on_created/on_moved
        self._process_missed_fovs(file_trigger)

        # run the fov callback process on the file
        self._fov_callback_driver(file_trigger)

        if check_last_fov:
            self._check_last_fov(file_trigger)

    def on_created(self, event: FileCreatedEvent, check_last_fov: bool = True):
        """Handles file creation events.

        If FOV structure is completed, the fov callback, `self.fov_func` will be run over the data.
        This function is automatically called; users generally shouldn't call this function

        Args:
            event (FileCreatedEvent): file creation event
            check_last_fov (bool): whether to check if the last fov has been processed
        """
        # reset event creation time
        current_time = datetime.now()
        self.last_event_time = current_time

        # this happens if _check_last_fov gets called by a prior FOV, no need to reprocess
        if self.last_fov_num_processed == self.run_structure.highest_fov:
            return

        with self.lock:
            super().on_created(event)
            self._run_callbacks(event, check_last_fov)

    def file_timer(self, fov_timeout, watcher_timeout):
        """Checks time since last file was generated.

        Args:
            fov_timeout (int):
                how long to wait for fov data to be generated once file detected
            watcher_timeout (int):
                length to wait for new file generation before timing out
        """
        while True:
            current_time = datetime.now()
            time_elapsed = (current_time - self.last_event_time).total_seconds()

            # 3 fov cycles and no new files --> timeout
            if time_elapsed > watcher_timeout:
                fov_num = self.last_fov_num_processed
                fov_name = list(self.run_structure.fov_progress.keys())[fov_num]
                print(f"Timed out waiting for {fov_name} files to be generated.")
                logging.info(
                    f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- Timed out'
                    f"waiting for {fov_name} files to be generated.\n"
                )
                logging.info(
                    f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                    f"Running {self.run_func.__name__} on FOVs\n"
                )

                # mark remaining fovs as completed to exit watcher
                for fov_name in list(self.run_structure.fov_progress.keys()):
                    self.run_structure.fov_progress[fov_name] = {"json": True, "bin": True}

                # trigger run callbacks
                self.run_func(self.run_folder)
                break
            time.sleep(fov_timeout)

    def on_moved(self, event: FileMovedEvent, check_last_fov: bool = True):
        """Handles file renaming events.

        If FOV structure is completed, the fov callback, `self.fov_func` will be run over the data.
        This function is automatically called; users generally shouldn't call this function

        Args:
            event (FileMovedEvent): file moved event
            check_last_fov (bool): whether to check if last fov was processed
        """
        # this happens if _check_last_fov gets called by a prior FOV, no need to reprocess
        if self.last_fov_num_processed == self.run_structure.highest_fov:
            return

        with self.lock:
            super().on_moved(event)
            self._run_callbacks(event, check_last_fov)

    def check_complete(self):
        """Checks run structure fov_progress status.

        If run is complete, all callbacks in `per_run` will be run over the whole run.

        NOTE: bin files that had new data written will first need to be re-extracted.
        """
        if all(self.run_structure.check_fov_progress().values()) and not self.all_fovs_complete:
            self.all_fovs_complete = True
            self._check_bin_updates()
            logging.info(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- All FOVs finished\n')
            logging.info(
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                f"Running {self.run_func.__name__} on whole run\n"
            )

            self.run_func(self.run_folder)


def start_watcher(
    run_folder: str,
    log_folder: str,
    fov_callback: Callable[[str, str], None],
    run_callback: Callable[[None], None],
    intermediate_callback: Callable[[str, str], None] = None,
    run_folder_timeout: int = 5400,
    completion_check_time: int = 30,
    zero_size_timeout: int = 7800,
    watcher_timeout: int = 3 * 7800,
):
    """Passes bin files to provided callback functions as they're created.

    Args:
        run_folder (str):
            path to run folder
        log_folder (str):
            where to create log file
        fov_callback (Callable[[str, str], None]):
            function to run on each completed fov. assemble this using
            `watcher_callbacks.build_callbacks`
        run_callback (Callable[[None], None]):
            function ran once the run has completed. assemble this using
            `watcher_callbacks.build_callbacks`
        intermediate_callback (Callable[[None], None]):
            function defined as run callback overriden as fov callback. assemble this using
            `watcher_callbacks.build_callbacks`
        run_folder_timeout (int):
            how long to wait for the run folder to appear before timing out, in seconds.
            note that the watcher cannot begin until this run folder appears.
        completion_check_time (int):
            how long to wait before checking watcher completion, in seconds.
            note, this doesn't effect the watcher itself, just when this wrapper function exits.
        zero_size_timeout (int):
            number of seconds to wait for non-zero file size
        watcher_timeout (int):
            number of seconds to wait for new fov file generation
    """
    # if the run folder specified isn't already there, ask the user to explicitly confirm the name
    if not os.path.exists(run_folder):
        warnings.warn(
            f"Waiting for {run_folder}. Please first double check that your run data "
            "doesn't already exist under a slightly different name in D:\\Data. "
            "Sometimes, the CACs change capitalization or add extra characters to the run folder. "
            "If this happens, stop the watcher and update the run_name variable in the notebook "
            "before trying again."
        )

    # allow the watcher to poll the run folder until it appears or times out
    run_folder_wait_time = 0
    while not os.path.exists(run_folder) and run_folder_wait_time < run_folder_timeout:
        time.sleep(run_folder_timeout / 10)
        run_folder_wait_time += run_folder_timeout / 10

    if run_folder_wait_time == run_folder_timeout:
        raise FileNotFoundError(
            f"Timed out waiting for {run_folder}. Make sure the run_name variable in the notebook "
            "matches up with the run folder name in D:\\Data, or try again a few minutes later "
            "if the run folder still hasn't shown up."
        )

    observer = Observer()
    event_handler = FOV_EventHandler(
        run_folder,
        log_folder,
        fov_callback,
        run_callback,
        intermediate_callback,
        zero_size_timeout,
        watcher_timeout,
    )
    observer.schedule(event_handler, run_folder, recursive=True)
    observer.start()

    try:
        while not all(event_handler.run_structure.check_fov_progress().values()):
            time.sleep(completion_check_time)
    except KeyboardInterrupt:
        observer.stop()

    observer.stop()
    observer.join()
