import os
import re
import warnings

import natsort as ns
import numpy as np
import pandas as pd
from alpineer import io_utils, load_utils
from mibi_bin_tools import bin_files

from toffy.json_utils import check_for_empty_files, list_moly_fovs, read_json_file


def extract_missing_fovs(
    bin_file_dir,
    extraction_dir,
    panel,
    extract_intensities=["Au", "chan_39"],
    replace=True,
):
    """Check for already extracted FOV bin files, and extract the remaining (excluding moly fovs
       and fovs with empty json files).

    Args:
        bin_file_dir (str): path to directory containing the bin and json files
        extraction_dir (str): path to directory of already extracted FOVs
        panel (pd.DataFrame): file defining the panel info for bin file extraction
        extract_intensities (bool): whether to extract intensities from the bin files
        replace (bool): whether to replace pulse images with intensity
    """
    # retrieve all fov names from base_dir and extracted fovs from extraction_dir
    fovs = io_utils.remove_file_extensions(io_utils.list_files(bin_file_dir, substrs=".bin"))
    extracted_fovs = io_utils.list_folders(extraction_dir, substrs="fov")

    # filter out empty json file fovs
    empty_fovs = check_for_empty_files(bin_file_dir)
    if empty_fovs:
        fovs = list(set(fovs).difference(empty_fovs))

    if len(fovs) == 0:
        warnings.warn(f"No viable bin files were found in {bin_file_dir}", Warning)
        return

    # check for moly fovs
    moly_fovs = list_moly_fovs(bin_file_dir, fovs)

    if extracted_fovs:
        print(
            "Skipping the following previously extracted FOVs: ",
            ", ".join(extracted_fovs),
        )
    if moly_fovs:
        print("Moly FOVs which will not be extracted: ", ", ".join(moly_fovs))
    if empty_fovs:
        print(
            "FOVs with empty json files which will not be extracted: ",
            ", ".join(empty_fovs),
        )

    # extract missing fovs to extraction_dir
    non_moly_fovs = list(set(fovs).difference(moly_fovs))
    missing_fovs = list(set(non_moly_fovs).difference(extracted_fovs))
    missing_fovs = ns.natsorted(missing_fovs)

    if missing_fovs:
        print(f"Found {len(missing_fovs)} FOVs to extract.")
        bin_files.extract_bin_files(
            bin_file_dir,
            extraction_dir,
            include_fovs=missing_fovs,
            panel=panel,
            intensities=extract_intensities,
            replace=replace,
        )
        print("Extraction completed!")
    else:
        warnings.warn(f"No viable bin files were found in {bin_file_dir}", Warning)


def incomplete_fov_check(
    bin_file_dir, extraction_dir, num_rows=10, num_channels=5, signal_percent=0.02
):
    """Read in the supplied number tiff files for each FOV to check for incomplete images.

    Args:
        bin_file_dir (str): directory containing the run json file
        extraction_dir (str): directory containing the extracted tifs
        num_rows (int): number of bottom rows of the images to check for zero values
        num_channels (int): number of channel images to check per FOV
        signal_percent (float): min amount of non-zero signal required for complete FOVs

    Raises:
        Warning if any FOVs have only partially generated images
    """
    io_utils.validate_paths([bin_file_dir, extraction_dir])

    # read in json file to get custom fov names
    run_name = os.path.basename(bin_file_dir)
    run_file_path = os.path.join(bin_file_dir, run_name + ".json")
    run_metadata = read_json_file(run_file_path, encoding="utf-8")

    # TODO: list_folders does not handle cases such as "fov0" correctly
    # need to add a fix in to alpineer to deal with this
    fovs = [f for f in os.listdir(extraction_dir) if "fov" in f]

    # NOTE: the specifically-named "Au" channel is no longer be assumed to be present
    # simply check the first "num_channels" channels found
    channels = io_utils.list_files(os.path.join(extraction_dir, fovs[0]), ".tiff")
    channels_subset = channels[:num_channels]

    incomplete_fovs = {}
    for fov in fovs:
        # load in channel images
        img_data = load_utils.load_imgs_from_tree(
            extraction_dir, fovs=[fov], channels=channels_subset
        )
        row_index = img_data.shape[1] - num_rows
        img_bottoms = img_data[0, row_index:, :, :]

        # check percentage of non-zero pixels in the bottom of the image
        total_pixels = img_data.shape[1] * num_rows * num_channels
        if np.count_nonzero(img_bottoms) / total_pixels < signal_percent:
            i = re.findall(r"\d+", fov)[0]
            custom_name = run_metadata["fovs"][int(i) - 1]["name"]
            incomplete_fovs[fov] = custom_name

    if incomplete_fovs:
        incomplete_fovs = pd.DataFrame(incomplete_fovs, index=[0]).T
        incomplete_fovs.columns = ["fov_name"]
        warnings.warn(
            f"\nThe following FOVs have less than {signal_percent*100}% positive signal on average"
            " at the bottom of the tiff and may have been only partially imaged: \n"
            f"{incomplete_fovs}\n"
        )
