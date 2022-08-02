import os
import tempfile
import pandas as pd
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, call

from toffy import bin_extraction
from ark.utils import test_utils
from mibi_bin_tools import io_utils, bin_files


@patch('builtins.print')
def test_extract_missing_fovs(mocked_print):
    bin_file_dir = os.path.join(Path(__file__).parent, "data", "tissue")
    moly_bin_file_dir = os.path.join(Path(__file__).parent, "data", "moly")

    panel = pd.DataFrame([{
        'Mass': 98,
        'Target': None,
        'Start': 97.5,
        'Stop': 98.5,
    }])

    # test that it does not re-extract fovs
    with tempfile.TemporaryDirectory() as extraction_dir:
        os.makedirs(os.path.join(extraction_dir, 'fov-1-scan-1'))
        bin_extraction.extract_missing_fovs(bin_file_dir, extraction_dir,
                                            panel, extract_intensities=False)

        assert mocked_print.mock_calls == \
               [call('Skipping the following previously extracted FOVs: ', 'fov-1-scan-1'),
                call('Found 1 FOVs to extract.')]
    mocked_print.reset_mock()

    # test that it does not re-extract fovs and no moly extraction
    with tempfile.TemporaryDirectory() as combined_bin_file_dir:
        # combine two regular fov and one moly fov files into single directory
        for file_name in io_utils.list_files(moly_bin_file_dir):
            shutil.copy(os.path.join(moly_bin_file_dir, file_name),
                        os.path.join(combined_bin_file_dir, file_name))
            if '.json' in file_name:
                ext = '.json'
            elif '.bin' in file_name:
                ext = '.bin'
            old_name = os.path.join(combined_bin_file_dir, file_name)
            new_name = os.path.join(combined_bin_file_dir, 'moly_fov' + ext)
            os.rename(old_name, new_name)

        for file_name in io_utils.list_files(bin_file_dir):
            shutil.copy(os.path.join(bin_file_dir, file_name),
                        os.path.join(combined_bin_file_dir, file_name))

        # create empty json fov
        test_utils._make_blank_file(combined_bin_file_dir, 'empty.bin')
        test_utils._make_blank_file(combined_bin_file_dir, 'empty.json')

        # check for correct output
        with tempfile.TemporaryDirectory() as extraction_dir:
            os.makedirs(os.path.join(extraction_dir, 'fov-1-scan-1'))
            bin_extraction.extract_missing_fovs(combined_bin_file_dir, extraction_dir,
                                                panel, extract_intensities=False)

            assert mocked_print.mock_calls == \
                   [call('Skipping the following previously extracted FOVs: ', 'fov-1-scan-1'),
                    call('Moly FOVs which will not be extracted: ', 'moly_fov'),
                    call('FOVs with empty json files which will not be extracted: ', 'empty'),
                    call('Found 1 FOVs to extract.')]

            # when given empty fov files will raise a warning
            with pytest.warns(UserWarning, match="The following FOVs have empty json files"):
                bin_extraction.extract_missing_fovs(combined_bin_file_dir, extraction_dir,
                                                    panel, extract_intensities=False)

            # test that neither moly nor empty fov were extracted
            assert io_utils.list_folders(extraction_dir) == ['fov-2-scan-1', 'fov-1-scan-1']

    # test successful extraction of fovs
    with tempfile.TemporaryDirectory() as extraction_dir:
        mocked_print.reset_mock()
        bin_extraction.extract_missing_fovs(bin_file_dir, extraction_dir,
                                            panel, extract_intensities=False)
        fovs = ['fov-1-scan-1', 'fov-2-scan-1']
        fovs_extracted = io_utils.list_folders(extraction_dir)

        # test no extra print statements
        assert mocked_print.mock_calls == [call('Found 2 FOVs to extract.')]

        # check both fovs were extracted
        assert fovs.sort() == fovs_extracted.sort()

        # all fovs extracted already will raise a warning
        with pytest.warns(UserWarning, match="No viable bin files were found"):
            bin_extraction.extract_missing_fovs(bin_file_dir, extraction_dir,
                                                panel, extract_intensities=False)

        # when given only moly fovs will raise a warning
        with pytest.warns(UserWarning, match="No viable bin files were found"):
            bin_extraction.extract_missing_fovs(moly_bin_file_dir, extraction_dir,
                                                panel, extract_intensities=False)