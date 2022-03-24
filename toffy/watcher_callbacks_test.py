import os
import tempfile
import pytest
from pytest_cases import parametrize_with_cases

from mibi_bin_tools import io_utils

from toffy import watcher_callbacks
from toffy.test_utils import (
    ExtractionQCGenerationCases,
    ExtractionQCCallCases,
    check_extraction_dir_structure,
    check_qc_dir_structure,
)


@parametrize_with_cases('panel,  kwargs', cases=ExtractionQCGenerationCases,
                        has_tag='extract')
@pytest.mark.parametrize('extraction_dir_name', ['', 'extracted'])
@parametrize_with_cases('data_path',  cases=ExtractionQCCallCases)
def test_build_extraction_callback(panel, extraction_dir_name, kwargs, data_path):

    intensities = kwargs.get('intensities', False)

    # test cb generates w/o errors
    cb = watcher_callbacks.build_extract_callback(panel, extraction_dir_name,
                                                  **kwargs)

    point_names = io_utils.list_files(data_path, substrs=['bin'])
    point_names = [name.split('.')[0] for name in point_names]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for name in point_names:
            cb(data_path, name, tmp_dir)

        ext_path = os.path.join(tmp_dir, extraction_dir_name)

        # just check SMA
        check_extraction_dir_structure(ext_path, point_names, ['SMA'], intensities)

        check_qc_dir_structure(data_path, point_names)


@parametrize_with_cases('panel,  kwargs', cases=ExtractionQCGenerationCases,
                        has_tag='qc')
@parametrize_with_cases('data_path',  cases=ExtractionQCCallCases)
def test_build_qc_callback(panel, kwargs, data_path):

    # test cb generates w/o errors
    cb = watcher_callbacks.build_qc_callback(panel, **kwargs)

    point_names = io_utils.list_files(data_path, substrs=['bin'])
    point_names = [name.split('.')[0] for name in point_names]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for name in point_names:
            cb(data_path, name, tmp_dir)

        check_qc_dir_structure(data_path, point_names)
