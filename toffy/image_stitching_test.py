import os
import tempfile
import pytest
import json

from toffy import image_stitching, json_utils
from ark.utils import io_utils, test_utils


def test_get_max_img_size():

    RUN_JSON_SPOOF = {
        'fovs': [
            {'runOrder': 1, 'scanCount': 1, 'frameSizePixels': {'width': 32, 'height': 32}},
            {'runOrder': 2, 'scanCount': 1, 'frameSizePixels': {'width': 16, 'height': 16}},
        ],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = os.path.join(tmp_dir, 'data', 'test_run')
        os.makedirs(test_dir)
        json_path = os.path.join(test_dir, 'test_run.json')
        json_utils.write_json_file(json_path, RUN_JSON_SPOOF)

        # test success
        max_img_size = image_stitching.get_max_img_size(test_dir)
        assert max_img_size == 32


def test_stitch_images(mocker):
    mocker.patch('toffy.image_stitching.get_max_img_size', return_value=32)

    with tempfile.TemporaryDirectory() as tmpdir:
        channel_list = ['Au', 'CD3', 'CD4', 'CD8', 'CD11c']
        stitched_tifs = ['Au.tiff', 'CD3.tiff', 'CD4.tiff', 'CD8.tiff', 'CD11c.tiff']
        fov_list = ['fov-1-scan-1', 'fov-2-scan-1', 'fov-3-scan-1']
        test_utils._write_tifs(tmpdir, fov_list, channel_list, (10, 10), '', False, int)

        # bad channel should raise an error
        with pytest.raises(ValueError, match='Not all values given in list channel inputs were '
                                             'found in list valid channels.'):
            image_stitching.stitch_images(tmpdir, tmpdir, ['Au', 'bad_channel'])

        # test successful stitching for all channels
        image_stitching.stitch_images(tmpdir, tmpdir)
        assert sorted(io_utils.list_files(os.path.join(tmpdir, 'stitched_images'))) == \
               sorted(stitched_tifs)

        # test stitching for specific channels
        image_stitching.stitch_images(tmpdir, tmpdir, ['Au', 'CD3'])
        assert sorted(io_utils.list_files(os.path.join(tmpdir, 'stitched_images'))) == \
               sorted(['Au.tiff', 'CD3.tiff'])
