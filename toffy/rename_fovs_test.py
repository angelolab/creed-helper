import tempfile
import json

# from ark.utils import test_utils
from toffy import rename_fovs


def create_sample_run(name_list, run_order_list, scan_count_list, create_json=False):
    """Creates example run metadata, create temporary JSON file if required
       Args:
           lists
           create_json (bool): whether to create temporary JSON file and return the path, defaults to False

       Returns:
           dict: correctly formatted dictionary of sample run metadata
           path: the path to a temporary JSON file for the sample run
          """
    fov_list = []
    sample_run = {"fovs": fov_list}

    for name, run_order, scan_count in zip(name_list, run_order_list, scan_count_list):
        ex_fov = {
            "scanCount": scan_count,
            "runOrder": run_order,
            "name": name
        }
        fov_list.append(ex_fov)

    for fov in sample_run.get('fovs', ()):
        if fov.get('name') == 'missing':
            del fov['name']

    if create_json:
        temp = tempfile.NamedTemporaryFile(mode="w")
        json.dump(sample_run, temp)
        print(temp.name)
        return temp.name

    return sample_run


def test_check_unnamed_fovs():
    ex_name = ['MoQC', 'missing', 'tonsil_bottom', 'moly_qc_tissue', 'missing']
    ex_run_order = list(range(1, 6))
    ex_scan_count = list(range(1, 6))
    ex_run = create_sample_run(ex_name, ex_run_order, ex_scan_count)

    rename_fovs.check_unnamed_fovs(ex_run)