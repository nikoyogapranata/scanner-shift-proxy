import subprocess

import numpy as np

from ssp import compact, data, fetch, mock


def test_compact_copy_matches_original_and_unpacks_from_drive(tmp_path):
    src, out = tmp_path / "orig", tmp_path / "work" / "mnms_compact"
    mock.make(src, {"B": 3, "A": 1, "C": 1, "D": 1}, fov_mm=90, slices=3, frames=6)
    meta = compact.build(src, out, verify=6)
    assert len(meta) == 6 and set(meta["ED"]) == {0} and set(meta["ES"]) == {1}

    orig = data.discover_cases(src).set_index("patient", drop=False)
    comp = data.discover_cases(out).set_index("patient", drop=False)
    assert set(orig.index) == set(comp.index)
    for code in comp.index:
        for phase in data.PHASES:
            a, b = data.load_volume(orig.loc[code], phase), data.load_volume(comp.loc[code], phase)
            assert np.array_equal(a[1], b[1]) and np.allclose(a[0], b[0]) and np.allclose(a[2], b[2])

    # A later session: the tar on "Drive" is unpacked instead of rebuilding.
    drive = tmp_path / "drive"
    drive.mkdir()
    subprocess.run(["tar", "-cf", str(drive / fetch.TAR_NAME), "-C", str(out.parent), out.name], check=True)
    fresh = tmp_path / "session2" / "mnms_compact"
    root = fetch.ensure_data(str(drive), str(fresh))
    assert len(data.discover_cases(fresh)) == 6 and root == str(fresh)
    assert fetch.ensure_data(str(drive), str(fresh)) == str(fresh)   # already unpacked: no work
