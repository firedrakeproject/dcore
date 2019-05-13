import pytest
from os.path import abspath, basename, dirname, join
import subprocess
import glob
import sys


cwd = abspath(dirname(__file__))
examples_dir = join(cwd, "..", "..", "examples")
example_files = glob.glob("%s/*.py" % examples_dir)

@pytest.fixture(params=[pytest.param(f, marks=pytest.mark.xfail(reason="unreliable")) if f.endswith("hs_mountain.py") else f for f in example_files],
                ids=lambda x: basename(x))
def example_file(request):
    return abspath(request.param)


def test_example_runs(example_file, tmpdir, monkeypatch):
    # This ensures that the test writes output in a temporary
    # directory, rather than where pytest was run from.
    monkeypatch.chdir(tmpdir)
    subprocess.check_call([sys.executable, example_file, "--running-tests"])
