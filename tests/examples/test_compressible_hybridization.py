import pytest
from os.path import abspath, basename, dirname, join
import subprocess
import glob
import sys


cwd = abspath(dirname(__file__))
examples_dir = join(cwd, "..", "..", "examples")


# Examples which use the compressible solvers
compressible_examples = ["compressible_eady.py",
                         "dense_bubble.py",
                         "dry_bf_bubble.py",
                         "moist_bf_bubble.py",
                         "nh_mountain.py",
                         "rising_bubble.py",
                         "sk_hydrostatic.py",
                         "sk_linear_advection.py",
                         "sk_nonlinear.py",
                         "tracer.py",
                         "unsaturated_bubble.py"]


@pytest.fixture(params=[ex for glb in [glob.glob("%s/%s" % (examples_dir, exmp))
                                       for exmp in compressible_examples]
                        for ex in glb],
                ids=lambda x: basename(x))
def compressible_example_file(request):
    return abspath(request.param)


def test_hybridized_compressible_examples(compressible_example_file, tmpdir, monkeypatch):
    # This ensures that the test writes output in a temporary
    # directory, rather than where pytest was run from.
    monkeypatch.chdir(tmpdir)
    subprocess.check_call([sys.executable, compressible_example_file,
                           "--running-tests", "--hybridization"])
