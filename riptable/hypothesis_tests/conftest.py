# pytest configuration for hypothesis-based riptable tests.
# Reference: https://docs.pytest.org/en/5.4.1/pythonpath.html#test-modules-conftest-py-files-inside-packages

from datetime import timedelta
import os
from hypothesis import settings, Verbosity
from hypothesis.database import ExampleDatabase

# Allow the examples directory for hypothesis to be configured via the HYPOTHESIS_EXAMPLE_DIR
# environment variable. This allows CI builds to share / re-use examples through one folder,
# which helps leverage the hypothesis support for re-running recently failing examples.
hypothesis_example_dir = os.getenv(u'HYPOTHESIS_EXAMPLE_DIR', None)
hypothesis_example_db = None if hypothesis_example_dir is None else ExampleDatabase(hypothesis_example_dir)

# Some CI build machines are slower than others, so extend the per-test-example timeout
# to avoid running into the HealthCheck for slow tests. This is preferable to disabling
# the health check altogether -- let's keep that in place to catch any grossly broken tests.
extended_deadline = timedelta(milliseconds=400)

###
# Register some non-default hypothesis settings profiles.
# Reference: https://hypothesis.readthedocs.io/en/latest/settings.html#settings-profiles
###

# ci_quick: profile used for CI builds where we want to run a few basic examples.
# This also utilizes the centralized example database (if configured); hypothesis utilizes
# failed examples saved in the database, so this means examples discovered in longer-running
# builds (with a larger search space) will be re-tested in "normal" builds and the overall
# build process will fail-fast(er) until
# examples found in long-running builds will be replayed -- this means that as long
# as those examples remain unfixed our builds will fail faster.
settings.register_profile(
    "ci_quick",
    database=hypothesis_example_db,
    print_blob=True
)

# ci: profile used for typical CI builds
settings.register_profile(
    "ci",
    database=hypothesis_example_db,
    # Run more examples compared to the default value (100), so we get better coverage.
    max_examples=200,
    deadline=extended_deadline,
    print_blob=True
)
# ci_long: profile used for special, long-running CI builds which will run less frequently
# but probe more of the search space to help find edge-case bugs.
settings.register_profile(
    "ci_long",
    database=hypothesis_example_db,
    # Run even more examples compared to normal CI builds to further explore the parameter space of the tests.
    max_examples=500,   # TODO: Ideally, we'd set this even higher if the build would still complete in a semi-reasonable amount of time.
    deadline=extended_deadline,
    print_blob=True
)

# Allow the profile to be specified via the HYPOTHESIS_PROFILE environment variable.
# If not specified, the default profile is used.
settings.load_profile(os.getenv(u'HYPOTHESIS_PROFILE', 'default'))
