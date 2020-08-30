How can I contribute?
---------------------

New developers should set up a dedicated environment for working with
`riptable`. To do so, first
[create an environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
from the environment file:
```bash
conda env create -f environment.yml
```

**Note**: some users have had trouble completing the installation of the
`riptide_cpp` requirement on Mac machines due to a dependence on
[`zstd`](https://github.com/facebook/zstd). A temporary work around is to use
Homebrew to install `zstd` ahead of time
```bash
brew install zstd
```
See the [open issue](https://github.com/rtosholdings/riptable/issues/6) on the
topic for more information.

One the environment is created, activate it
```bash
conda activate rtdev
```
and finally install `riptable`
```bash
python setup.py install
```