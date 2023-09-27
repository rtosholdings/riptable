How to contribute
-----------------

The Riptable team welcomes your help to identify problems and report unexpected
behavior. Set up a developer environment to make changes to the code and to build
Riptable from source.

Set up your development environment
-----------------------------------

Use [Conda](https://docs.conda.io/en/latest/) to create and manage your environment,
including required packages and dependencies.

Riptable relies on the `riptide_cpp` package, a C++ library that enables multithreaded
processing of arrays and the SDS file format. You can set up your development
environment for Riptable only or for both Riptable and riptide_cpp. See the riptide_cpp
contributing guide for instructions on setting up a development environment.

>**Note:** Riptable supports 64-bit Windows and Linux.

To set up a Riptable development environment:

1. Clone the [Riptable](https://github.com/rtosholdings/riptable) repository from
Github.
1. Add the `rtosholdings` and `conda-forge` channels to Conda to access the correct
package versions.

    ```bash
    conda config --add channels conda-forge rtosholdings
    ```

1. Create a new Conda environment, and install the `riptide_cpp` package and all
required packages returned by the provided `gen_requirements.py` script:

    ```bash
    conda create --name riptable_dev riptide_cpp $(python riptable/dev_tools/gen_requirements.py -q developer)
    ```

1. Activate the new Conda environment:

    ```bash
    conda activate riptable_dev
    ```

1. Add your local Riptable repository to `PYTHONPATH`.
1. Check that Python is using your developer Riptable repository:

    ```bash
    python -c 'import riptable; print(riptable)'
    ```

    This command returns the path to your local repository if everything is correctly
    configured.

Run tests for your developer environment
----------------------------------------

After making changes to the code, run the tests in the `riptable/tests` directory to
confirm that Riptable works as expected. Use the `runtest.py` script to run all tests
with `pytest`:

```bash
python riptable/runtest.py
```

Report issues in GitHub
-----------------------

We encourage you to submit GitHub issues and pull requests to report and fix problems
with Riptable. After you submit a pull request, we perform our own testing and
validation of all changes before releasing them.

>**Note:** Before you submit an issue, search to see if the same bug has already exists.

