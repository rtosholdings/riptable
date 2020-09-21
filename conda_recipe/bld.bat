%PYTHON% setup.py install
:: When setuptools is a install_requires in setup.py (either implicit or explicit), "python setup.py
:: install" will create a file setuptools.pth in the site-packages directory. Currently numba
:: requires setuptools so it is an implicit install_requires. The content of setuptools.pth is the
:: absolute path to the site-package directory. On Linux, conda-build will replace the prefix of the
:: path with a placeholder and let conda to replace the placeholder with the environment's prefix
:: when installing the package. But on Windows, conda-build doesn't do that, resulting in the path
:: not pointing to the environment's site-packages directory. It doesn't impact using the
:: environment, but conda-pack will consider it as editable package (installed by "python setup.py
:: develop" or "pip install -e") and refuse to pack the environment. It could be worked around by
:: using the option --ignore-editable-packages but it is better to not having a "problematic"
:: (inconsistent) file to begin with.
del /Q /F %SP_DIR%\setuptools.pth
