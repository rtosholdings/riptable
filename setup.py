from setuptools import setup

package_name = "riptable"

install_requires = [
    "ansi2html >=1.5.2",
    "numba >=0.56.2",
    "numpy >=1.23",
    "pandas >=1.0,<3.0",
    "python-dateutil",
    "riptide_cpp >=1.14.0,<2",
]

setup(
    name=package_name,
    packages=[package_name],
    use_scm_version={
        "version_scheme": "post-release",
        "write_to": "riptable/_version.py",
        "write_to_template": '__version__ = "{version}"',
    },
    setup_requires=["setuptools_scm"],
    package_dir={package_name: "riptable"},
    description="Python Package for riptable studies framework",
    author="RTOS Holdings",
    author_email="rtosholdings-bot@sig.com",
    url="https://github.com/rtosholdings/riptable",
    python_requires=">=3.8",
    install_requires=install_requires,
    zip_safe=False,
    include_package_data=True,
    package_data={
        package_name: [
            "tests/*",
            "tests/test_files/*",
            "benchmarks/*",
            "hypothesis_tests/*",
            "Utils/*",
            "test_tooling_integration/*",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
