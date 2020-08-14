import os
import subprocess
from typing import List, Optional


# The HTML rendering is expected to be a snippet of an HTML document as opposed to a standalone webpage.
# The following will ignore validation errors that are meant for a standalone HTML document.
_WHITELIST_ERRORS: List[str] = [
    # DOCTYPE tag is expected at the top of the HTML document; ignore in snippet.
    r'Expected "<!DOCTYPE html>"',
    # Title element defines the documents title shown in the browser; ignore in snippet.
    r'Element "head" is missing a required instance of child element "title"',
    r"Non-space character in page trailer",
    # Some CSS background-color hex values are reported as invalid.
    r'not a "background-color" value',
    # The following are due to multiple HTML <html> elements in a single snippet.
    # Valid HTML has one per document, but we render two:
    # 1) Riptable object and styles
    # 2) Metadata about the Riptable object that was rendered such as shape and bytes
    r'Stray start tag "html"',
    r'Stray start tag "p"',
    r"fatal: Cannot recover after last error",
]


class VNUChecker:
    _CN = "VNUChecker"
    _JAVA = r"java"
    _test_dispaly_path = os.path.join(
        os.getcwd(),
        r"Python",
        r"core",
        r"riptable",
        r"test_tooling_integration",
        r"test_display",
    )
    _JAR_PATH = os.path.join(_test_dispaly_path, r"vnu_jar", r"vnu.jar")
    _BASE_PATH = os.path.join(_test_dispaly_path, r"test_display", r"html_output")

    def __init__(
        self,
        java: Optional[str] = None,
        jar_path: Optional[str] = None,
        base_path: Optional[str] = None,
        errors_only: bool = False,
        ascii_quotes: bool = False,
    ):
        self._java = java
        self._jar_path = jar_path
        self.base_path = base_path
        self._errors_only = errors_only
        self._ascii_quotes = ascii_quotes

        if self._java is None:
            self._java = VNUChecker._JAVA
        if self._jar_path is None:
            self._jar_path = VNUChecker._JAR_PATH
        if self.base_path is None:
            self.base_path = VNUChecker._BASE_PATH

        self._args = self._build_args()

    def __str__(self):
        return " ".join(self._args)  # The command line representation.

    def __repr_(self):
        return f"{VNUChecker._CN}(java={self._java}, jar_path={self._jar_path}, dir_path={self.base_path}, errors_only={self._errors_only}, ascii_quotes={self._ascii_quotes})"

    def _is_whitelist_error(
        self, error_text: str, extra_whitelists: Optional[List[str]] = None
    ) -> bool:
        """Returns ``False`` if ``error_text`` is a whitelisted error, otherwise ``True``."""
        if extra_whitelists is None:
            extra_whitelists = []
        whitelist_errors = _WHITELIST_ERRORS.copy() + extra_whitelists
        for we in whitelist_errors:
            if error_text.find(we) != -1:  # found whitelisted error
                return False
        return True

    def _build_args(self) -> List[str]:
        """Returns a list of program arguments that are used to kick of the VNU Checker."""
        cmd: List[str] = [self._java, "-jar", self._jar_path]
        if self._ascii_quotes:
            cmd.append("--asciiquotes")
        if self._errors_only:
            cmd.append("--errors-only")
        cmd.append(self.base_path)
        return cmd

    def _run(self, args: Optional[List[str]] = None) -> List[str]:
        """Runs the VNU Checker and returns a list of errors that are not whitelisted.
        Uses the default program arguments if none are specified.
        """
        if args is None:
            args = self._args
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = p.communicate()
        errors = stderr.decode("utf-8").splitlines()
        return list(filter(self._is_whitelist_error, errors))

    def validate(self, dir_path: Optional[str] = None) -> List[str]:
        """``validate`` will run the VNU Checker and return a list of errors. An empty list signals no errors.
        ``validate`` takes an optional directory path to run the VNU Checker, otherwise the default one is used.
        """
        if dir_path is None:
            if self.base_path is None:
                raise ValueError(f"{self._CN}.validate: need a directory to validate")
            dir_path = self.base_path
        if not os.listdir(dir_path):
            raise ValueError(
                f"{self._CN}.validate: {dir_path} is empty; no files to validate"
            )
        return self._run()
