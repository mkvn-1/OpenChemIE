"""Poppler-backed compatibility module for the ``pdftotext`` package.

OpenChemIE only needs ``pdftotext.PDF(file_obj)`` as a list-like collection of
page strings. This module delegates extraction to Poppler's ``pdftotext`` CLI,
which is easier to ship in Docker than the compiled Python extension.
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path


__version__ = "poppler-cli-compat"


class PDF:
    def __init__(self, file, *, physical: bool = False, raw: bool = False, password: str | None = None):
        self._pages = self._extract_pages(file, physical=physical, raw=raw, password=password)

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def __iter__(self):
        return iter(self._pages)

    @staticmethod
    def _extract_pages(file, *, physical: bool, raw: bool, password: str | None):
        executable_names = ["pdftotext.exe", "pdftotext"] if sys.platform == "win32" else ["pdftotext"]
        pdftotext = next((path for name in executable_names if (path := shutil.which(name))), None)
        if pdftotext is None:
            raise RuntimeError("Poppler pdftotext executable was not found on PATH")

        temp_path = None
        pdf_path = getattr(file, "name", None)

        if not pdf_path or not os.path.exists(pdf_path):
            data = file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp.write(data)
                temp_path = temp.name
                pdf_path = temp_path

        command = [pdftotext, "-enc", "UTF-8", "-eol", "unix"]
        if physical:
            command.append("-layout")
        if raw:
            command.append("-raw")
        if password:
            command.extend(["-upw", password])
        command.extend([str(Path(pdf_path)), "-"])

        try:
            proc = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        text = proc.stdout.decode("utf-8", errors="replace")
        if text.endswith("\f"):
            text = text[:-1]
        return text.split("\f") if text else []
