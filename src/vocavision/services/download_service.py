"""HTTP download helpers."""

from __future__ import annotations

import os
from pathlib import Path

import requests


class DownloadService:
    def __init__(self, timeout_sec: int = 120) -> None:
        self.timeout_sec = timeout_sec

    def download_to_file(self, url: str, output_path: Path, *, overwrite: bool = False) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
            return output_path

        temp_path = output_path.with_suffix(output_path.suffix + ".part")
        with requests.get(url, timeout=self.timeout_sec, stream=True) as response:
            response.raise_for_status()
            with temp_path.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file_obj.write(chunk)
        os.replace(temp_path, output_path)
        return output_path
