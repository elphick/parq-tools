"""
Archive Extraction
===================

This example demonstrates how to extract archives using the `parq_tools` library. The utility supports
both standard extraction using Python's `zipfile` module and a fallback to `7-Zip` for cases where
unsupported compression methods, such as Deflate64, are used.

Use Case
--------
Large files zipped in Windows can result in compression with Deflate64. In such cases,
`parq_tools` provides a fallback mechanism to use `7-Zip` for extraction.
"""

from pathlib import Path
import tempfile
from parq_tools.utils.archive_utils import extract_archive

# %%
# Create an Example Archive
# --------------------------
#
# For demonstration purposes, we create a simple ZIP archive. Note that this example assumes the archive
# does not use unsupported compression methods like Deflate64. If such a method is used, the fallback
# to `7-Zip` will be triggered during extraction.

import zipfile

def create_example_archive(archive_path: Path):
    with zipfile.ZipFile(archive_path, 'w') as zipf:
        zipf.writestr("example.txt", "This is an example file.")

# Create a temporary directory and archive
temp_dir = Path(tempfile.gettempdir()) / "archive_example"
temp_dir.mkdir(parents=True, exist_ok=True)
example_archive = temp_dir / "example.zip"
create_example_archive(example_archive)

# %%
# Extract the Archive
# --------------------
#
# Use the `extract_archive` function to extract the contents of the archive. If the archive uses an
# unsupported compression method, the function will automatically fall back to `7-Zip` (if installed).

output_dir = temp_dir / "extracted"
extract_archive(example_archive, output_dir, show_progress=True)

# %%
# Verify the Extracted Contents
# ------------------------------
#
# Check that the extracted file exists and contains the expected content.

extracted_file = output_dir / "example.txt"
assert extracted_file.exists()
print(f"Extracted file content: {extracted_file.read_text()}")