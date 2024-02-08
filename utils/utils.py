import io
import os
import zipfile
from typing import List, Dict
from fastapi import Response


def zip_files(filenames: List[Dict[str, str]]) -> Response:
    """
    Function that prepares an archive with all the matched images
    Args:
        filenames: response from Qdrant vector similarity search

    Returns: fastAPI response

    """
    # Define archive name
    zip_filename = "images.zip"

    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    for entry in filenames:
        fpath = entry['path']

        # Calculate path for file in zip
        fdir, fname = os.path.split(fpath)

        # Add file, at correct path
        zf.write(fpath, fname)

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

    return resp
