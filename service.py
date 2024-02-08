import io
import os
import zipfile
from typing import List, Dict
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse
from utils.text2img import Text2Img
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# Instantiate app and template
app = FastAPI()

# Mount static directory
app.mount("/images", StaticFiles(directory="images"), name="images")

# Define Jinja template
templates = Jinja2Templates(directory='utils')

# Instantiate text 2 image
text2img = Text2Img()


def zip_files(filenames: List[Dict[str, str]]) -> Response:
    """
    Function that prepares an archive with all the matched images
    Args:
        filenames: response from Qdrant vector similarity searcg

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


@app.get("/api/search", response_class=HTMLResponse)
async def search(request: Request, text: str):
    # Grab response from Qdrant
    results = text2img.search(text=text)

    # Form the data with images names, needed for the Jinja template
    data = {'names': ['/' + res['path'].split('/')[-1] for res in results]}

    return templates.TemplateResponse(request=request, name='images_template.html', context=data)

    # This is another option where you directly download the images
    # return zip_files(results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
