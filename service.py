from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from utils.text2img import Text2Img
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# Instantiate app and template
app = FastAPI()

# Mount static directory
app.mount("/images", StaticFiles(directory="images"), name="images")

# Define Jinja template
templates = Jinja2Templates(directory='templates')

# Instantiate text 2 image
text2img = Text2Img()


@app.get("/api/search", response_class=HTMLResponse)
async def search(request: Request, text: str):

    # Grab response from Qdrant
    results = text2img.search(text=text)

    # Form the data with images names, needed for the Jinja template and return template response
    data = {'names': ['/' + res['path'].split('/')[-1] for res in results]}
    return templates.TemplateResponse(request=request, name='images_template.html', context=data)

    # OR, you can directly download the images
    # return zip_files(results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
