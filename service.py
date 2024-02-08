from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from utils.search import Text2Img
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.schemas import SearchText
# from pydantic import ValidationError


# Instantiate app and template
app = FastAPI()

# Mount static directory
app.mount("/images", StaticFiles(directory="images"), name="images")

# Define Jinja template
templates = Jinja2Templates(directory='templates')

# Instantiate text 2 image
text2img = Text2Img()


# Define the root path to show the search form
@app.get("/", response_class=HTMLResponse)
async def show_search_form(request: Request):
    return templates.TemplateResponse(name="form_template.html", context={"request": request})


@app.post("/api/search", response_class=HTMLResponse)
async def create_item(request: Request):
    form_data = await request.form()
    try:
        # Transform data to Pydantic model
        form_data_dict = dict(form_data)
        search_text = SearchText(**form_data_dict)

        # Grab response from Qdrant
        results = text2img.search(text=search_text.text)

        # Form the data with images names, needed for the Jinja template and return template response
        context = {'names': ['/' + res['path'].split('/')[-1] for res in results],
                   'request': request
                   }

        # return zip_files(results)
        return templates.TemplateResponse(request=request, name='images_template.html', context=context)

    except ValueError as e:
        context = {
            'request': request,
            'error': str(e)
        }
        return templates.TemplateResponse("form_template.html", context)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
