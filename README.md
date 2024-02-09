# Text 2 Image Search


## :grey_question: Introduction
This is a Text-to-Image system based on Qdrant vector database, a system capable of searching for similar images based on textual queries.

The data for retrieval is represented by adds posters from Google, provided at these links: 
[first part](https://storage.googleapis.com/ads-dataset/subfolder-0.zip), [second part](https://storage.googleapis.com/ads-dataset/subfolder-0.zip).
You do not need to download the data, as everything is managed automatically in the code.

## :open_file_folder: Project structure
```
├───templates                   <- includes Jinja templates used in the project
│   ├───data_report_template.py
│   └───images_template.py
├───src                         <- FastAPI dources
│   └───schemas.py              <- schemas for FastAPI
├───utils
│   ├───data.py                 <- data related utils
│   ├───search.py               <- search related utils
│   └───utils.py                <- general utils
├───service.py                  <- endpoint for launching FastAPI app
└───prepare.py                  <- endpoint for creating data report and populating the images vector DB
```

## :gear: Installation
Run the following command:
```bash
pip install requirements.txt
```

## :hammer_and_wrench: Architecture
**Details**:
- Both image and text embeddings (dim=512) are created using a `CLIP-ViT-B-32` model
- I use Jinja templates for creating the data report that uses local plots
- I use Jinja templates to render the search form and images in the FastAPI app
- The search form is validated to contain only words (letters) and spaces
- After entering the text in a search form, I am making a POST to the API which does the following:
  - Send the form data to the search engine, which embeds the text and performs ANN
  - The search engine outputs the payload of the 5 most relevant images. The payload consists of the images paths.
  - The local images paths are sent to a Jinja template so that they are displayed in the same UI as the search form

<object data="docs/architecture.pdf" type="application/pdf" width="700px" height="700px">
</object>

[//]: # (![]&#40;docs/architecture.pdf&#41;)

## :computer: Usage
Use the following command to:
- Download, organise and extract information about the data under  `resources/data_info.csv`
- Create data report under `resources/data_report.html`
- Build the image embeddings and store them under `resources/image_embeddings.parquet`
- Create and populate a Qdrant collection with the image embeddings

```bash
python prepare.py
```

To launch the FastAPI app, execute the following command:
```bash
python service.py
```

## :fire: Results

#### :smiley: Good examples:
- Search term: **beer**
![](docs/beer.jpg)
- Search term: **astronaut**
![](docs/astronaut.jpg)
- Search term: **fries with ketchup**
![](docs/fries_with_ketchup.jpg)
- Search term: **bad weather**
![](docs/bad_weather.jpg)

#### :disappointed: Bad examples...hmmmmm...or not really.....
- Search term: **black cat** -> are images of black cats actually in the data?
![](docs/black_cat.jpg)
- Search term: **happiness** -> ..."less harmful than alcohol" they say.... But chocolate brings me happiness, indeed.
![](docs/happiness.jpg)

## Conclusions and suggestions

- It seems that the text-2-img system does a pretty good job and that the CLIP embeddings are good enough

#### Suggestion for a quantitative evaluation of retrieval accuracy
TODO


## :chart_with_downwards_trend: :chart_with_upwards_trend: Challenges and Improvements

#### Challenges
- Connection between Jinja templates and FastAPI backend (especially the form data in the template)

#### Improvements
- Use `poetry` for better dependencies solving
- Use better models for embedding the images and texts (e.g. maybe use a service like AWS, Eden AI, or models from MTEB leaderboard)

## :man: Contributors
Mihai David - [davidmihai9805@gmail.com](mailto:davidmihai9805@gmail.com)