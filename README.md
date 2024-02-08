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
TODO

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

Good examples:
- Search term: **beer**
[](docs/beer.jpg)
- Search term: **astronaut**
[](docs/astronaut.jpg)
- Search term: **fries with ketchup**
[](docs/fries_with_ketchup.jpg)

- Bad examples

## :man: Contributors
Mihai David - [davidmihai9805@gmail.com](mailto:davidmihai9805@gmail.com)