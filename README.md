# Text 2 Image Search


## :grey_question: Introduction
This is a Text-to-Image system based on Qdrant vector database, a system capable of searching for similar images based on textual queries.

The data for retrieval is represented by adds posters from Google, provided at these links: 
[first part]('https://storage.googleapis.com/ads-dataset/subfolder-0.zip'), [second part]('https://storage.googleapis.com/ads-dataset/subfolder-0.zip').

## :open_file_folder: Project structure
```
├───templates                   <- includes Jinja templates used in the project
│   ├───data_report_template.py
│   └───images_template.py
├───utils
│   ├───data.py                 <- data related utils
│   ├───search.py               <- search related utils
│   └───utils.py                <- general utils
├───service.py                  <- endpoint for launching FastAPI app
└───prepare.py                  <- endpoint for creating data report and populating the images vector DB
```
## :computer: Usage


## :fire: Results


## :man: Contributors
Mihai David