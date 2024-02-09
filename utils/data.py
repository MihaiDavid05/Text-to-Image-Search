import shutil
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from jinja2 import Environment, FileSystemLoader
from .utils import build_image_embeddings, update_db_collection, download_and_extract, specs


class Preparator:
    def __init__(self,
                 imgs_path: str = 'images',
                 docs_path: str = 'resources',
                 collection_name: str = 'images',
                 m: int = 16,
                 ef_construct: int = 100
                 ):
        """
        Args:
            imgs_path: Path to the images directory
            docs_path: Dir path to all the resources: plots, data info csv, embeddings, data report
            collection_name: name of the collection for image embeddings
            m: number of edges per node during the index building
            ef_construct: number of neighbours to consider during the index building

        """
        self.imgs_path = imgs_path
        self.docs_path = docs_path
        self.collection_name = collection_name
        self.m = m
        self.ef_construct = ef_construct
        self.im_df = None

    def run(self):
        self.get_data()
        self.im_df = self.store_image_info()
        self.create_report()
        self.build_embeddings()
        self.update_collection()

    def get_data(self):
        """
        Store images locally.
        """

        if os.path.isdir(self.imgs_path):
            images = os.listdir(self.imgs_path)
            if len(images) > 0:
                print("Data was already obtained.")
                return

        # URLs of the zip files to download
        urls = ['https://storage.googleapis.com/ads-dataset/subfolder-0.zip',
                'https://storage.googleapis.com/ads-dataset/subfolder-1.zip']

        # Create the target directory if it doesn't exist
        if not os.path.exists(self.imgs_path):
            os.makedirs(self.imgs_path)

        # Download and extract each zip file
        print("Downloading data...")
        for url in tqdm(urls):
            download_and_extract(url, extract_to=self.imgs_path)
        print("Finished")

        # Copy extracted files to the target directory
        print("Organising data...")
        for folder in ['0', '1']:
            src_path = os.path.join(self.imgs_path, folder)
            for filename in os.listdir(src_path):
                shutil.move(os.path.join(src_path, filename), self.imgs_path)

            # Remove the old directory
            shutil.rmtree(src_path)
        print("Finished")

    def store_image_info(self) -> pd.DataFrame:
        """
        Creates a dataframe with info about images
        Returns:
            DataFrame with all information
        """

        if not os.path.exists(self.docs_path):
            os.makedirs(self.docs_path)

        # Check if data already exists and return it if so
        data_info_path = os.path.join(self.docs_path, 'data_info.csv')
        if os.path.isfile(data_info_path):
            print("Image data already exists. Reading it...")
            results_df = pd.read_csv(data_info_path)
            print("Finished.")

            return results_df

        # Get images names
        all_imgs = os.listdir(self.imgs_path)

        # Initialize dataframe
        columns_df = ['path', 'width', 'height', 'area', 'aspect_ratio']
        imgs_df = pd.DataFrame(columns=columns_df)

        print("Storing image information...")
        for im_name in tqdm(all_imgs):
            im_path = os.path.join(self.imgs_path, im_name)

            # Read image
            im = Image.open(im_path)
            w, h = im.size

            # Add row to dataframe
            new_df = pd.DataFrame([[im_path, w, h, w * h, w / h]], columns=columns_df)
            imgs_df = new_df.copy() if imgs_df.empty else pd.concat([imgs_df, new_df])

        result_df = imgs_df.reset_index(drop=True)

        # Save the data to file
        result_df.to_csv(data_info_path, index=False)

        print("Finished.")

        return imgs_df.reset_index(drop=True)

    def create_report(self):
        """
        Creates report for the dataset.
        """

        if not os.path.exists(self.docs_path):
            os.makedirs(self.docs_path)

        # Check for report existence
        data_report_path = os.path.join(self.docs_path, 'data_report.html')
        if os.path.isfile(data_report_path):
            print("Report already created.")
            return

        print("Creating data report...")

        # Compute biggest and lowest resolutions
        areas_sorted = self.im_df.sort_values(by='area', ascending=False)
        highest_resolution = areas_sorted.iloc[0][['width', 'height']].values
        lowest_resolution = areas_sorted.iloc[-1][['width', 'height']].values

        # Plot distribution of aspect ratio
        plot_path1 = os.path.join(self.docs_path, 'area_distrib.png')
        sns_plot = sns.displot(self.im_df, x="aspect_ratio", kde=True).set(title="Aspect ratio distribution")
        sns_plot.map(specs, 'aspect_ratio')
        plt.legend()
        sns_plot.savefig(plot_path1)
        plt.close(sns_plot.fig)

        # Joint plot with width and height of the images
        plot_path2 = os.path.join(self.docs_path, 'width_height_distrib.png')
        sns_plot = sns.jointplot(x='width', y='height', data=self.im_df)
        plt.suptitle("Width and Height distributions")
        sns_plot.savefig(plot_path2)
        plt.close(sns_plot.fig)

        # Select 5 random images and plot them
        random_imgs = np.random.choice(self.im_df['path'], size=5, replace=False)

        plot_path3 = os.path.join(self.docs_path, 'images.png')
        fig, axes = plt.subplots(1, 5, figsize=(12, 6))
        for i, im_path in enumerate(random_imgs):
            im = Image.open(im_path)
            axes[i].imshow(im)
            axes[i].axis('off')
        fig.savefig(plot_path3)
        plt.close()

        # Prepare data for the template
        data = {
            'nr_imgs': len(self.im_df),
            'high_res': f"{highest_resolution[0]} x {highest_resolution[1]}",
            'low_res': f"{lowest_resolution[0]} x {lowest_resolution[1]}",
            'plot_paths': [os.path.join('..', plot_path1), os.path.join('..', plot_path2)],
            'images': os.path.join('..', plot_path3)
        }

        # Load Jinja2 template
        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('data_report_template.html')

        # Render the template with data
        html_output = template.render(**data)

        # Write the HTML content to a file
        with open(data_report_path, 'w') as file:
            file.write(html_output)

        print("Finished.")

    def build_embeddings(self):
        """
        Builds image embeddings
        """
        build_image_embeddings(self.im_df, self.docs_path)

    def update_collection(self):
        """
        Updates Qdrant collection with the embeddings
        """
        update_db_collection(self.collection_name, self.docs_path, self.m, self.ef_construct)
