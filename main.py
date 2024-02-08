from utils.data import create_report, store_image_info, get_data
from utils.utils import build_image_embeddings, update_db_collection


if __name__ == '__main__':

    # Obtain data
    get_data()

    # Store images info
    im_df = store_image_info()

    # Create data report and export to .html file
    create_report(im_df)

    # Build and save image embeddings
    build_image_embeddings(im_df)

    # Create collection
    update_db_collection()
