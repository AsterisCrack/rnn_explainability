import gdown
import zipfile


def download_extract_data() -> None:
    """Function to download and extract the data used in the project."""
    # Google Drive direct download link
    url = 'https://drive.google.com/uc?id=1zQRH1zYBHJ_vU_uMkKvvvwQiZwP5N7wW'

    # Destination file name
    output = 'NLP_DATA.zip'

    # Download the file
    gdown.download(url, output, quiet=False)

    # Unzip the downloaded file
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('.')

    return None


if __name__ == '__main__':
    download_extract_data()
