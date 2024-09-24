import os
import requests
import shutil
import zipfile
import xmltodict
import json
import logging

# Get the logger for this module
logger = logging.getLogger(__name__)


class DataDownloader:
    """
    Class for downloading, unzipping, and processing the XML data files into a JSON file.

    Attributes:
    -----------
    data_url : str
        URL to download the zip file containing the data.
    zip_file_name : str
        Name of the zip file to be downloaded.
    data_folder : str
        Directory to store the downloaded and processed files.
    download_path : str
        Local directory where the file will be initially downloaded.
    unzip_folder : str
        Directory where the unzipped files will be stored.
    json_file_name : str
        Name of the JSON file to store the unified XML data.
    Methods:
    --------
    download_file():
        Downloads a zip file from a specified URL.
    move_file():
        Moves the downloaded file to a data folder.
    unzip_and_delete_downloaded_file():
        Unzips the downloaded file and deletes the zip file afterward.
    create_json_file():
        Combines all unzipped XML files into a single JSON file.
    remove_unzipped_xml_files():
        Deletes the unzipped XML files.
    automate_download_and_unzip():
        Automates the download, move, unzip, conversion to JSON, and cleanup processes.
    Raises:
    -------
    RequestException:
        If downloading the file fails.
    OSError:
        If file or directory operations fail.

    """

    def __init__(self):
        self.data_url = 'https://www.nsf.gov/awardsearch/download?DownloadFileName=2020&All=true'
        self.zip_file_name = '2020.zip'
        self.data_folder = 'data'
        self.download_path = os.path.expanduser('~/Downloads')
        self.unzip_folder = 'data/unzipped_files'
        self.json_file_name = 'all_data.json'

    def download_file(self) -> None:
        """
        Downloads the zip file from the specified URL and saves it to the download directory.
        """
        logging.info('Download started.')
        try:
            response = requests.get(self.data_url, stream=True)
            response.raise_for_status()
            raw_data_file = os.path.join(self.download_path, self.zip_file_name)
            with open(raw_data_file, mode='wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            logging.info(f'File downloaded.')
        except requests.RequestException as e:
            logging.error(f'Error downloading the file: {e}')
            raise

    def move_file(self) -> None:
        """
        Moves the downloaded file to the data folder.
        """
        os.makedirs(self.data_folder, exist_ok=True)
        src_path = os.path.join(self.download_path, self.zip_file_name)
        new_path = os.path.join(self.data_folder, self.zip_file_name)

        try:
            shutil.move(src_path, new_path)
            logging.info(f'File moved to {new_path}')
        except shutil.Error as e:
            logging.error(f'Error moving the file: {e}')
            raise

    def unzip_and_delete_downloaded_file(self) -> None:
        """
        Unzips the downloaded file and deletes the zip file.
        """
        zip_path = os.path.join(self.data_folder, self.zip_file_name)
        os.makedirs(self.unzip_folder, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, mode='r') as zip_ref:
                zip_ref.extractall(self.unzip_folder)
            logging.info(f'Unzipping file {self.zip_file_name}')

            os.remove(zip_path)
            logging.info(f'Deleted file {self.zip_file_name}')
        except (zipfile.BadZipFile, OSError) as e:
            logging.error(f'Error unzipping or deleting file: {e}')
            raise

    def create_json_file(self) -> None:
        """
        Combines all XML files in the unzipped folder into a single JSON file.
        """
        logging.info('Creating JSON file.')
        all_data = []

        try:
            for xml_file_name in os.listdir(self.unzip_folder):
                if xml_file_name.endswith('.xml'):
                    xml_file_path = os.path.join(self.unzip_folder, xml_file_name)
                    with open(xml_file_path, mode='r', encoding='utf-8') as xml_file:
                        xml_content = xml_file.read()
                        xml_dict = xmltodict.parse(xml_content)
                        all_data.append(xml_dict)

            json_path = os.path.join(self.data_folder, self.json_file_name)
            with open(json_path, mode='w', encoding='utf-8') as json_file:
                json.dump(all_data, json_file, indent=4)

            logging.info(f'JSON file created: {self.json_file_name}')
        except Exception as e:
            logging.error(f'Error creating JSON file: {e}')
            raise

    def remove_unzipped_xml_files(self) -> None:
        """
        Deletes the unzipped XML files from the directory.
        """
        try:
            shutil.rmtree(self.unzip_folder)
            logging.info(f'Removed unzipped XML files from {self.unzip_folder}')
        except OSError as e:
            logging.error(f'Error removing unzipped files: {e}')
            raise

    def automate_download_and_unzip(self) -> None:
        """
        Automates the entire process: download, move, unzip, create JSON, and clean up files.
        """
        try:
            self.download_file()
            self.move_file()
            self.unzip_and_delete_downloaded_file()
            self.create_json_file()
            self.remove_unzipped_xml_files()
            logging.info(f'Data file {self.json_file_name} ready to be used.')
        except Exception as e:
            logging.error(f'Automation failed: {e}')
            raise
