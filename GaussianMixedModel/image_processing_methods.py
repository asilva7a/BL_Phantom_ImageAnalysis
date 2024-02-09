import os
import numpy as np
import tifffile
from PIL import Image
from sklearn.mixture import GaussianMixture
import pandas as pd
import cv2


class DataWrangling:
    """
    Data Wrangling Class to manage the directories and files in the project folder.

    Loops through the directories in the project folder and lists the files in each directory.

    Parameters:
    project_folder (str): Path to the project folder.

    returns:
    list of directories in the project folder
    """
    # project_folder is the folder where the data is stored
    def __init__(self, project_folder): # project_folder is the folder where the data is stored
        self.project_folder = project_folder # project_folder is the folder where the data is stored
        self.directory_df = self.initialize_directory_df() # directory_df is a dataframe that contains the names and paths of the directories in the project folder
    
    # initialize a dataframe that contains the names and paths of the directories in the project folder
    def initialize_directory_df(self): # initialize a dataframe that contains the names and paths of the directories in the project folder
        directories = [d for d in os.listdir(self.project_folder) if os.path.isdir(os.path.join(self.project_folder, d))] # list of directories in the project folder
        directory_data = [{'directory_name': d, 'directory_path': os.path.join(self.project_folder, d)} for d in directories] # list of dictionaries containing the names and paths of the directories
        return pd.DataFrame(directory_data, columns=['directory_name', 'directory_path']) # dataframe containing the names and paths of the directories
    
    # list of directories in the project folder
    def list_directories(self): # list of directories in the project folder
        return [d for d in os.listdir(self.project_folder) if os.path.isdir(os.path.join(self.project_folder, d))] # list of directories in the project folder
    
    # list of files in the directory
    def list_files(self, folder_name): # list of directories in the project folder
        folder_path = os.path.join(self.project_folder, folder_name) # path to the directory    
        all_files = [] # list of files in the directory
        for root, dirs, files in os.walk(folder_path): # walk through the directory
            for file in files: # list of files in the directory
                all_files.append(os.path.join(root, file)) # list of files in the directory
        return all_files # list of files in the directory
    
    # list of files in those files
    def list_files_in_files(self, folder_name): # list of files in those files
        folder_path = os.path.join(self.project_folder, folder_name)
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                all_files.append(os.path.join(root, file))
        return all_files
    

    def generate_dark_image(self, tiff_path, num_frames=100):
        """
        Generates a median 'dark' image from the first specified number of frames in a multi-frame TIFF file.

        This method is used for compensating the dark pixel offset in bioluminescence imaging data.

        Parameters:
        tiff_path (str): Path to the multi-frame TIFF file.
        num_frames (int, optional): Number of frames to consider for generating the dark image. Defaults to 200.

        Returns:
        numpy.ndarray: A median image representing the 'dark' image.
        """
        with Image.open(tiff_path) as img:
            frames = [np.array(img.getdata(), dtype=np.float32).reshape(img.size[::-1]) for i in range(num_frames)]
            median_frame = np.median(frames, axis=0)
            return median_frame

    def subtract_dark_image(self, raw_tiff_path, dark_image):
        """
        Subtracts a 'dark' image from each frame of a multi-frame TIFF file.

        This method is used to compensate for the average dark pixel offset in bioluminescence imaging data.

        Parameters:
        raw_tiff_path (str): Path to the raw multi-frame TIFF file.
        dark_image (numpy.ndarray): The 'dark' image to be subtracted from each frame of the raw image.

        Returns:
        list of numpy.ndarray: A list of images, each representing a frame from the raw image with the dark image subtracted.
        """
        with Image.open(raw_tiff_path) as img:
            compensated_images = []
            for i in range(img.n_frames):
                img.seek(i)
                frame = np.array(img.getdata(), dtype=np.float32).reshape(img.size[::-1])
                compensated_image = cv2.subtract(frame, dark_image)
                compensated_images.append(compensated_image)
            return compensated_images
        