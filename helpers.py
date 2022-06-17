import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import cv2

import hashlib
import time
from tqdm import tqdm
import warnings


def hash_files(data_dir_path):
    """
    Renames files in data directory to its file hash.
    
    Parameters
    ----------
    data_dir_path : str
        Path to data dictionary
    """
    def sha1_file(file_path):
        """
        Create hashed name for file path
        """
        f = open(file_path, "rb")
        r = hashlib.sha1(f.read()).hexdigest()
        f.close()
        return r

    # List data directory
    data_dir = os.listdir(data_dir_path)

    # List sub directories in data directory
    for label_dir_path in data_dir:
        label_dir_path = os.path.join(data_dir_path, label_dir_path)

        # Get full relative file paths for images
        for img_name in os.listdir(label_dir_path):
            img_name_path = os.path.join(label_dir_path, img_name)
            
            # Rename image file with its hash
            if img_name_path.endswith(".jpg"):
                hexh = f"{os.path.join(label_dir_path, sha1_file(img_name_path))}.jpg"
                print(f"Renamed {img_name_path} to {hexh}")
                os.rename(img_name_path, hexh)
                
                
def create_metadata(data_dir_path: str) -> pd.DataFrame:
    """
    Parses through data directory and sub-directories to create a metadata csv,
    containing paths, names and labels.
    
    Parameters
    ----------
    data_dir_path : str
        Relative path to data directory.
        
    Returns
    -------
    pd.DataFrame
        Dataframe containing relevant metadata collected from data directory.
    """
    
    data_dir = os.listdir(data_dir_path)

    # Create empty dictionary 
    data_metadata = pd.DataFrame(columns=["img_name", "img_path", "label"])
    
    for label in data_dir:
        label_dir_path = os.path.join(data_dir_path, label)
        
        if os.path.isdir(label_dir_path):
            for img_name in os.listdir(label_dir_path):
                img_name_path = os.path.join(label_dir_path, img_name)
                
                if ".jpg" in img_name_path:
                    # Add to metadata
                    data_metadata = data_metadata.append(
                        {
                            "img_name": img_name, 
                            "img_path": img_name_path, 
                            "label": label
                        }, ignore_index=True
                    )            
    
    return data_metadata


def read_images(data_dir_path: str, 
                rows:int=512, 
                cols:int=512, 
                channels:int=3, 
                write_images:bool=False, 
                output_data_dir_path:str=None)->np.array:
    """
    Reads all images in all labels/sub-directories in data_dir_path into np.array.
    
    Parameters
    -----------
    data_dir_path : str
        Path to image directory.
    rows : int
        Row (height) dimension to resize images to.
    cols : int
        Columns (width) dimension to resize images to.
    channels : int
        Number of image channels, e.g. RGB are 3 channels, GREY is 2 channels.
    write_images : bool
        Specifies whether to write new images to specified data directory
    output_data_dir_path : str
        If write_images is True, directory path to write new images to.
    
    Returns
    -------
    np.array
        Array containing all images in directory.
    """
    
    assert len(os.listdir(data_dir_path)) > 0, "Empty parent directory."
    
    img_array = np.empty(shape=(0, 512, 512, 3), dtype=np.uint8)
    
    print(f"Reading images from: {data_dir_path}")
    print(f"Rows set to {rows}")
    print(f"Columns set to {cols}")
    print(f"Channels set to {channels}")
    if write_images:
        print("\nWriting images to disk!")
        print(f"Writing images to: {output_data_dir_path}")
        time.sleep(4)
    else:
        print(f"Writing images is set to: {write_images}")
    print("Reading images...")
    
    # Loop through all labels and images
    for label in os.listdir(data_dir_path):
        # Label/sub dir path
        label_path = os.path.join(data_dir_path, label)
        
        # Check for directory
        if os.path.isdir(label_path) & (label[0] != "."):
            for img_name in tqdm(os.listdir(label_path)):
                # Individual image path
                img_path = os.path.join(label_path, img_name)
                # Read image, resize and append to array
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (cols, rows))
                img_expand = np.expand_dims(img, axis=0)
                
                # Create directories and write images if specified
                if write_images:
                    label_path_write = os.path.join(output_data_dir_path, label)
                    img_path_write = os.path.join(label_path_write, img_name)
                    if os.path.isdir(label_path_write):
                        cv2.imwrite(img_path_write, img)
                    else:
                        os.mkdir(label_path_write)
                        cv2.imwrite(img_path_write, img)
                
                img_array = np.append(img_array, img_expand, axis=0)
            
    time.sleep(1)
    print("Image reading complete.")
    print(f"Image array shape: {img_array.shape}")
    
    return img_array


def display_img(img: np.array=None) -> None:
    """
    View multiple images stored in files, stacking vertically

    Parameters
    ----------
    img : np.array 
        Numpy array containing image to display.
            
    Returns
    -------
    plt.figure
        Plotted image.
    """
    plt.figure()
    plt.imshow(img)
    
    
