import pandas as pd
import os
import cv2

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


