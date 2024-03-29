import glob
import logging
import os
import random
import shutil
import sys
import traceback
from os.path import join
from pathlib import Path
from threading import Lock

import numpy as np
import torch

"""
    This file might be moved inside utils folder.
"""

LOGGER_NAME = ''

def make_deterministic(seed=0):
    """Make results deterministic. If seed == -1, do not make deterministic.
    Running the script in a deterministic way might slow it down.
    """
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def enumerate_file(output_folder: str, filename: str) -> str:
    path_filename = Path(filename)
    glob_magic_match = os.path.join(output_folder, f"{path_filename.stem}*{path_filename.suffix}")
    num_files = len(glob.glob(glob_magic_match))
    filename = f"{path_filename.stem}_{num_files}{path_filename.suffix}"
    return filename

def setup_logging(output_folder, console="debug",
                  info_filename="info.log", debug_filename="debug.log"):
    """Set up logging files and console output.
    Creates one file for INFO logs and one for DEBUG logs.
    Args:
        output_folder (str): creates the folder where to save the files.
        debug (str):
            if == "debug" prints on console debug messages and higher
            if == "info"  prints on console info messages and higher
            if == None does not use console (useful when a logger has already been set)
        info_filename (str): the name of the info file. if None, don't create info file
        debug_filename (str): the name of the debug file. if None, don't create debug file
    """

    global LOGGER_NAME

    if os.path.exists(output_folder):
        print("log folder already exists")#raise FileExistsError(f"{output_folder} already exists!")
    else: 
        os.makedirs(output_folder, exist_ok=True)
    # logging.Logger.manager.loggerDict.keys() to check which loggers are in use
    base_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    if console != None and not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        if console == "debug": console_handler.setLevel(logging.DEBUG)
        if console == "info":  console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)

    # Useful for avoiding the addition of the same handler 
    if info_filename != None:
        #Mode set to append so that the folder 
        info_filename = enumerate_file(output_folder, info_filename)
        info_file_handler = logging.FileHandler(join(output_folder, info_filename))
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)
    
    if debug_filename != None:
        debug_filename = enumerate_file(output_folder, debug_filename)
        debug_file_handler = logging.FileHandler(join(output_folder, debug_filename))
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)
    
    def exception_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = exception_handler
