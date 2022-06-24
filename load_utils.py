import os
import torch
import shutil
from os.path import join, exists

#Here we can write complex functions to load our data given our data folder structure

def get_wsi_paths(data_folder: str, annotation_folder: str, wsi_to_load: int):
    #i guess the right way to do this is prepare a text file with paths in advance, divided in test and train wsi by hand
    wsi_fnames = os.listdir(data_folder)
    wsi_paths = []
    xml_paths = []

    for i in range(0, wsi_to_load):
        if exists(join(data_folder, wsi_fnames[i])) == False:
            continue
        wsi_paths.append(join(data_folder, wsi_fnames[i]))

        wsi_fname = wsi_fnames[i].split(".")
        xml_fname = ".".join(wsi_fname[0:-1])+".xml"

        xml_paths.append(join(annotation_folder, xml_fname))
    
    return wsi_paths, xml_paths

def save_checkpoint(args, state, is_best: bool, filename: str) -> None:
    model_path = join(args.output_folder, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.output_folder, "best_model.pth"))