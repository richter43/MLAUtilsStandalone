import os
import subprocess
import json
import logging
from pathlib import Path

from typing import List, Dict, Any, Optional

from ..log_utils import LOGGER_NAME
from ..utils import InitializationException, TarException

module_dir = os.path.dirname(__file__)
path = Path(module_dir)

def setup_aws(json_setting: Dict[str, Any]):
    
    subprocess.run(['aws', '--profile', 'wasabi', 'configure', 'set', 'aws_access_key_id', json_settings["AWS_ACCESS_KEY_ID"]])
    subprocess.run(['aws', '--profile', 'wasabi', 'configure', 'set', 'aws_secret_access_key', json_settings["AWS_SECRET_ACCESS_KEY"]])
    subprocess.run(['aws', '--profile', 'wasabi', 'configure', 'set', 'region', 'eu-central-1'])

with open(os.path.join(path.parent.absolute(), "settings.json")) as fp:
    json_settings = json.load(fp)
    if json_settings["AWS_ACCESS_KEY_ID"] == "<aws_access_key_id>" or json_settings["AWS_SECRET_ACCESS_KEY"] == "<aws_secret_access_key>":
        raise InitializationException

    setup_aws(json_settings)

logger = logging.getLogger(LOGGER_NAME)

def tar_encrypt_upload(path_s3: str, password: str, tar_location:str, file_list: List[str]):

    global logger

    logger.info(f"Uploading file {path_s3}")

    tar_command = ["tar", "cz", "-C", tar_location] + file_list

    logger.debug(f"Tar command: \"{' '.join(tar_command)}\"")

    tar_popen = subprocess.Popen(tar_command, stdout=subprocess.PIPE)
    encrypt_popen = subprocess.Popen(['openssl', 'enc', '-aes-256-cbc', '-salt', '-pbkdf2', '-e', '-k', password], stdin=tar_popen.stdout, stdout=subprocess.PIPE)
    aws_upload = subprocess.Popen(["aws", "s3", "cp", "-", path_s3, "--profile", "wasabi", "--endpoint-url=https://s3.eu-central-1.wasabisys.com"], stdin=encrypt_popen.stdout)
    aws_upload.wait()

    logger.info(f"Finished uploading file {path_s3}")

def download_decrypt_untar(path_s3: str, password: str, untar_location:str, top_level_dir: Optional[str]):

    global logger

    logger.info(f"Downloading file {path_s3}")

    tar_command = ["tar", "xz", "-C", untar_location]

    if top_level_dir:
        tar_command.append(f"--one-top-level={top_level_dir}")

    aws_download = subprocess.Popen(["aws", "s3", "cp", path_s3, "-", "--profile", "wasabi", "--endpoint-url=https://s3.eu-central-1.wasabisys.com"], stdout=subprocess.PIPE)
    decrypt = subprocess.Popen(["openssl", "enc", "-aes-256-cbc", "-salt", "-pbkdf2", "-d", "-k", password], stdin=aws_download.stdout, stdout=subprocess.PIPE)
    untar = subprocess.Popen(tar_command, stdin=decrypt.stdout)
    untar.wait()

    if untar.poll() == 2:
        raise TarException

    logger.info(f"Finished downloading file {path_s3}")