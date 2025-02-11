import os
from typing import Optional
from pathlib import Path

from modelscope import snapshot_download
from modelscope.hub.api import HubApi

from diffsynth_engine.utils import logging
from diffsynth_engine.utils.lock import HeartbeatFileLock
from diffsynth_engine.utils.env import DIFFSYNTH_FILELOCK_DIR, DIFFSYNTH_CACHE


logger = logging.get_logger(__name__)

def fetch_modelscope_model(
        model_id:str, 
        revision:Optional[str]=None, 
        subpath:Optional[str]=None, 
        access_token:Optional[str]=None
    ) -> str:

    lock_file_name = f"modelscope.{model_id.replace('/', '--')}.{revision if revision else '__version'}.lock"
    lock_file_path = os.path.join(DIFFSYNTH_FILELOCK_DIR, lock_file_name)
    ensure_directory_exists(lock_file_path)
    if access_token is not None:
        api = HubApi()
        api.login(access_token)        
    with HeartbeatFileLock(lock_file_path):
        directory = os.path.join(DIFFSYNTH_CACHE, "modelscope", model_id, revision if revision else "__version")    
        path = snapshot_download(
            model_id, 
            revision=revision, 
            local_dir=directory, 
        )
    if subpath is not None:
        path = os.path.join(path, subpath)
    return path

def fetch_civitai_model(model_url:str) -> str:
    pass

def ensure_directory_exists(filename: str):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)