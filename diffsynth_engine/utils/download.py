import os
import json
import requests
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm

from diffsynth_engine.utils import logging, hash
from diffsynth_engine.utils.lock import HeartbeatFileLock
from diffsynth_engine.utils.modelscope import ModelscopeClient, ModelscopeUrlInfo, parse_model_scope_url
from diffsynth_engine.utils.env import DIFFSYNTH_CACHE, DIFFSYNTH_FILELOCK_DIR
from diffsynth_engine.utils.constants import LOCAL_SCHEME, HTTP_SCHEME, HTTPS_SCHEME, MODELSCOPE_SCHEME, MB

logger = logging.get_logger(__name__)


def download_model(model_url: str, local_dir: str | ModelscopeUrlInfo = None) -> str:
    """
    Downloads a model from an HTTP URL or Modelhub URL
    """
    download_path = ""
    lock_file_path = get_lock_path(model_url)
    ensure_directory_exists(lock_file_path)
    with HeartbeatFileLock(lock_file_path):
        if model_url.startswith(LOCAL_SCHEME) or os.path.exists(model_url):
            download_path = strip_local_prefix(model_url)
        elif model_url.startswith((HTTP_SCHEME, HTTPS_SCHEME)):
            download_path = download_http_model(model_url)
        elif model_url.startswith(MODELSCOPE_SCHEME):
            download_path = ModelscopeClient.download_model_url(model_url, local_dir)
        else:
            raise ValueError(f"not supported model url scheme: {model_url}")

    download_path = find_model_file_path(download_path, model_url)
    logger.info(f"{model_url} downloaded to {download_path}")
    return download_path


def download_http_model(model_url: str, model_name: str | None = None) -> str:
    """
    Downloads a model from an HTTP URL.
    """
    if model_name is None:
        model_name = model_url.split("/")[-1]
    return _download_http_url(model_url, model_name)


def _download_http_url(http_url: str, model_name: str, headers: dict | None = None) -> str:
    snapshot_name = hash.string_sha256(http_url[:http_url.rfind('/')])
    snapshot_dir = os.path.join(DIFFSYNTH_CACHE, snapshot_name)

    download_path = os.path.join(snapshot_dir, model_name)
    if os.path.exists(download_path):
        return download_path
    response = requests.get(http_url, stream=True, timeout=30 * 60, headers=headers)  # 30min
    if response.status_code >= 400:
        raise RuntimeError(f"Download {model_name} failed, please check the model url")
    total_bytes = int(response.headers.get("content-length", 0))
    bar = tqdm(desc=f"Download {model_name}", total=total_bytes, unit="B", unit_divisor=1024, unit_scale=True)

    with tempfile.NamedTemporaryFile() as f:
        for chunk in response.iter_content(chunk_size=16 * MB):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        shutil.copy(f.name, download_path)
    return download_path


def ensure_directory_exists(filename: str):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)


def get_lock_path(model_url: str) -> str:
    if model_url.startswith(MODELSCOPE_SCHEME):
        modelscope_url_info = parse_model_scope_url(model_url)
        model_id = modelscope_url_info.model_id.replace('/', '--')
        revision = "__version" if not modelscope_url_info.revision else modelscope_url_info.revision
        lock_path = f"modelscope.{model_id}.{revision}"
    else:
        lock_path = hash.string_sha256(model_url)
    return os.path.join(DIFFSYNTH_FILELOCK_DIR, lock_path + ".lock")


def strip_local_prefix(path: str) -> str:
    return _strip_prefix(path=path, prefix=LOCAL_SCHEME)


def _strip_prefix(path: str, prefix: str) -> str:
    prefix = f"{prefix}://"
    if path.startswith(prefix):
        path = path[len(prefix):]
    return path


def find_model_file_path(download_path: str, model_url: str) -> str:
    if model_url.startswith(MODELSCOPE_SCHEME):
        return _find_modelscope_model_file_path(download_path, model_url)
    if not os.path.exists(download_path):
        raise RuntimeError(f"Download model failed, download path {download_path} not exists")
    if os.path.isfile(download_path):
        return download_path

    entries = os.listdir(download_path)
    # with single model file in the download_path
    if len(entries) == 1:
        model_file_path = os.path.join(download_path, entries[0])
        if os.path.isfile(model_file_path):
            return model_file_path
    return download_path


def _find_modelscope_model_file_path(download_path: str, model_url: str) -> str:
    if not os.path.exists(download_path) or not os.path.isdir(download_path):
        raise RuntimeError(
            f"Download model failed, {download_path} not exists or not directory, model_url: {model_url}")

    config_json_path = os.path.join(download_path, "configuration.json")
    if not os.path.exists(config_json_path) or not os.path.isfile(config_json_path):
        raise ValueError(
            f"Invalid model format, {config_json_path} not exists or not a file, model_url: {model_url}")

    with open(config_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        if "model_file_location" not in config:
            raise ValueError(
                f"Invalid configuration, cannot find `model_file_location` in {config_json_path}, model_url: {model_url}")
        model_file_path = os.path.join(download_path, config["model_file_location"])
        if not os.path.exists(model_file_path):
            raise ValueError(f"Model file not exists, model_url: {model_url}, configuration: {config}")
        return model_file_path
