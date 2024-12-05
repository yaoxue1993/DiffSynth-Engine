import os
import enum
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs
from modelscope import HubApi, snapshot_download

from diffsynth_engine.utils import logging
from diffsynth_engine.utils.env import DIFFSYNTH_CACHE, ENV_MODELSCOPE_DOMAIN
from diffsynth_engine.utils.constants import MODELSCOPE_SCHEME

logger = logging.get_logger(__name__)


class ModelscopeUrlInfo(BaseModel):
    model_id: str
    revision: str = ""
    access_token: str = ""
    endpoint: str | None = None


def parse_model_scope_url(url: str) -> ModelscopeUrlInfo:
    # modelscope://muse/flux_vae?revision=20241015120836
    parsed = urlparse(url)
    repo_name = parsed.netloc
    model_name = parsed.path
    model_id = repo_name + model_name
    params = parse_qs(parsed.query)
    revision = params.get("revision", [""])[0]
    access_token = params.get("access_token", [""])[0]
    endpoint = params.get("endpoint", [None])[0]
    return ModelscopeUrlInfo(model_id=model_id, revision=revision, access_token=access_token, endpoint=endpoint)


class ModelscopeClient:
    MODEL_FILE_SUFFIX = [".safetensors"]

    class ModelType(enum.Enum):
        lora = 1
        ckpt = 2
        VAE = 3

    class AlgoVersion(enum.Enum):
        SD_1_5 = 1
        SD_3 = 2
        SD_XL = 3
        FLUX_1 = 4

    @classmethod
    def _get_download_directory(
            cls,
            path: str | ModelscopeUrlInfo,
    ) -> str:
        if path is None or isinstance(path, str):
            return path
        if isinstance(path, ModelscopeUrlInfo):
            directory = os.path.join(
                DIFFSYNTH_CACHE, "modelscope", path.model_id, path.revision if path.revision else "__version")
        else:
            raise ValueError(f"Unknown path type: {type(path)}")
        return directory

    @classmethod
    def download_model(
            cls,
            model_id: str,
            revision: str,
            access_token: str = "",
            endpoint: str = "",
            local_dir: str | ModelscopeUrlInfo = None,
    ) -> str:
        local_dir = cls._get_download_directory(local_dir)
        old_endpoint = os.environ.get(ENV_MODELSCOPE_DOMAIN, None)
        if access_token:
            api = HubApi()
            api.login(access_token)
        try:
            if endpoint:
                os.environ[ENV_MODELSCOPE_DOMAIN] = endpoint
            download_path = snapshot_download(
                model_id, revision=revision, local_dir=local_dir)
        except Exception as ex:
            raise ex
        finally:
            if old_endpoint is not None:
                os.environ[ENV_MODELSCOPE_DOMAIN] = old_endpoint
            else:
                if ENV_MODELSCOPE_DOMAIN in os.environ:
                    del os.environ[ENV_MODELSCOPE_DOMAIN]
        return download_path

    @classmethod
    def download_model_url(
            cls,
            model_url: str,
            local_dir: str | ModelscopeUrlInfo | None = None,
    ) -> str:
        if not model_url.startswith(MODELSCOPE_SCHEME):
            raise ValueError(
                f"model url must startswith {MODELSCOPE_SCHEME}://")
        modelscope_url_info = parse_model_scope_url(model_url)
        if not local_dir:
            local_dir = modelscope_url_info
        return cls.download_model(
            modelscope_url_info.model_id,
            modelscope_url_info.revision,
            access_token=modelscope_url_info.access_token,
            endpoint=modelscope_url_info.endpoint,
            local_dir=local_dir,
        )
