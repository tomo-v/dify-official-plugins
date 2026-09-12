import base64
import binascii
import logging
import re
import time
from collections.abc import Generator
from typing import Any

import httpx
import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.errors.model import InvokeError
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

OMNI_MODELS = {"gemini-omni-1.1-flash"}
VEO_MODELS = {
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-lite-generate-preview",
}
SUPPORTED_MODELS = OMNI_MODELS | VEO_MODELS

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
FILES_URL = "https://generativelanguage.googleapis.com/v1beta/files"
REQUEST_TIMEOUT = (30, 600)
MAX_POLL_ATTEMPTS = 120


class GeminiVideoTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        model = tool_parameters.get("model", "gemini-omni-1.1-flash")
        if model not in SUPPORTED_MODELS:
            raise InvokeError(f"model:{model} is not supported")

        self._validate_parameters(tool_parameters, model)
        api_key = self.runtime.credentials["gemini_api_key"]
        proxy_url = tool_parameters.get("proxy_url")

        if model in OMNI_MODELS:
            video_bytes = self._generate_with_omni(tool_parameters, model, api_key, proxy_url)
        else:
            video_bytes = self._generate_with_veo(tool_parameters, model, api_key, proxy_url)

        yield self.create_blob_message(
            blob=video_bytes,
            meta={"mime_type": "video/mp4", "filename": "output.mp4"},
        )

    def _generate_with_omni(
        self,
        tool_parameters: dict[str, Any],
        model: str,
        api_key: str,
        proxy_url: str | None,
    ) -> bytes:
        payload = self._build_omni_payload(tool_parameters, model)
        response = self._request(
            "post", INTERACTIONS_URL, api_key, proxy_url, json=payload
        )
        interaction = response.json()

        if interaction.get("status") == "failed":
            raise InvokeError(self._interaction_error(interaction))

        video = self._find_output_video(interaction)
        if video is None:
            raise InvokeError(self._interaction_error(interaction, "no video data returned"))

        if video.get("data"):
            try:
                return base64.b64decode(video["data"], validate=True)
            except (binascii.Error, TypeError, ValueError) as error:
                raise InvokeError("video generation failed: invalid base64 video data returned") from error

        video_uri = video.get("uri")
        if not video_uri:
            raise InvokeError("video generation failed: video response contains neither data nor URI")

        self._wait_for_file(video_uri, api_key, proxy_url)
        return self._request("get", video_uri, api_key, proxy_url).content

    def _build_omni_payload(self, tool_parameters: dict[str, Any], model: str) -> dict[str, Any]:
        prompt = tool_parameters["prompt"].strip()
        negative_prompt = tool_parameters.get("negative_prompt")
        duration_seconds = GeminiVideoTool._duration_seconds(tool_parameters)

        # Omni does not expose negative_prompt or duration parameters. Google
        # recommends expressing both constraints in the regular prompt.
        prompt = f"{prompt}\n\nTarget duration: approximately {duration_seconds} seconds."
        if negative_prompt:
            prompt += f"\nAvoid the following in the generated video: {negative_prompt.strip()}"

        input_content: list[dict[str, str]] = []
        image = tool_parameters.get("image")
        last_frame = tool_parameters.get("last_frame")
        ref_images = tool_parameters.get("ref_images") or []
        ref_video = tool_parameters.get("ref_video")

        if ref_video:
            input_content.append(self._inline_media("video", ref_video))
        if image:
            input_content.append(self._inline_media("image", image))
        if last_frame:
            input_content.append(self._inline_media("image", last_frame))
        input_content.extend(self._inline_media("image", reference) for reference in ref_images)
        input_content.append({"type": "text", "text": prompt})

        return {
            "model": model,
            "input": input_content,
            "response_format": {
                "type": "video",
                "aspect_ratio": tool_parameters.get("aspect_ratio", "16:9"),
                "resolution": tool_parameters.get("resolution", "720p"),
                # URI delivery avoids the Interactions API's 4 MB inline limit.
                "delivery": "uri",
            },
            "background": False,
            # URI-delivered video output must be persisted by the API.
            "store": True,
            "stream": False,
        }

    @staticmethod
    def _inline_media(media_type: str, file: Any) -> dict[str, str]:
        return {
            "type": media_type,
            "data": base64.b64encode(file.blob).decode("ascii"),
            "mime_type": file.mime_type,
        }

    def _wait_for_file(self, video_uri: str, api_key: str, proxy_url: str | None) -> None:
        match = re.search(r"/files/([^/:?]+)", video_uri)
        if not match:
            raise InvokeError("video generation failed: invalid video URI returned")

        status_url = f"{FILES_URL}/{match.group(1)}"
        for _ in range(MAX_POLL_ATTEMPTS):
            file_info = self._request("get", status_url, api_key, proxy_url).json()
            state = file_info.get("state")
            if isinstance(state, dict):
                state = state.get("name")
            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise InvokeError("video generation failed while processing the output file")
            logger.info("Waiting for Omni video file to become active...")
            time.sleep(5)

        raise InvokeError("video generation timeout after 10 minutes")

    @staticmethod
    def _find_output_video(interaction: dict[str, Any]) -> dict[str, Any] | None:
        output_video = interaction.get("output_video")
        if isinstance(output_video, dict):
            return output_video

        for step in reversed(interaction.get("steps") or []):
            if step.get("type") != "model_output":
                continue
            for content in reversed(step.get("content") or []):
                if content.get("type") == "video":
                    return content
        return None

    @staticmethod
    def _interaction_error(interaction: dict[str, Any], fallback: str = "request failed") -> str:
        errors = interaction.get("errors") or []
        if errors:
            error = errors[0]
            detail = error.get("message") if isinstance(error, dict) else str(error)
            return f"video generation failed: {detail}"
        return f"video generation failed: {fallback}"

    @staticmethod
    def _request(
        method: str,
        url: str,
        api_key: str,
        proxy_url: str | None,
        **kwargs: Any,
    ) -> requests.Response:
        request_kwargs: dict[str, Any] = {
            "headers": {"x-goog-api-key": api_key},
            "timeout": REQUEST_TIMEOUT,
            **kwargs,
        }
        if proxy_url:
            request_kwargs["proxies"] = {"http": proxy_url, "https": proxy_url}

        try:
            response = requests.request(method, url, **request_kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            detail = str(error)
            if error.response is not None:
                try:
                    detail = error.response.json().get("error", {}).get("message", detail)
                except ValueError:
                    pass
            raise InvokeError(f"Gemini API request failed: {detail}") from error

    def _generate_with_veo(
        self,
        tool_parameters: dict[str, Any],
        model: str,
        api_key: str,
        proxy_url: str | None,
    ) -> bytes:
        source = types.GenerateVideosSource()
        config = types.GenerateVideosConfig()
        self._configure_veo(tool_parameters, config, source)

        if proxy_url:
            genai_client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(httpx_client=httpx.Client(proxy=proxy_url)),
            )
        else:
            genai_client = genai.Client(api_key=api_key)

        operation: types.GenerateVideosOperation = genai_client.models.generate_videos(
            model=model, source=source, config=config
        )
        for _ in range(60):
            if operation.done:
                break
            logger.info("Waiting for Veo video generation to complete...")
            time.sleep(10)
            operation = genai_client.operations.get(operation)
            if operation.error:
                raise InvokeError(f"video generation failed: {operation.error.message}")
        else:
            raise InvokeError("video generation timeout after 10 minutes")

        generated_videos = operation.response.generated_videos if operation.response else None
        if not generated_videos:
            raise InvokeError("video generation failed: no video data returned")
        video = generated_videos[0].video

        if proxy_url:
            if not video.uri:
                raise InvokeError("video generation failed: no video URI returned")
            return self._request("get", video.uri, api_key, proxy_url).content

        genai_client.files.download(file=video)
        if not video.video_bytes:
            raise InvokeError("video generation failed: downloaded video is empty")
        return video.video_bytes

    @staticmethod
    def _configure_veo(
        tool_parameters: dict[str, Any],
        config: types.GenerateVideosConfig,
        source: types.GenerateVideosSource,
    ) -> None:
        image = tool_parameters.get("image")
        last_frame = tool_parameters.get("last_frame")
        ref_images = tool_parameters.get("ref_images") or []
        ref_video = tool_parameters.get("ref_video")

        source.prompt = tool_parameters["prompt"]
        source.image = types.Image(image_bytes=image.blob, mime_type=image.mime_type) if image else None
        source.video = types.Video(video_bytes=ref_video.blob, mime_type=ref_video.mime_type) if ref_video else None

        config.duration_seconds = GeminiVideoTool._duration_seconds(tool_parameters)
        config.aspect_ratio = tool_parameters.get("aspect_ratio", "16:9")
        config.resolution = tool_parameters.get("resolution", "720p")
        config.negative_prompt = tool_parameters.get("negative_prompt")
        config.last_frame = (
            types.Image(image_bytes=last_frame.blob, mime_type=last_frame.mime_type)
            if last_frame
            else None
        )
        config.reference_images = (
            [
                types.VideoGenerationReferenceImage(
                    image=types.Image(image_bytes=reference.blob, mime_type=reference.mime_type),
                    reference_type="asset",
                )
                for reference in ref_images
            ]
            if ref_images
            else None
        )

    @staticmethod
    def _validate_parameters(tool_parameters: dict[str, Any], model: str) -> None:
        prompt = tool_parameters.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvokeError("prompt is required")

        aspect_ratio = tool_parameters.get("aspect_ratio", "16:9")
        resolution = tool_parameters.get("resolution", "720p")
        duration_seconds = GeminiVideoTool._duration_seconds(tool_parameters)
        image = tool_parameters.get("image")
        last_frame = tool_parameters.get("last_frame")
        ref_images = tool_parameters.get("ref_images") or []
        ref_video = tool_parameters.get("ref_video")

        if aspect_ratio not in {"16:9", "9:16"}:
            raise InvokeError(f"aspect_ratio:{aspect_ratio} is not supported")
        if duration_seconds not in {4, 6, 8}:
            raise InvokeError(f"duration_seconds:{duration_seconds} is not supported")
        if len(ref_images) > 3:
            raise InvokeError("ref_images count can not be more than 3")

        if model in OMNI_MODELS:
            if resolution not in {"360p", "720p", "1080p", "4k"}:
                raise InvokeError(f"resolution:{resolution} is not supported by {model}")
            if last_frame and not image:
                raise InvokeError("first image is required when last_frame is set")
            return

        if resolution not in {"720p", "1080p", "4k"}:
            raise InvokeError(f"resolution:{resolution} is not supported by {model}")
        if last_frame and not image:
            raise InvokeError("first image is required when last_frame is set")
        if ref_images and (image or last_frame or ref_video):
            raise InvokeError("Veo reference images cannot be combined with image, last_frame, or ref_video")
        if resolution in {"1080p", "4k"} and duration_seconds != 8:
            raise InvokeError(f"{resolution} requires an 8-second duration for Veo")
        if (ref_images or ref_video) and duration_seconds != 8:
            raise InvokeError("Veo reference images and video extension require an 8-second duration")
        if ref_video and resolution != "720p":
            raise InvokeError("Veo video extension only supports 720p resolution")
        if model == "veo-3.1-lite-generate-preview":
            if resolution == "4k":
                raise InvokeError("Veo 3.1 Lite does not support 4k resolution")
            if ref_images or ref_video:
                raise InvokeError("Veo 3.1 Lite does not support reference images or video extension")

    @staticmethod
    def _duration_seconds(tool_parameters: dict[str, Any]) -> int:
        value = tool_parameters.get("duration_seconds", "4")
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise InvokeError(f"duration_seconds:{value} is invalid") from error
