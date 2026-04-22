import base64
import json
import re
import time
import uuid
from io import BytesIO
from urllib import error, request

import numpy as np
import torch
from PIL import Image


def _clean_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_base_url(base_url, endpoint):
    base = _clean_text(base_url)
    if not base:
        raise ValueError("base_url cannot be empty")

    base = base.rstrip("/")
    if base.endswith("/v1/images/generations") or base.endswith("/v1/images/edits"):
        if endpoint.endswith("/images/generations"):
            return re.sub(r"/v1/images/edits$", "/v1/images/generations", base)
        return re.sub(r"/v1/images/generations$", "/v1/images/edits", base)
    if base.endswith("/v1"):
        return f"{base}{endpoint[3:]}"
    return f"{base}{endpoint}"


def _normalize_size_value(size_value):
    value = _clean_text(size_value)
    if not value:
        return None
    if value == "auto" or value.startswith("auto "):
        return None
    match = re.match(r"^(\d+x\d+)", value)
    if match:
        return match.group(1)
    return value


def _tensor_batch_to_list(images):
    if images is None:
        return []
    if images.ndim == 3:
        return [images]
    return [image for image in images]


def _tensor_to_uint8_rgb(image_tensor):
    image_np = image_tensor.detach().cpu().numpy()
    if image_np.ndim != 3 or image_np.shape[-1] < 3:
        raise ValueError("Reference image must be an IMAGE tensor in HxWxC format")
    return np.clip(image_np[..., :3] * 255.0, 0, 255).astype(np.uint8)


def _tensor_to_png_bytes(image_tensor):
    image = Image.fromarray(_tensor_to_uint8_rgb(image_tensor), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _tensor_to_base64_string(image_tensor):
    return base64.b64encode(_tensor_to_png_bytes(image_tensor)).decode("utf-8")


def _mask_to_png_bytes(mask_tensor, reference_tensor):
    mask_np = np.squeeze(mask_tensor.detach().cpu().numpy())
    if mask_np.ndim != 2:
        raise ValueError("mask must be a single-channel MASK tensor")

    ref_h, ref_w = reference_tensor.shape[0], reference_tensor.shape[1]
    if mask_np.shape != (ref_h, ref_w):
        raise ValueError("mask size must match the first input image")

    rgba = np.ones((ref_h, ref_w, 4), dtype=np.uint8) * 255
    rgba[..., 3] = np.clip((1.0 - mask_np) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_data_url(data_url):
    if not data_url.startswith("data:"):
        raise ValueError("Unsupported data URL")
    try:
        _, encoded = data_url.split(",", 1)
    except ValueError as exc:
        raise ValueError("Invalid data URL") from exc
    return base64.b64decode(encoded)


def _download_bytes(url, timeout, api_key=None):
    headers = {"User-Agent": "ComfyUI-GPTImageCustomBaseURL/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(url, method="GET", headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except TimeoutError as exc:
        raise RuntimeError(f"Timed out while downloading result image: {url}") from exc
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to download result image: HTTP {exc.code} url={url}\n{detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to download result image: url={url} error={exc}") from exc


def _load_image_bytes_from_entry(entry, timeout, api_key=None):
    if isinstance(entry, str):
        if entry.startswith("data:"):
            return _decode_data_url(entry)
        if entry.startswith("http://") or entry.startswith("https://"):
            return _download_bytes(entry, timeout, api_key=api_key)
        try:
            return base64.b64decode(entry)
        except Exception as exc:
            raise ValueError("Unsupported image payload string") from exc

    if not isinstance(entry, dict):
        raise ValueError(f"Unsupported image payload type: {type(entry)}")

    if entry.get("b64_json"):
        return base64.b64decode(entry["b64_json"])
    if entry.get("url"):
        return _download_bytes(entry["url"], timeout, api_key=api_key)
    if entry.get("data"):
        value = entry["data"]
        if isinstance(value, str):
            if value.startswith("data:"):
                return _decode_data_url(value)
            return base64.b64decode(value)

    raise ValueError("Response does not contain a usable image field")


def _extract_image_entries(payload):
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("data"), dict):
            return _extract_image_entries(payload["data"])
        if isinstance(payload.get("images"), list):
            return payload["images"]
        if isinstance(payload.get("result"), (dict, list)):
            return _extract_image_entries(payload["result"])
        if any(key in payload for key in ("b64_json", "url", "data")):
            return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError("Response does not contain image data")


def _pil_to_image_and_mask_tensors(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    rgba = image.convert("RGBA")
    rgba_np = np.array(rgba).astype(np.float32) / 255.0
    rgb_tensor = torch.from_numpy(rgba_np[..., :3]).unsqueeze(0)
    alpha_tensor = torch.from_numpy(1.0 - rgba_np[..., 3]).unsqueeze(0)
    return rgb_tensor, alpha_tensor


def _parse_extra_body_json(extra_body_json):
    extra_text = _clean_text(extra_body_json)
    if not extra_text or extra_text.lower() in ("null", "none", "undefined"):
        return {}

    try:
        extra_obj = json.loads(extra_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extra_body_json is not valid JSON: {exc}") from exc

    if extra_obj is None:
        return {}
    if isinstance(extra_obj, list) and len(extra_obj) == 0:
        return {}
    if isinstance(extra_obj, str):
        nested_text = extra_obj.strip()
        if not nested_text or nested_text.lower() in ("null", "none", "undefined"):
            return {}
        try:
            extra_obj = json.loads(nested_text)
        except json.JSONDecodeError as exc:
            raise ValueError("extra_body_json must be a JSON object") from exc

    if extra_obj is None:
        return {}
    if isinstance(extra_obj, list) and len(extra_obj) == 0:
        return {}
    if not isinstance(extra_obj, dict):
        raise ValueError("extra_body_json must be a JSON object")
    return extra_obj


def _drop_none_fields(data):
    return {key: value for key, value in data.items() if value is not None}


def _http_json(url, api_key, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise RuntimeError(
            "Timed out while waiting for the relay API response. "
            "The relay may still be processing the image server-side."
        ) from exc
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Relay request failed: HTTP {exc.code}\n{detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Relay connection failed: {exc}") from exc


def _append_multipart_field(body, boundary, name, value):
    if isinstance(value, list):
        for item in value:
            _append_multipart_field(body, boundary, name, item)
        return

    if isinstance(value, (dict, bool)):
        value = json.dumps(value, ensure_ascii=False)
    elif value is None:
        return
    else:
        value = str(value)

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
    body.extend(value.encode("utf-8"))
    body.extend(b"\r\n")


def _append_multipart_file(body, boundary, field_name, filename, content_type, content):
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(content)
    body.extend(b"\r\n")


def _http_multipart(url, api_key, fields, files, timeout):
    boundary = f"----ComfyUIGPTImage{uuid.uuid4().hex}"
    body = bytearray()

    for key, value in fields.items():
        _append_multipart_field(body, boundary, key, value)
    for file_info in files:
        _append_multipart_file(
            body,
            boundary,
            file_info["field_name"],
            file_info["filename"],
            file_info["content_type"],
            file_info["content"],
        )

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise RuntimeError("Timed out while waiting for the relay edit response") from exc
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Relay edit request failed: HTTP {exc.code}\n{detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Relay edit connection failed: {exc}") from exc


class GPTImageCustomBaseURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "default": "https://api.bltcy.ai/v1/images/generations",
                        "multiline": False,
                        "tooltip": "Relay root URL, /v1 URL, or full images endpoint URL",
                    },
                ),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "gpt-image-1.5", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "request_mode": (
                    ["auto", "generations", "edits"],
                    {
                        "default": "auto",
                        "tooltip": "auto always uses generations; choose edits manually when needed",
                    },
                ),
                "size": (
                    [
                        "auto (default)",
                        "1024x1024 (square)",
                        "1536x1024 (landscape)",
                        "1024x1536 (portrait)",
                        "1280x720 (HD landscape 16:9)",
                        "720x1280 (HD portrait 9:16)",
                        "1920x1080 (FHD landscape 16:9)",
                        "1080x1920 (FHD portrait 9:16)",
                        "2048x2048 (2K square)",
                        "2048x1152 (2K landscape)",
                        "1440x2560 (2K portrait 9:16)",
                        "3840x2160 (4K landscape)",
                        "2160x3840 (4K portrait)",
                    ],
                    {"default": "auto (default)"},
                ),
                "aspect_ratio": (
                    ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                    {
                        "default": "auto",
                        "tooltip": "Relay-supported aspect ratio preset",
                    },
                ),
                "extra_body_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional JSON object passed only in edits mode",
                    },
                ),
                "timeout_seconds": ("INT", {"default": 300, "min": 5, "max": 600, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "alpha_mask", "response_json")
    FUNCTION = "generate"
    CATEGORY = "api/image"
    DESCRIPTION = (
        "ComfyUI relay node for BLTCY-style GPT image generation. "
        "The generations request uses model, prompt, size, aspect_ratio, and image."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def generate(
        self,
        base_url,
        api_key,
        model,
        prompt,
        request_mode,
        size,
        aspect_ratio,
        extra_body_json,
        timeout_seconds,
        image=None,
        mask=None,
    ):
        api_key = _clean_text(api_key)
        if not api_key:
            raise ValueError("api_key cannot be empty")

        prompt = _clean_text(prompt)
        if not prompt:
            raise ValueError("prompt cannot be empty")

        model = _clean_text(model)
        if not model:
            raise ValueError("model cannot be empty")

        timeout = int(timeout_seconds)
        normalized_size = _normalize_size_value(size)
        local_reference_tensors = _tensor_batch_to_list(image)
        if len(local_reference_tensors) > 16:
            raise ValueError("At most 16 reference images are supported")

        if mask is not None and not local_reference_tensors:
            raise ValueError("mask requires at least one input image")

        resolved_mode = "generations" if request_mode == "auto" else request_mode

        if mask is not None and resolved_mode != "edits":
            raise ValueError("mask is only supported in edits mode")
        if resolved_mode == "edits" and not local_reference_tensors:
            raise ValueError("edits mode requires at least one input image")

        common_payload = _drop_none_fields(
            {
                "model": model,
                "prompt": prompt,
                "size": normalized_size,
                "aspect_ratio": None if aspect_ratio == "auto" else aspect_ratio,
            }
        )

        if resolved_mode == "generations":
            request_url = _normalize_base_url(base_url, "/v1/images/generations")
            request_payload = dict(common_payload)
            request_payload["image"] = [_tensor_to_base64_string(img) for img in local_reference_tensors]
            if not request_payload["image"]:
                request_payload.pop("image")
            response_payload = _http_json(request_url, api_key, request_payload, timeout)
        else:
            request_url = _normalize_base_url(base_url, "/v1/images/edits")
            form_fields = dict(common_payload)
            form_fields.pop("aspect_ratio", None)
            user_extra_body = _parse_extra_body_json(extra_body_json)
            if user_extra_body:
                form_fields["extra_body_json"] = user_extra_body

            form_files = []
            for index, ref_tensor in enumerate(local_reference_tensors):
                form_files.append(
                    {
                        "field_name": "image",
                        "filename": f"reference_{index + 1}.png",
                        "content_type": "image/png",
                        "content": _tensor_to_png_bytes(ref_tensor),
                    }
                )

            if mask is not None:
                form_files.append(
                    {
                        "field_name": "mask",
                        "filename": "mask.png",
                        "content_type": "image/png",
                        "content": _mask_to_png_bytes(mask, local_reference_tensors[0]),
                    }
                )

            response_payload = _http_multipart(request_url, api_key, form_fields, form_files, timeout)

        entries = _extract_image_entries(response_payload)
        image_batches = []
        mask_batches = []
        for entry in entries:
            image_bytes = _load_image_bytes_from_entry(entry, timeout, api_key=api_key)
            image_tensor, alpha_tensor = _pil_to_image_and_mask_tensors(image_bytes)
            image_batches.append(image_tensor)
            mask_batches.append(alpha_tensor)

        if not image_batches:
            raise ValueError("The relay response did not contain any image output")

        image_output = torch.cat(image_batches, dim=0)
        mask_output = torch.cat(mask_batches, dim=0)
        response_json = json.dumps(response_payload, ensure_ascii=False, indent=2)
        return (image_output, mask_output, response_json)


NODE_CLASS_MAPPINGS = {
    "GPTImageCustomBaseURL": GPTImageCustomBaseURL,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageCustomBaseURL": "GPT Image (Custom Base URL)",
}
