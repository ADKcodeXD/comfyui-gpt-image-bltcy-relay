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
        raise ValueError("base_url 不能为空")

    base = base.rstrip("/")
    if base.endswith("/v1/images/generations") or base.endswith("/v1/images/edits"):
        if endpoint.endswith("/images/generations"):
            return re.sub(r"/v1/images/edits$", "/v1/images/generations", base)
        return re.sub(r"/v1/images/generations$", "/v1/images/edits", base)
    if base.endswith("/v1"):
        return f"{base}{endpoint[3:]}"
    return f"{base}{endpoint}"


def _make_auth_header_value(base_url, api_key, auth_mode="auto"):
    base = (_clean_text(base_url) or "").lower()
    key = _clean_text(api_key)
    if not key:
        return ""
    if key.lower().startswith("bearer "):
        return key
    if auth_mode == "bearer":
        return f"Bearer {key}"
    if auth_mode == "raw":
        return key
    if "openai.com" in base or "bltcy.ai" in base:
        return f"Bearer {key}"
    return key


def _split_reference_urls(value):
    text = _clean_text(value)
    if not text:
        return []
    parts = re.split(r"[\r\n,]+", text)
    return [item.strip() for item in parts if item.strip()]


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


def _resolve_image_payload_mode(base_url, parameter_form, image_payload_mode):
    if image_payload_mode != "auto":
        return image_payload_mode

    base = (_clean_text(base_url) or "").lower()
    if "openai.com" in base or "bltcy.ai" in base:
        return "data_url"
    if parameter_form == "official":
        return "data_url"
    return "base64"


def _tensor_batch_to_list(images):
    if images is None:
        return []
    if images.ndim == 3:
        return [images]
    return [image for image in images]


def _tensor_to_uint8_rgb(image_tensor):
    image_np = image_tensor.detach().cpu().numpy()
    if image_np.ndim != 3 or image_np.shape[-1] < 3:
        raise ValueError("参考图必须是 HxWxC 的 IMAGE 类型")
    image_np = np.clip(image_np[..., :3] * 255.0, 0, 255).astype(np.uint8)
    return image_np


def _tensor_to_png_bytes(image_tensor):
    image = Image.fromarray(_tensor_to_uint8_rgb(image_tensor), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _tensor_to_data_url(image_tensor):
    png_bytes = _tensor_to_png_bytes(image_tensor)
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _tensor_to_base64_string(image_tensor):
    png_bytes = _tensor_to_png_bytes(image_tensor)
    return base64.b64encode(png_bytes).decode("utf-8")


def _mask_to_png_bytes(mask_tensor, reference_tensor):
    mask_np = mask_tensor.detach().cpu().numpy()
    mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        raise ValueError("mask 必须是单通道 MASK")

    ref_h, ref_w = reference_tensor.shape[0], reference_tensor.shape[1]
    if mask_np.shape != (ref_h, ref_w):
        raise ValueError("mask 和第一张参考图尺寸必须一致")

    rgba = np.ones((ref_h, ref_w, 4), dtype=np.uint8) * 255
    rgba[..., 3] = np.clip((1.0 - mask_np) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_data_url(data_url):
    if not data_url.startswith("data:"):
        raise ValueError("不支持的 data URL")
    try:
        _, encoded = data_url.split(",", 1)
    except ValueError as exc:
        raise ValueError("data URL 格式不正确") from exc
    return base64.b64decode(encoded)


def _download_bytes(url, timeout, api_key=None, base_url=None, auth_mode="auto"):
    headers = {
        "User-Agent": "ComfyUI-GPTImageCustomBaseURL/1.0",
    }
    if api_key:
        headers["Authorization"] = _make_auth_header_value(base_url or url, api_key, auth_mode=auth_mode)

    req = request.Request(url, method="GET", headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"下载结果图片失败: HTTP {exc.code} url={url}\n{detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"下载结果图片失败: url={url} error={exc}") from exc


def _load_image_bytes_from_entry(entry, timeout, api_key=None, base_url=None, auth_mode="auto"):
    if isinstance(entry, str):
        if entry.startswith("data:"):
            return _decode_data_url(entry)
        if entry.startswith("http://") or entry.startswith("https://"):
            return _download_bytes(entry, timeout, api_key=api_key, base_url=base_url, auth_mode=auth_mode)
        try:
            return base64.b64decode(entry)
        except Exception as exc:
            raise ValueError("无法识别返回的图片字段") from exc

    if not isinstance(entry, dict):
        raise ValueError(f"不支持的图片结果格式: {type(entry)}")

    if entry.get("b64_json"):
        return base64.b64decode(entry["b64_json"])
    if entry.get("url"):
        return _download_bytes(entry["url"], timeout, api_key=api_key, base_url=base_url, auth_mode=auth_mode)
    if entry.get("data"):
        value = entry["data"]
        if isinstance(value, str):
            if value.startswith("data:"):
                return _decode_data_url(value)
            return base64.b64decode(value)

    raise ValueError("响应中没有可用的 b64_json/url/data 字段")


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
    raise ValueError("接口响应中没有找到图片数据")


def _pil_to_image_and_mask_tensors(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    rgba = image.convert("RGBA")
    rgba_np = np.array(rgba).astype(np.float32) / 255.0

    rgb_tensor = torch.from_numpy(rgba_np[..., :3]).unsqueeze(0)
    alpha_tensor = torch.from_numpy(1.0 - rgba_np[..., 3]).unsqueeze(0)
    return rgb_tensor, alpha_tensor


def _parse_extra_body_json(extra_body_json):
    extra_text = _clean_text(extra_body_json)
    if not extra_text:
        return {}
    if extra_text.lower() in ("null", "none", "undefined"):
        return {}

    try:
        extra_obj = json.loads(extra_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extra_body_json 不是合法 JSON: {exc}") from exc

    if extra_obj is None:
        return {}
    if isinstance(extra_obj, list) and len(extra_obj) == 0:
        return {}
    if isinstance(extra_obj, str):
        nested_text = extra_obj.strip()
        if not nested_text or nested_text.lower() in ("null", "none", "undefined"):
            return {}
        try:
            nested_obj = json.loads(nested_text)
        except json.JSONDecodeError:
            raise ValueError("extra_body_json 必须是 JSON 对象")
        if nested_obj is None:
            return {}
        if isinstance(nested_obj, dict):
            return nested_obj
        if isinstance(nested_obj, list) and len(nested_obj) == 0:
            return {}
        raise ValueError("extra_body_json 必须是 JSON 对象")

    if not isinstance(extra_obj, dict):
        raise ValueError("extra_body_json 必须是 JSON 对象")

    return extra_obj


def _drop_none_fields(data):
    return {key: value for key, value in data.items() if value is not None}


def _http_json(url, api_key, payload, timeout, base_url=None, auth_mode="auto"):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": _make_auth_header_value(base_url or url, api_key, auth_mode=auth_mode),
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"接口请求失败: HTTP {exc.code}\n{detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"接口连接失败: {exc}") from exc


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
    disposition = f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
    body.extend(disposition.encode("utf-8"))
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(content)
    body.extend(b"\r\n")


def _http_multipart(url, api_key, fields, files, timeout, base_url=None, auth_mode="auto"):
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
            "Authorization": _make_auth_header_value(base_url or url, api_key, auth_mode=auth_mode),
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"接口请求失败: HTTP {exc.code}\n{detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"接口连接失败: {exc}") from exc


class GPTImageCustomBaseURL:
    DEFAULT_N = 1
    DEFAULT_OUTPUT_COMPRESSION = 100

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "default": "https://api.bltcy.ai/v1/images/generations",
                        "multiline": False,
                        "tooltip": "支持根地址、/v1，或完整的 /v1/images/... 地址",
                    },
                ),
                "auth_mode": (
                    ["auto", "bearer", "raw"],
                    {
                        "default": "auto",
                        "tooltip": "auto 按域名判断；bearer 强制加 Bearer；raw 直接发送原始 Authorization 值",
                    },
                ),
                "api_key": ("STRING", {"default": "sk-kvq2S3R5geWcjdBFzpzHWBQKuJv8YCIyMKhtbBN6g8TkxQ1y", "multiline": False}),
                "model": ("STRING", {"default": "gpt-image-1.5", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "num_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Client-side repeated generation count. The node sends one request per image and merges results.",
                    },
                ),
                "request_mode": (
                    ["auto", "generations", "edits"],
                    {
                        "default": "auto",
                        "tooltip": "auto 默认始终走 generations；只有手动选择 edits 才会调用编辑接口",
                    },
                ),
                "parameter_form": (
                    ["relay", "official"],
                    {
                        "default": "relay",
                        "tooltip": "relay 使用中转站最小参数格式；official 使用 OpenAI 官方图片参数格式",
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
                        "1920x1088 (FHD landscape 16:9 aligned)",
                        "1088x1920 (FHD portrait 9:16 aligned)",
                        "2048x2048 (2K square)",
                        "2048x1152 (2K landscape)",
                        "1440x2560 (2K portrait 9:16)",
                        "3840x2160 (4K landscape)",
                        "2160x3840 (4K portrait)",
                    ],
                    {"default": "auto (default)"},
                ),
                "image_payload_mode": (
                    ["auto", "base64", "data_url"],
                    {
                        "default": "auto",
                        "tooltip": "How to encode each image entry in generations mode. auto prefers data URLs for official/OpenAI-style relays.",
                    },
                ),
                "aspect_ratio": (
                    ["auto", "1:1", "16:9", "9:16", "4:3", "3:4"],
                    {
                        "default": "auto",
                        "tooltip": "中转站 generations 通用接口常见扩展参数，OpenAI 官方会忽略不支持的字段",
                    },
                ),
                "official_quality": (
                    ["auto", "low", "medium", "high"],
                    {
                        "default": "auto",
                        "tooltip": "仅 official 表单生效。OpenAI 文档支持 low/medium/high/auto",
                    },
                ),
                "official_background": (
                    ["auto", "opaque", "transparent"],
                    {
                        "default": "auto",
                        "tooltip": "仅 official 表单生效。部分模型不支持 transparent",
                    },
                ),
                "official_moderation": (
                    ["auto", "low"],
                    {
                        "default": "auto",
                        "tooltip": "仅 official 表单生效。OpenAI 文档支持 auto/low",
                    },
                ),
                "official_output_format": (
                    ["png", "jpeg", "webp"],
                    {
                        "default": "png",
                        "tooltip": "仅 official 表单生效。jpeg/webp 会使用固定 output_compression",
                    },
                ),
                "official_input_fidelity": (
                    ["auto", "low", "high"],
                    {
                        "default": "auto",
                        "tooltip": "仅 official 表单生效，主要用于 edits/参考图工作流；gpt-image-2 官方建议通常省略",
                    },
                ),
                "extra_body_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "relay 模式 edits 时可附加 JSON 对象；official 表单会忽略它",
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
    DESCRIPTION = "兼容 GPT Image 风格参数的图片生成节点，支持自定义 base_url、参考图、可选 mask，以及中转站常见 generations 扩展字段。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Remote generation nodes should execute every time the prompt is queued.
        return time.time()

    def generate(
        self,
        base_url,
        auth_mode,
        api_key,
        model,
        prompt,
        num_images,
        request_mode,
        parameter_form,
        size,
        image_payload_mode,
        aspect_ratio,
        official_quality,
        official_background,
        official_moderation,
        official_output_format,
        official_input_fidelity,
        extra_body_json,
        timeout_seconds,
        image=None,
        mask=None,
    ):
        api_key = _clean_text(api_key)
        if not api_key:
            raise ValueError("api_key 不能为空")

        prompt = _clean_text(prompt)
        if not prompt:
            raise ValueError("prompt 不能为空")

        model = _clean_text(model)
        if not model:
            raise ValueError("model 不能为空")

        timeout = int(timeout_seconds)
        output_compression = self.DEFAULT_OUTPUT_COMPRESSION
        n = self.DEFAULT_N
        num_images = int(num_images)
        normalized_size = _normalize_size_value(size)
        resolved_image_payload_mode = _resolve_image_payload_mode(base_url, parameter_form, image_payload_mode)
        local_reference_tensors = _tensor_batch_to_list(image)
        if len(local_reference_tensors) > 16:
            raise ValueError("最多支持 16 张参考图")

        if mask is not None and not local_reference_tensors:
            raise ValueError("使用 mask 时必须至少传入 1 张 image")

        resolved_mode = request_mode
        if request_mode == "auto":
            resolved_mode = "generations"

        if mask is not None and resolved_mode != "edits":
            raise ValueError("mask 仅支持 edits 模式；如需使用 mask，请将 request_mode 手动切到 edits")

        if resolved_mode == "edits" and not local_reference_tensors:
            raise ValueError("edits 模式至少需要 1 张 image")

        common_payload = _drop_none_fields(
            {
                "model": model,
                "prompt": prompt,
                "size": normalized_size,
                "aspect_ratio": None if aspect_ratio == "auto" else aspect_ratio,
            }
        )

        image_batches = []
        mask_batches = []
        response_payloads = []
        user_extra_body = _parse_extra_body_json(extra_body_json) if resolved_mode == "edits" else {}

        for _ in range(num_images):
            if resolved_mode == "generations":
                endpoint = "/v1/images/generations"
                request_url = _normalize_base_url(base_url, endpoint)
                request_payload = dict(common_payload)

                if parameter_form == "official":
                    request_payload = _drop_none_fields(
                        {
                            "model": model,
                            "prompt": prompt,
                            "n": n,
                            "size": normalized_size,
                            "quality": None if official_quality == "auto" else official_quality,
                            "background": None if official_background == "auto" else official_background,
                            "moderation": None if official_moderation == "auto" else official_moderation,
                            "output_format": official_output_format,
                            "output_compression": (
                                output_compression if official_output_format in ("jpeg", "webp") else None
                            ),
                        }
                    )

                if resolved_image_payload_mode == "data_url":
                    request_payload["image"] = [_tensor_to_data_url(img) for img in local_reference_tensors]
                else:
                    request_payload["image"] = [_tensor_to_base64_string(img) for img in local_reference_tensors]
                if not request_payload["image"]:
                    request_payload.pop("image")

                response_payload = _http_json(
                    request_url, api_key, request_payload, timeout, base_url=base_url, auth_mode=auth_mode
                )
            else:
                endpoint = "/v1/images/edits"
                request_url = _normalize_base_url(base_url, endpoint)
                form_fields = dict(common_payload)
                form_fields.pop("aspect_ratio", None)
                if parameter_form == "official":
                    form_fields = _drop_none_fields(
                        {
                            "model": model,
                            "prompt": prompt,
                            "n": n,
                            "size": normalized_size,
                            "quality": None if official_quality == "auto" else official_quality,
                            "background": None if official_background == "auto" else official_background,
                            "moderation": None if official_moderation == "auto" else official_moderation,
                            "output_format": official_output_format,
                            "output_compression": (
                                output_compression if official_output_format in ("jpeg", "webp") else None
                            ),
                            "input_fidelity": None if official_input_fidelity == "auto" else official_input_fidelity,
                        }
                    )
                elif user_extra_body:
                    form_fields["extra_body_json"] = user_extra_body

                form_files = []
                for index, ref_tensor in enumerate(local_reference_tensors):
                    form_files.append(
                        {
                            "field_name": "image[]" if parameter_form == "official" else "image",
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

                response_payload = _http_multipart(
                    request_url, api_key, form_fields, form_files, timeout, base_url=base_url, auth_mode=auth_mode
                )

            response_payloads.append(response_payload)
            entries = _extract_image_entries(response_payload)
            for entry in entries:
                image_bytes = _load_image_bytes_from_entry(
                    entry, timeout, api_key=api_key, base_url=base_url, auth_mode=auth_mode
                )
                image_tensor, alpha_tensor = _pil_to_image_and_mask_tensors(image_bytes)
                image_batches.append(image_tensor)
                mask_batches.append(alpha_tensor)

        if not image_batches:
            raise ValueError("接口没有返回任何图片")

        image_output = torch.cat(image_batches, dim=0)
        mask_output = torch.cat(mask_batches, dim=0)
        response_data = response_payloads[0] if len(response_payloads) == 1 else response_payloads
        response_json = json.dumps(response_data, ensure_ascii=False, indent=2)
        return (image_output, mask_output, response_json)


NODE_CLASS_MAPPINGS = {
    "GPTImageCustomBaseURL": GPTImageCustomBaseURL,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageCustomBaseURL": "GPT Image (Custom Base URL)",
}
