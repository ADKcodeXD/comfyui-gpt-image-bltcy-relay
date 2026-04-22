# ComfyUI GPT Image BLTCY Relay

A lightweight ComfyUI custom node for GPT image generation through the `https://api.bltcy.ai/` relay format.

This node is designed around the BLTCY relay `images/generations` request shape:

```json
{
  "model": "string",
  "prompt": "string",
  "size": "string",
  "aspect_ratio": "string",
  "image": ["string"]
}
```

Local reference images are converted to base64 strings and sent through the top-level `image` array.

## Features

- Custom `base_url`
- BLTCY relay-compatible `images/generations` payload
- Local reference image support through the `image` input
- Optional `edits` mode with image + mask multipart upload
- ComfyUI cache bypass so each queue run triggers a fresh remote request
- Returns `IMAGE`, `MASK`, and raw `response_json`

## Install

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ADKcodeXD/comfyui-gpt-image-bltcy-relay.git
```

Restart ComfyUI, then search for:

`GPT Image (Custom Base URL)`

## Default Relay Endpoint

The node defaults to:

```text
https://api.bltcy.ai/v1/images/generations
```

You still need to provide your own relay API key in the node UI.

## Main Inputs

- `base_url`: relay root, `/v1`, or full images endpoint
- `api_key`: your relay API key
- `model`: model name such as `gpt-image-1.5`
- `prompt`: generation prompt
- `request_mode`: `auto`, `generations`, or `edits`
- `size`: preset output size
- `aspect_ratio`: optional relay aspect ratio
- `image`: optional local reference image batch
- `mask`: optional mask for `edits` mode
- `timeout_seconds`: request timeout

## Notes

- In `auto` mode, the node uses `generations`.
- In `generations` mode, the request stays aligned with the BLTCY relay parameter format.
- In `edits` mode, `extra_body_json` can be used to pass additional JSON fields if your relay supports them.

## Security

This repository intentionally does **not** ship with a default API key.

