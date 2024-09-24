# coding: utf-8

import base64
import io
import zlib
from typing import Optional

import cv2
import numpy as np
import supervisely as sly
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

app = FastAPI()
server = app


class ImageDimensions(BaseModel):
    height: int
    width: int


class ImageMetadata(BaseModel):
    size: ImageDimensions


class MetadataResponse(BaseModel):
    data: Optional[ImageMetadata] = None
    error: Optional[str] = None


class ImageMetaReq(BaseModel):
    image: str


@server.post("/get-image-metadata")
def get_image_metadata(req: ImageMetaReq):
    tm = sly.TinyTimer()

    batch_result = []

    image_meta = MetadataResponse()

    image_data = req.image
    try:
        try:
            imencoded = zlib.decompress(base64.b64decode(image_data))
            n = np.frombuffer(imencoded, np.uint8)
            imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
        except zlib.error:
            # If the string is not compressed, we'll not use zlib.
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))
            imdecoded = np.array(img)

        imdecoded = cv2.cvtColor(imdecoded, cv2.COLOR_RGB2BGR)
        if imencoded is None:
            raise Exception("Can't read image.")

        size = ImageDimensions(height=imdecoded.shape[0], width=imdecoded.shape[1])
        image_meta.data = ImageMetadata(size=size)

    except Exception as exc:
        image_meta.error = str(exc)

    sly.logger.debug("Image processing done", extra={"durat_msec": tm.get_sec() * 1000.0})

    return image_meta
