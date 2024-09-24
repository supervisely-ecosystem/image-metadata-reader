# coding: utf-8

from typing import Optional

import io
import uuid
import cv2
import numpy as np
import supervisely as sly
from fastapi import FastAPI, Request
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


@server.post("/get-image-metadata")
async def get_image_metadata(req: Request):
    tm = sly.TinyTimer()

    req_id = req.headers.get("x-request-uid", uuid.uuid4())
    extra_log_meta={"requestUid": req_id}
    sly.logger.debug("Image processing started", extra=extra_log_meta)

    # read the raw binary data from the request body
    binary_data = await req.body()
    image_data = np.frombuffer(binary_data, np.uint8)
    image_meta = MetadataResponse()

    try:
        try:
            # convert binary data to a NumPy array
            imdecoded = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)

            if imdecoded is not None:
                # we can't just use IMREAD_UNCHANGED because there might be EXIF data
                # and we want opencv to transform the image for us

                channels_num = imdecoded.shape[2]
                cv_decode_flags = None

                if channels_num == 1:
                    cv_decode_flags = cv2.IMREAD_GRAYSCALE
                elif channels_num == 3 or channels_num == 4:
                    cv_decode_flags = cv2.IMREAD_COLOR

                if cv_decode_flags is not None:
                    imdecoded = cv2.imdecode(image_data, cv_decode_flags)
            else:
                # if opencv can't read the image, try to use PIL
                img = Image.open(io.BytesIO(image_data))
                imdecoded = np.array(img)

        except Exception as e:
            raise Exception(f"Can't read image. {str(e)}")

        size = ImageDimensions(height=imdecoded.shape[0], width=imdecoded.shape[1])
        image_meta.data = ImageMetadata(size=size)

    except Exception as exc:
        image_meta.error = str(exc)

    sly.logger.debug("Image processing finished", extra={**extra_log_meta, "responseTime": round(tm.get_sec() * 1000.0)})

    return image_meta
