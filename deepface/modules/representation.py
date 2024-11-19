# built-in dependencies
from typing import Any, Dict, List, Union, Optional
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.commons import image_utils
from deepface.modules import modeling, detection, preprocessing
from deepface.models.FacialRecognition import FacialRecognition

def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
) -> List[Dict[str, Any]]:
   
    resp_objs = []

    model: FacialRecognition = modeling.build_model(
        task="facial_recognition", model_name=model_name
    )

    # ---------------------------------
    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    target_size = model.input_shape
    if detector_backend != "skip":
        img_objs = detection.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
    else:  # skip
        # Try load. If load error, will raise exception internal
        img, _ = image_utils.load_image(img_path)

        if len(img.shape) != 3:
            raise ValueError(f"Input img must be 3 dimensional but it is {img.shape}")

        # make dummy region and confidence to keep compatibility with `extract_faces`
        img_objs = [
            {
                "face": img,
                "facial_area": {"x": 0, "y": 0, "w": img.shape[0], "h": img.shape[1]},
                "confidence": 0,
            }
        ]

    if max_faces is not None and max_faces < len(img_objs):
        # sort as largest facial areas come first
        img_objs = sorted(
            img_objs,
            key=lambda img_obj: img_obj["facial_area"]["w"] * img_obj["facial_area"]["h"],
            reverse=True,
        )
        # discard rest of the items
        img_objs = img_objs[0:max_faces]

    for img_obj in img_objs:
        if anti_spoofing is True and img_obj.get("is_real", True) is False:
            raise ValueError("Spoof detected in the given image.")
        img = img_obj["face"]

        # rgb to bgr
        img = img[:, :, ::-1]

        region = img_obj["facial_area"]
        confidence = img_obj["confidence"]

        # resize to expected shape of ml model
        img = preprocessing.resize_image(
            img=img,
            # thanks to DeepId (!)
            target_size=(target_size[1], target_size[0]),
        )

        # custom normalization
        img = preprocessing.normalize_input(img=img, normalization=normalization)

        embedding = model.forward(img)

        resp_objs.append(
            {
                "embedding": embedding,
                "facial_area": region,
                "face_confidence": confidence,
            }
        )

    return resp_objs


if __name__ == '__main__':
    # img_obj =  img_objs = detection.extract_faces(
    #         img_path='testdata/thaotam/004.jpg',
    #     )


    # model: FacialRecognition = modeling.build_model(
    #     task="facial_recognition", model_name='VGG-Face'
    # ) 

    # img = img_obj[0]["face"]

    # # rgb to bgr
    # img = img[:, :, ::-1]

    # # resize to expected shape of ml model
    # img = preprocessing.resize_image(
    #     img=img,
    #     target_size=(model.input_shape[0], model.input_shape[1]),
    # )

    # # # custom normalization
    # # img = preprocessing.normalize_input(img=img, normalization='base')

    # # embedding = model.forward(img)

    # # print(len(embedding))

    # print(img.shape)



    antispoof_model = modeling.build_model(task="spoofing", model_name="Fasnet")

    print(antispoof_model)