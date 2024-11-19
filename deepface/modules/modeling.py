# built-in dependencies
from typing import Any


from deepface.models.face_detection import (
    MtCnn,
    OpenCv,
    Yolo,
   
)
from deepface.models.spoofing import FasNet


def build_model(task: str, model_name: str) -> Any:

    # singleton design pattern
    global cached_models

    models = {
        "spoofing": {
            "Fasnet": FasNet.Fasnet,
        },
        "face_detector": {
            "opencv": OpenCv.OpenCvClient,
            "mtcnn": MtCnn.MtCnnClient,
            "yolov8": Yolo.YoloClient,
        },
    }

    if models.get(task) is None:
        raise ValueError(f"unimplemented task - {task}")

    if not "cached_models" in globals():
        cached_models = {current_task: {} for current_task in models.keys()}

    if cached_models[task].get(model_name) is None:
        model = models[task].get(model_name)
        if model:
            cached_models[task][model_name] = model()
        else:
            raise ValueError(f"Invalid model_name passed - {task}/{model_name}")

    return cached_models[task][model_name]


if __name__ == "__main__":
        
        
        antispoof_model =  FasNet.Fasnet()
        print(antispoof_model)

