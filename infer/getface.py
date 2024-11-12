import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from models.mtcnn import MTCNN
# from facenet_pytorch import MTCNN

mtcnn = MTCNN(
image_size=160, margin=0, min_face_size=20,
thresholds=[0.5, 0.65, 0.65], factor=0.709, post_process=True
)

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo = YOLO(model_path)



def get_face_yolo(image_path, model):
    output = model(Image.open(image_path))
    image = Image.open(image_path)
    results = Detections.from_ultralytics(output[0])
    
    highest_confidence = 0
    best_cropped_image = None

    for i in range(len(results.xyxy)):
        x1, y1, x2, y2 = results.xyxy[i] 
        confidence = results.confidence[i]  
        class_name = results.data['class_name'][i]


        if confidence > highest_confidence:
            highest_confidence = confidence
            best_cropped_image = image.crop((x1, y1, x2, y2))

    return best_cropped_image, results.xyxy
    


def get_face_mtcnn(image_path, mtcnn):
    image = Image.open(image_path).convert('RGB')
    results = mtcnn.detect(np.array(image))

    best_cropped_image = None
    highest_prob = 0

    if results[0] is not None:
        boxes, probs = results 

        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = list(map(int, box))

            if prob > highest_prob:
                highest_prob = prob
                best_cropped_image = image.crop((x1, y1, x2, y2))

    return best_cropped_image, boxes



if __name__ == '__main__':
    
    model_select = input('Input the model yolo or mtcnn:')

    image_path = 'testdata/sontung/001.jpg'
    
    if model_select =='yolo':
        cropped_images, boxes = get_face_yolo(image_path, yolo)
    else: 
        cropped_images, boxes = get_face_mtcnn(image_path, mtcnn)

    plt.imshow(cropped_images)
    plt.show()
    print(boxes)
    
