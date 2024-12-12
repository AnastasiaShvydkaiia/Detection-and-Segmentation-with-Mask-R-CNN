import cv2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision import transforms
import torch
import numpy as np
import random
import json

def random_colour_masks(image):
    """
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],
               [80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

# Initialize a model
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
model=model.eval()

classnames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
    'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'
]
# List for model predictions
model_predictions = []

annotations_out_path = 'mrcnn_annotations.json'
input_video_path='input_video.mp4' # insert path to your video
output_video_path='output_video.mp4'

cap=cv2.VideoCapture(input_video_path)
# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Initialize VideoWriter
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
image_id=1

imgtransform =transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

while True:
    ret,frame=cap.read()
    image = imgtransform(frame)
    with torch.no_grad():
        ypred = model([image])
        bbox,scores,labels = ypred[0]['boxes'],ypred[0]['scores'],ypred[0]['labels']
        masks = (ypred[0]['masks']>0.5).squeeze().detach().numpy()
        nums = torch.argwhere(scores > 0.8).shape[0]
        for i in range(nums):
            x,y,w,h = bbox[i].numpy().astype('int')
            classname = labels[i].numpy().astype('int')
            classdetected = classnames[classname-1]

            cv2.rectangle(frame, (x,y),(w,h),color=(0, 255, 0), thickness=1)
            cv2.putText(frame,f"{classdetected}: {round(float(scores[i]),2)}", [x,y], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),thickness=2)
            rgb_mask = random_colour_masks(masks[i])
            frame = cv2.addWeighted(frame, 1, rgb_mask, 0.5, 0)
            # Save predictions in a dictionary
            ann_dict = {
                "image_id": image_id,
                "category_id": int(classname-1),
                "bbox": [float(x), float(y), float(w - x), float(h - y)],
                'score': float(scores[i])
                    }
            model_predictions.append(ann_dict)
            
    cv2.imshow('frame' ,frame)
    # Write video
    out.write(frame)
    image_id+=1
    if cv2.waitKey(10)&0xFF==ord('q'):
        break
# Save model predictions in a JSON file
json.dump(model_predictions, open(annotations_out_path, 'w'))
cap.release()
cv2.destroyAllWindows()