import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont

image_name = 'pedestrian.jpg'

# Preprocessing steps from the documentation
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    imageWidth, imageHeight = image.size
    width, height = size
    scale = min(width/imageWidth, height/imageHeight)
    newWidth = int(imageWidth*scale)
    newHeight = int(imageHeight*scale)
    
    # Resize the original image
    image = image.resize((newWidth,newHeight), Image.BICUBIC)
    
    #Create blank canvas
    new_image = Image.new('RGB', size, (128,128,128))
    
    #Paste the resized image to the center of the blank canvas
    new_image.paste(image, ((width-newWidth)//2, (height-newHeight)//2))
    
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, model_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    
    #Normalize pixel values
    image_data /= 255.0
    
    image_data = np.transpose(image_data, [2, 0, 1])
    
    #Add another dimension (needed for the model)
    image_data = np.expand_dims(image_data, 0)
    
    return image_data

# Convert the box coordinates to the original image scale
def scale_boxes(boxes, image_shape):
    height, width = image_shape
    new_boxes = []
    for box in boxes:
        top, left, bottom, right = box
        
        # Add 0.5 to round to nearest integer
        top = max(0, np.floor(top + 0.5).astype(int))
        left = max(0, np.floor(left + 0.5).astype(int))
        bottom = min(height, np.floor(bottom + 0.5).astype(int))
        right = min(width, np.floor(right + 0.5).astype(int))
    
        new_boxes.append([top, left, bottom, right])
    return new_boxes

# Draw bounding boxes on the image
def draw_boxes(image, out_boxes, out_scores, out_classes, class_names):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    for i, box in enumerate(out_boxes):
        top, left, bottom, right = box
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)
        label = f"{class_names[out_classes[i]]} {out_scores[i]:.2f}"
        draw.text((left, top), label, fill=(255, 255, 255), font=font)

# Load and preprocess the image
img_path = image_name
image = Image.open(img_path)
image_data = preprocess(image)
image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)

# Complete list of class names for YOLOv3
with open("coco_classes.txt", "r") as file:
    class_names = file.readlines()

# Load the ONNX model with ONNX Runtime
onnx_model_path = 'yolov3.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# Get input names from model
input_name_1 = ort_session.get_inputs()[0].name  # First input (image)
input_name_2 = ort_session.get_inputs()[1].name  # Second input (image shape)

# Run the model with both inputs and return all outputs
boxes, scores, indices = ort_session.run(None, {input_name_1: image_data, input_name_2: image_size})

# Indices provide information about all the "relevant" scores

out_boxes, out_scores, out_classes = [], [], []

# Goes through indices one row at a time
for idx_ in indices:
    #Append the class id to out_classes from indices
    out_classes.append(idx_[1])
    
    #Append the score to out_scores using the indexs from indices
    out_scores.append(scores[tuple(idx_)])
    
    #Get the coordinates of the relevant box using valus from indices
    idx_1 = (idx_[0], idx_[2])
    out_boxes.append(boxes[idx_1])

# Call scale_boxes with the coordinates of the boxes and the original size of the image
scaled_boxes = scale_boxes(out_boxes, image.size)
    

# Draw the boxes on the original image
draw_boxes(image, scaled_boxes, out_scores, out_classes, class_names)

# Save or display the image with bounding boxes
image.show()