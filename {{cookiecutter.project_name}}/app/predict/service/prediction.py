import io
import torchvision.transforms as transforms
from PIL import Image

from app.architecture import classes
from app.architecture.NN import Net

net = Net()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                        transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def predict(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = net.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    
    return predicted_idx, classes[predicted_idx]