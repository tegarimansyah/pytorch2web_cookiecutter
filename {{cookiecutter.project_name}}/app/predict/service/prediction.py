from app.architecture import trainloader, classes
from app.architecture.NN import Net

dataiter = iter(trainloader)
images, labels = dataiter.next()
net = Net()

def predict(image_bytes):
    image = transform_image(image_bytes)
    return 1, 'tshirt'

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)