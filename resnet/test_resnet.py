# coding: utf-8


import torch
import urllib
from PIL import Image
from torchvision import transforms


def main():
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
    model.eval()
    filename = 'dog.jpg'
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 10)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

if __name__ == '__main__':
    main()

