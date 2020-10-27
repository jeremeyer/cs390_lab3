import torch
import torch.nn as nn
import torchvision
import numpy as np
import scipy
from torchvision import transforms
from PIL import Image


def load_img(path):
    preprocessor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.408],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(path)
    img_t = preprocessor(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t

#layer list
def gen_models():
    CONTENT_LAYER = 0
    STYLE_LAYERS = [-4, -8]

    #print(vgg19.features.state_dict())

    #content_network = nn.Module().load_state_dict(vgg19.state_dict())
    content_network = torchvision.models.vgg19(pretrained=True)
    content_network.classifier = nn.Sequential()
    content_network.features = content_network.features[:]
    content_network.eval()

    style_networks = []
    for layer_name in STYLE_LAYERS:
        #style_network = nn.Module().load_state_dict(vgg19.state_dict())
        style_network = torchvision.models.vgg19(pretrained=True)
        style_network.classifier = nn.Sequential()
        style_network.features = style_network.features[:layer_name]
        style_network.eval()
        style_networks.append(style_network)
    return (content_network, style_networks)

#print(style_networks[0].state_dict())

def run_model(network, image):
    return network(image).detach().numpy()

def gram_matrix(arr):
    return arr.flatten().T * arr.flatten()

def calc_content_loss(content_result, gen_result):
    return np.sum(np.square(gen_result - content_result))

def calc_style_loss(style_result, gen_result):
   pass 

def main():
    content_net, style_nets = gen_models()
    content_img = load_img('cat.jpg')
    style_img = load_img('starry_night.jpg')

    #forward pass
    out = run_model(content_net, content_img)
    print(out)
    
    for net in style_nets:
        print(run_model(net, style_img).shape)

if __name__ == '__main__':
    main()
    pass


