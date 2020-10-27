import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

import copy

import matplotlib.pyplot as plt

'''
based on the tutorial found here: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

many elements such as built-in matrix operation functions, the pytorch module paradigm, device/gpu selection, etc. was very difficult to learn by scouring through APIs
modified to use the lowest-loss image, as higher epoch runs have a tendency to diverge after finding a good transfer, using the lowest-loss image instead grabs the most converged version
'''



device = torch.device("cuda")

imsize = 500

#NOTE need to reorganize transformations and constants into dictionary
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

#deprecated names, nothing's more permanent than a temporary solution
test_var_one = None
test_var_two = None

#wrapper for image loading transformation
def image_loader(path):
    image = Image.open(path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    pass
pass

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)

    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

#module for encapsulating style layers for grabbing loss
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    pass
pass

#module for preprocessing image within the network
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

        #self.mean = mean.clone().detach()
        #self.std = std.clone().detach()

    def forward(self, img):
        return (img - self.mean) / self.std
    pass
pass

content_layers = ['conv_4'] #conv_4 originally
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] #originally 1, 2, 3, 4, 5

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    #reformat layer names for easy access and add loss layers
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        pass
    
        model.add_module(name, layer)

        #insert loss layers for backprop
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
    pass
        
    #extra layer removal
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    #optimization function to use with tensor
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps=300, style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    global test_var_one
    global test_var_two

    test_var_one = None
    test_var_two = float('inf')

    run = [0]
    while run[0] < num_steps:

        #function passed to optimizer, custom loss calculations defined here + can save mid-train data (in this case lowest loss image)
        def exec_epoch():

            global test_var_one
            global test_var_two

            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0.0
            content_score = 0.0

            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            print('\r[%s] loss: %s' % (run[0], loss.item()), end='')
            if (loss < test_var_two):
                #original tensor tied to optimization function, clone tensor and detach from modified vgg19 layers and optimizer
                test_var_one = input_img.clone().detach()
                #print("new loss low")
            loss.backward()

            run[0] += 1
            return style_score + content_score
        #pass in function for optimizer step
        optimizer.step(exec_epoch)
    #print(test_var_one)
    input_img.data.clamp_(0, 1)
    test_var_one.data.clamp_(0, 1)

    #return input_img
    return test_var_one

def transfer_run(style_path, content_path, start_image='noise', out_path=None, params=(1000, 1, 300)):
    transfer_net = models.vgg19(pretrained=True).features.to(device).eval()

    #grab data for preprocessing layer
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    #image loading
    content_img = image_loader(content_path)
    style_img = image_loader(style_path)

    #start with random noise image
    input_img = torch.rand(content_img.data.size(), device=device)
    #input_img = content_img.clone()

    print("style image size: %s" % str(style_img.size()))
    print("content image size: %s" % str(content_img.size()))

    print()

    output = run_style_transfer(transfer_net, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img, style_weight=params[0], content_weight=params[1], num_steps=params[2])

    print()

    unloader = transforms.ToPILImage()

    #copy image back to cpu
    image = output.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)

    if (out_path == None):

        plt.ion()

        plt.figure()
        plt.imshow(image)

        plt.ioff()
        plt.show()
    else:
        image.save(out_path)
    pass

def parse_args():
    options = {"start_image" : "noise", "out_path" : None, "alpha" : 1000, "beta" : 1, "steps" : 300}
    options["style_image"] = sys.argv[1]
    options["content_image"] = sys.argv[2]
    for i in range(3, len(sys.argv)):
        if (sys.argv[i] == "-o"):
            i += 1
            options["out_path"] = sys.argv[i]
        elif (sys.argv[i] == "-s"):
            i += 1
            options["start_image"] = sys.argv[i]
        elif (sys.argv[i] == "--alpha"):
            i += 1
            options["alpha"] = sys.argv[i]
        elif (sys.argv[i] == "--beta"):
            i += 1
            options["beta"] = sys.argv[i]
        elif (sys.argv[i] == "--steps"):
            i += 1
            options["steps"] = sys.argv[i]
        pass
    print(options)
    return options

def main():
    options = parse_args()
    transfer_run(options["style_image"], options["content_image"], start_image=options["start_image"], out_path=options["out_path"], params=(int(options["alpha"]), int(options["beta"]), int(options["steps"])))

if __name__ == '__main__':
    main()
