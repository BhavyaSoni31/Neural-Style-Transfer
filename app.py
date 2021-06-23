import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from io import BytesIO
import base64
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import time
import copy
import os
import math
output_images = []
# st.image(images,width=224)



st.title("ART WORK")
st.write("Hello ... Create your own art by just uploading your images !")
st.write("")
st.write("")
st.write("Processing will be slow because free host doesn't provide GPU support so it will be running on CPU so have some patience!")
' ### PLEASE TAKE A LOOK TO BELOW EXAMPLE IMAGES .'
st.write("")
st.write("")
st.image(Image.open("kedarnath1.jpg"), caption='Example of Content Image.', width=224)
st.image(Image.open("style06.jpg"), caption='Example of Style Image.', width=224)
st.image(Image.open("Style06Kedarnath1.jpg"), caption='Example of Output Image.', width=224)

st.markdown("Choose images for content image and style image from sidebar")
st.sidebar.markdown("Content image here")
content_img = st.sidebar.file_uploader("content image",type = 'jpg')

if content_img is not None:
    content_image_show = Image.open(content_img)
    st.image(content_image_show, caption='Uploaded Content Image.', width=224)


st.sidebar.markdown("Style image here")
style_img = st.sidebar.file_uploader("style image",type = 'jpg')
st.sidebar.markdown("Your uploaded photos will be shown to you on main page.")
if style_img is not None:
    style_image_show = Image.open(style_img)
    st.image(style_image_show, caption='Uploaded Style Image.', width=224)

device = "cpu"

original_shape = None
imsize = (512,512)
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    global original_shape
    image = Image.open(image_name)
    original_shape = image.size
    print("Original shape",original_shape)
    image = loader(image).unsqueeze(0)   # [B,C,H,W]
    return image.to(device, torch.float)

def getImage(content,style):

    style_img = image_loader(style)
    content_img = image_loader(content)
    assert style_img.size() == content_img.size()
    return content_img,style_img

if content_img and style_img is not None:
    content_img,style_img = getImage(content_img,style_img)

    unloader = transforms.ToPILImage()  # reconvert into PIL image


    class ContentLoss(nn.Module):

        def __init__(self, target):
            super(ContentLoss, self).__init__()
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)

        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    class StyleLoss(nn.Module):

        def __init__(self, target_feature,w = 0.2):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
            self.w = w

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)*self.w
            return input

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)
        normalization = Normalization(normalization_mean, normalization_std).to(device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
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
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

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

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]
        print(model)
        return model, style_losses, content_losses



    def get_image_download_link(img):
    	"""Generates a link allowing the PIL image to be downloaded
    	in:  PIL image
    	out: href string
    	"""
    	buffered = BytesIO()
    	img.save(buffered, format="JPEG")
    	img_str = base64.b64encode(buffered.getvalue()).decode()
    	href = f'<a href="data:file/txt;base64,{img_str}">Download result</a>'
    	return href
    input_img = content_img.clone()

    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer
    st.write("Progress bar:")
    latest_iteration = st.empty()
    bar = st.progress(0)
    latest_iteration.text('0')

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1):

        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        optimizer = get_input_optimizer(input_img)

        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()
                if (run[0]/3) > 99:
                    latest_iteration.text('100')
                    bar.progress(100)
                else:
                    latest_iteration.text(f'{math.floor(run[0]/3)}')
                    bar.progress(math.floor(run[0]/3))
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                    st.write("after ",run[0],"iteration ")
                    image = input_img.squeeze(0)
                    image = unloader(image)
                    image = image.resize(original_shape)
                    st.image(image)
                    st.markdown(get_image_download_link(image), unsafe_allow_html=True)
                return style_score + content_score

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)
        return input_img

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)
