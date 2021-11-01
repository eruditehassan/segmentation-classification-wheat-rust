
from os import stat
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from PIL import Image
from torch.autograd import Variable
from matplotlib import pyplot as plt
import time
import argparse



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='data/1.jpg',
						help='input the path to image for inference (default: data/1.jpg)')

    parser.add_argument('--model-path', type=str, default='model/model.pt',
						help='input the path to model for inference (default: model/model.pt)')

    args = parser.parse_args()
    image_path = args.image_path
    model_path = args.model_path

    model = models.resnet18(pretrained=True)



    model.fc = torch.nn.Linear(in_features=512, out_features=3)
    loss_fn = torch.nn.CrossEntropyLoss()


    model.load_state_dict(torch.load(model_path))



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    #optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    imsize = 256
    loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

    def image_loader(image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        return image.cuda()  #assumes that you're using GPU



    # image = image_loader('./C_78.jpg')




    



    # s = time.time()
    # op = model(image)
    # pred = op.data.max(1, keepdim=True)[1]
    # e = time.time()



    # e-s


    #test_healthy = './data/1.jpg'




    def inference(img_path):
        image = image_loader(img_path)
        op = model(image)
        pred = int(op.data.max(1, keepdim=True)[1])
        if (pred == 0):
            return "Healthy"
        elif (pred == 1):
            return "Resistant"
        else:
            return "Susceptible"



    print(inference(image_path))

if __name__ == '__main__':
	main()



