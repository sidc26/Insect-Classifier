from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import torch
from PIL import Image
import pandas as pd
import numpy as np
def transforms_validation(image):
    crop_size=224
    resize_size=256
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    interpolation=InterpolationMode.BILINEAR
    transforms_val = transforms.Compose(
                    [
                    transforms.Resize(resize_size, interpolation=interpolation),
                    transforms.CenterCrop(crop_size),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=mean, std=std)])
    image = Image.fromarray(np.uint8(image))
    image=transforms_val(image).reshape((1,3,224,224))
    return image

def evaluate(model,image):
    model.eval()
    device= torch.device('cpu')
    image=transforms_validation(image)
    df=pd.read_csv('classes.csv')
    scientific_name=list(df['genus']+' '+df['species'])
    role=list(df['Role in Ecosystem'])
    with torch.inference_mode():
            image = image.to(device, non_blocking=True)
            output = model(image)
            op = torch.nn.functional.softmax(output, dim=1)
            op_ix= torch.argmax(op)
            
            if(op[0][op_ix]>=0.97):
                return 'Scientific Name: '+scientific_name[op_ix]+'\nRole in Ecosystem: '+role[op_ix]
            else:
                return 'Maybe OOD. '+ '\nScientific Name: '+scientific_name[op_ix]+' \nRole in Ecosystem: '+role[op_ix]
