# Insect-Classifier

## Loading libraries

You are required to load the following libraries to load the classifier model
```{r test-python, engine='python'}
import torch
import torchvision
import evaluate
```

## Creating the classifier model

```{r test-python, engine='python'}
model=torchvision.models.regnet_y_32gf()
```

## Loading weights of the model

```{r test-python, engine='python'}
weights=torch.load(PATH_TO_PTH_FILE+'/model.pth',map_location=torch.device('cpu'))['model']
model.fc=torch.nn.Linear(3712,142)
model.load_state_dict(weights,strict=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model.eval()
```

## Function to load the image and predict Insect class

```{r test-python, engine='python'}
image=cv2.imread(PATH_TO_IMAGE)
result=evaluate(model,image)
print(result)          
```


