# PytorchModuleStorage
### Easy to use API to store forward/backward features
*Francesco Saverio Zuppichini*

## Installation

```
pip install 

``

## Quick Start

You have a model, e.g. `vgg19` and you want to store the features in the third layer given an input `x`. 

![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModuleStorage/master/images/vgg-19.png)

First, we need a model. We will load `vgg19` from `torchvision.models`. Then, we create a random input `x`


```python
import torch

from torchvision.models import vgg19
from storage import ForwardModuleStorage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn = vgg19(False).to(device).eval()
```

Then, we define a `ForwardModuleStorage` instance by passing the model and the list of layer we are interested on.


```python
storage = ForwardModuleStorage(cnn, [cnn.features[3]])
```

Finally, we can pass a input to the `storage`.


```python
x = torch.rand(1,3,224,224).to(device) # random input, this can be an image
storage(x) PytorchStorage
storage[cnn.features[3]][0] # the features can be accessed by passing the layer as a key
```




    tensor([[[[0.0000, 0.0096, 0.0000,  ..., 0.0000, 0.0779, 0.0000],
              [0.0838, 0.0567, 0.0973,  ..., 0.0000, 0.1429, 0.0132],
              [0.0417, 0.0249, 0.0000,  ..., 0.0000, 0.0653, 0.0000],
              ...,
              [0.1135, 0.0429, 0.0000,  ..., 0.0000, 0.0187, 0.0000],
              [0.0000, 0.0715, 0.0000,  ..., 0.0140, 0.0000, 0.0000],
              [0.0569, 0.0000, 0.0228,  ..., 0.0000, 0.0000, 0.0000]],
    
             [[0.0000, 0.0998, 0.0073,  ..., 0.0000, 0.0725, 0.0000],
              [0.0538, 0.1496, 0.1861,  ..., 0.1608, 0.2325, 0.0000],
              [0.2112, 0.1708, 0.4880,  ..., 0.1965, 0.2087, 0.1108],
              ...,
              [0.0504, 0.0474, 0.1651,  ..., 0.3195, 0.1704, 0.1532],
              [0.2454, 0.2351, 0.2507,  ..., 0.1891, 0.3085, 0.0966],
              [0.0627, 0.1082, 0.1874,  ..., 0.1319, 0.3948, 0.1490]],
    
             [[0.0936, 0.1198, 0.1036,  ..., 0.2526, 0.1110, 0.0000],
              [0.1442, 0.0190, 0.1689,  ..., 0.2353, 0.0020, 0.0406],
              [0.0000, 0.1516, 0.0460,  ..., 0.0000, 0.1804, 0.0991],
              ...,
              [0.0507, 0.2950, 0.3272,  ..., 0.1901, 0.0700, 0.0000],
              [0.1223, 0.3540, 0.2394,  ..., 0.2161, 0.1435, 0.0186],
              [0.0000, 0.0350, 0.0000,  ..., 0.1354, 0.0000, 0.1089]],
    
             ...,
    
             [[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0668, 0.0193, 0.0044,  ..., 0.0000, 0.0000, 0.0000],
              ...,
              [0.0436, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0932, 0.0000, 0.0456,  ..., 0.0000, 0.0000, 0.0000],
              [0.2492, 0.1023, 0.1021,  ..., 0.1513, 0.0287, 0.0000]],
    
             [[0.1635, 0.1237, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0937, 0.0449, 0.1596,  ..., 0.0000, 0.0000, 0.0000],
              [0.0238, 0.0000, 0.1184,  ..., 0.0734, 0.0339, 0.0000],
              ...,
              [0.1878, 0.1007, 0.0138,  ..., 0.1992, 0.0000, 0.0000],
              [0.1491, 0.1331, 0.0044,  ..., 0.0611, 0.0000, 0.0015],
              [0.0541, 0.0000, 0.0000,  ..., 0.0958, 0.0000, 0.0000]],
    
             [[0.0457, 0.1096, 0.1747,  ..., 0.1116, 0.0000, 0.0792],
              [0.0000, 0.0703, 0.0524,  ..., 0.0000, 0.0440, 0.1196],
              [0.0000, 0.0000, 0.0000,  ..., 0.0373, 0.0000, 0.1906],
              ...,
              [0.0000, 0.0128, 0.1082,  ..., 0.0000, 0.1785, 0.1091],
              [0.0000, 0.1115, 0.0473,  ..., 0.0000, 0.0323, 0.0230],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0605, 0.0080]]]],
           grad_fn=<ReluBackward1>)



The storage keeps an internal `state` (`storage.state`) where we can use the layers as key to access the stored value.

### Hook to a list of layers
You can pass a list of layers and then access the stored outputs


```python
storage = ForwardModuleStorage(cnn, [cnn.features[3], cnn.features[5]])
x = torch.rand(1,3,224,224).to(device) # random input, this can be an image
storage(x) # pass the input to the storage
print(storage[cnn.features[3]][0].shape)
print(storage[cnn.features[5]][0].shape)
```

    torch.Size([1, 64, 224, 224])
    torch.Size([1, 128, 112, 112])


### Multiple Inputs

You can also pass multiple inputs, they will be stored using the call order

![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModuleStorage/master/images/vgg-19-1.png)


```python
storage = ForwardModuleStorage(cnn, [cnn.features[3]])
x = torch.rand(1,3,224,224).to(device) # random input, this can be an image
y = torch.rand(1,3,224,224).to(device) # random input, this can be an image
storage([x, y]) # pass the inputs to the storage
print(storage[cnn.features[3]][0].shape) # x
print(storage[cnn.features[3]][1].shape) # y
```

    torch.Size([1, 64, 224, 224])
    torch.Size([1, 64, 224, 224])


### Different inputs for different layers
Image we want to run `x` on a set of layers and `y` on an other, this can be done by specify a dictionary of `{ NAME: [layers...], ...}
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PytorchModuleStorage/master/images/vgg-19-2.png)


```python
storage = ForwardModuleStorage(cnn, {'style' : [cnn.features[5]], 'content' : [cnn.features[5], cnn.features[10]]})
storage(x, 'style') # we run x only on the 'style' layers
storage(y, 'content') # we run y only on the 'content' layers


print(storage['style']) 
print(storage['style'][cnn.features[5]])
```

    MutipleKeysDict([(Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), tensor([[[[0.0968, 0.0824, 0.0756,  ..., 0.0599, 0.0000, 0.0959],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0049, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0302],
              ...,
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0142],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0026],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0008]],
    
             [[0.1660, 0.0000, 0.0743,  ..., 0.0000, 0.0091, 0.0000],
              [0.0217, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0015, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              ...,
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]],
    
             [[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0705, 0.0371],
              [0.0300, 0.0381, 0.0000,  ..., 0.0272, 0.0000, 0.0014],
              [0.0086, 0.0331, 0.0628,  ..., 0.0000, 0.0386, 0.1134],
              ...,
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.1305, 0.0218],
              [0.0052, 0.0000, 0.0504,  ..., 0.0086, 0.1189, 0.0581],
              [0.0832, 0.0000, 0.0314,  ..., 0.0092, 0.0687, 0.0443]],
    
             ...,
    
             [[0.0000, 0.0000, 0.0553,  ..., 0.0467, 0.0618, 0.1097],
              [0.0000, 0.0130, 0.0000,  ..., 0.0000, 0.0000, 0.0050],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0518],
              ...,
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0406],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0031,  ..., 0.0000, 0.0000, 0.0330]],
    
             [[0.0348, 0.1152, 0.1537,  ..., 0.1757, 0.0895, 0.0753],
              [0.1088, 0.2074, 0.1711,  ..., 0.2581, 0.2460, 0.1418],
              [0.1159, 0.1025, 0.1937,  ..., 0.2325, 0.1241, 0.1589],
              ...,
              [0.0924, 0.1112, 0.2120,  ..., 0.1434, 0.1963, 0.1373],
              [0.1491, 0.1037, 0.2318,  ..., 0.1723, 0.2092, 0.1302],
              [0.0889, 0.1818, 0.0412,  ..., 0.0963, 0.1432, 0.1261]],
    
             [[0.2055, 0.2298, 0.2894,  ..., 0.1463, 0.2408, 0.0973],
              [0.1319, 0.3015, 0.2181,  ..., 0.2508, 0.1761, 0.1047],
              [0.1276, 0.2128, 0.1924,  ..., 0.2007, 0.0880, 0.0276],
              ...,
              [0.0818, 0.2478, 0.1866,  ..., 0.1827, 0.0966, 0.1312],
              [0.1621, 0.1849, 0.0898,  ..., 0.1612, 0.1358, 0.0713],
              [0.0068, 0.0632, 0.0062,  ..., 0.0539, 0.0000, 0.0419]]]],
           grad_fn=<ReluBackward1>))])
    tensor([[[[0.0968, 0.0824, 0.0756,  ..., 0.0599, 0.0000, 0.0959],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0049, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0302],
              ...,
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0142],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0026],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0008]],
    
             [[0.1660, 0.0000, 0.0743,  ..., 0.0000, 0.0091, 0.0000],
              [0.0217, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0015, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              ...,
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]],
    
             [[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0705, 0.0371],
              [0.0300, 0.0381, 0.0000,  ..., 0.0272, 0.0000, 0.0014],
              [0.0086, 0.0331, 0.0628,  ..., 0.0000, 0.0386, 0.1134],
              ...,
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.1305, 0.0218],
              [0.0052, 0.0000, 0.0504,  ..., 0.0086, 0.1189, 0.0581],
              [0.0832, 0.0000, 0.0314,  ..., 0.0092, 0.0687, 0.0443]],
    
             ...,
    
             [[0.0000, 0.0000, 0.0553,  ..., 0.0467, 0.0618, 0.1097],
              [0.0000, 0.0130, 0.0000,  ..., 0.0000, 0.0000, 0.0050],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0518],
              ...,
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0406],
              [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0031,  ..., 0.0000, 0.0000, 0.0330]],
    
             [[0.0348, 0.1152, 0.1537,  ..., 0.1757, 0.0895, 0.0753],
              [0.1088, 0.2074, 0.1711,  ..., 0.2581, 0.2460, 0.1418],
              [0.1159, 0.1025, 0.1937,  ..., 0.2325, 0.1241, 0.1589],
              ...,
              [0.0924, 0.1112, 0.2120,  ..., 0.1434, 0.1963, 0.1373],
              [0.1491, 0.1037, 0.2318,  ..., 0.1723, 0.2092, 0.1302],
              [0.0889, 0.1818, 0.0412,  ..., 0.0963, 0.1432, 0.1261]],
    
             [[0.2055, 0.2298, 0.2894,  ..., 0.1463, 0.2408, 0.0973],
              [0.1319, 0.3015, 0.2181,  ..., 0.2508, 0.1761, 0.1047],
              [0.1276, 0.2128, 0.1924,  ..., 0.2007, 0.0880, 0.0276],
              ...,
              [0.0818, 0.2478, 0.1866,  ..., 0.1827, 0.0966, 0.1312],
              [0.1621, 0.1849, 0.0898,  ..., 0.1612, 0.1358, 0.0713],
              [0.0068, 0.0632, 0.0062,  ..., 0.0539, 0.0000, 0.0419]]]],
           grad_fn=<ReluBackward1>)

