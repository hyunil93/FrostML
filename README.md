# FrostML



### Usage

```
import torch
from models import EfficientFormer

net = EfficientFormer.EfficientFormer_s12(num_class=1000)

img = torch.randn((2, 3, 704, 704))
output = net(img)
```
