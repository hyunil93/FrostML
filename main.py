import torch
from segmentation_model.backbone import *

net = EfficientNet.EfficientNet(num_classes=1000, img_size=704, pretrained=False)
# net = ReResSegNet.ReResSegNet(num_classes=1000, img_size=704)
# net = EfficientFormer.efficientformer_l3(num_classes=1000, img_size=704)
# net = volo.volo_d1(num_classes=1000, img_size=704)
# net = MobileNetV2.MobileNetV2(num_classes=1000, img_size=704)
# net = MAE_ViT.MAE_VIT_B16(num_classes=1000, img_size=704)
# net = Wide_ResNet.Wide_Resnet(num_classes=1000, img_size=704)
# net = unet.UNet(num_classes=1000, img_size=704)
# net = resnet_18.ResNet18(num_classes=1000, img_size=704)
# net = resnet_50.ResNet50(num_classes=1000, img_size=704)

img = torch.randn((2, 3, 704, 704))

output = net(img)
print(f"debug")