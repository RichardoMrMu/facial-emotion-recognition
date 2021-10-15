from .vgg import *
from .resnet import *
from .resnet112 import resnet18x112
from .resnet50_scratch_dims_2048 import resnet50_pretrained_vgg
from .centerloss_resnet import resnet18_centerloss
from .resatt import *
from .alexnet import *
from .densenet import *
from .googlenet import *
from .inception import *
from .inception_resnet_v1 import *
from .residual_attention_network import *
from .fer2013_models import *
from .res_dense_gle import *
from .masking import masking
from .resmasking import (
    resmasking,
    resmasking_dropout1,
    resmasking_dropout2,
    resmasking50_dropout1,
)
from .swin_transformer import *
# from .resnest import *
from .resmasking_naive import resmasking_naive_dropout1
from .brain_humor import *
from .runet import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from timm.models import create_model
from .botnet import *
from .crossformer import CrossFormer
def resattnet56(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("resattnet56", pretrained=False)
    model.output = nn.Linear(2048, 7)
    return model


def cbam_resnet50(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("cbam_resnet50", pretrained=True)
    model.output = nn.Linear(2048, 7)
    return model


def bam_resnet50(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("bam_resnet50", pretrained=True)
    model.output = nn.Linear(2048, 7)
    return model


def efficientnet_b7b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b7b", pretrained=True)
    model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(2560, 7))
    return model


def efficientnet_b3b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b3b", pretrained=True)
    model.output = nn.Sequential(nn.Dropout(p=0.3, inplace=False), nn.Linear(1536, 7))
    return model


def efficientnet_b2b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b2b", pretrained=True)
    model.output = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False), nn.Linear(1408, 7, bias=True)
    )
    return model


def efficientnet_b1b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b1b", pretrained=True)
    print(model)
    model.output = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False), nn.Linear(1280, 7, bias=True)
    )
    return model


def crossformer_l():

    model = CrossFormer(img_size=224,
                        patch_size=[4, 8, 16, 32],
                        in_chans=3,
                        num_classes=1000,
                        embed_dim=128,
                        depths=[ 2, 2, 18, 2 ],
                        num_heads= [ 4, 8, 16, 32 ],
                        group_size= [7,7,7,7],
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.0,
                        drop_path_rate=0.5,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False,
                        merge_size=[[2, 4], [2,4], [2, 4]]
                         )

    return model

def crossformer_b():

    model = CrossFormer(img_size=224,
                        patch_size=[4, 8, 16, 32],
                        in_chans=3,
                        num_classes=1000,
                        embed_dim=96,
                        depths=[2, 2, 18, 2],
                        num_heads=[3, 6, 12, 24],
                        group_size=[7, 7, 7, 7],
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.0,
                        drop_path_rate=0.3,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False,
                        merge_size=[[2, 4], [2, 4], [2, 4]]
                        )

    return model

class FaceNet(nn.Module):
    def __init__(self, backbone='hrnet_w64', pretrained=True):
        super(FaceNet, self).__init__()

        self.backbone = create_model(
            backbone,
            pretrained=pretrained
        )

        self.gaze_fc = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Linear(1000, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 7),
        )


    def forward(self, x):
        x = self.backbone(x)
        gaze = self.gaze_fc(x)

        return gaze


class BotNet(nn.Module):
    def __init__(self):
        super(BotNet, self).__init__()
        self.backbone = botnet()

        self.gaze_fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 7),
        )

    def forward(self, face):
        x = self.backbone(face)
        x = torch.flatten(x, 1)
        gaze = self.gaze_fc(x)

        return gaze


def get_face_model(name, **kwargs):

    if name == 'botnet':
        model = BotNet()
    else:
        model = FaceNet(backbone=name, **kwargs)

    return model



if __name__ == "__main__":

    # model = GazeNet()
    # model = get_model('botnet')
    model = get_face_model('hrnet_w64')
    # print(model)

    x = torch.randn(8, 3, 224, 224)
    outs = model(x)
    print(outs.shape)

