# cd {workspace}/

# train efficientnet_b2b
python main_fer2013.py --config ./config/efficientnet_b2b_config.json

# train efficientnet_b3b
# python main_fer2013.py --config ./config/efficientnet_b3b_config.json

# train cbam_resnet50
# python main_fer2013.py --config ./config/cbam_resnet50_config.json

# train hrnet_w64
# python main_fer2013.py --config ./config/hrnet_w64_config.json

# train resmasking
# python main_fer2013.py --config ./config/resmasking_config.json

# train resmasking_dropout1
# python main_fer2013.py --config ./config/resmasking_dropout1_config.json

# train resnest269e
# python main_fer2013.py --config ./config/resnest269e_config.json

# train swin
# python main_fer2013.py --config ./config/swin_config.json