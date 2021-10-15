# cd {workspace}/

# test efficientnet_b2b get xxx.npy file
python gen_results.py --config ./config/efficientnet_b2b_config.json --model_name efficientnet_b2b --checkpoint_path efficientnet_b2b_2021Jul25_17.08

# test efficientnet_b3b get xxx.npy file
# python gen_results.py --config ./config/efficientnet_b3b_config.json --model_name efficientnet_b3b --checkpoint_path efficientnet_b3b_2021Jul25_20.08

# test cbam_resnet50 get xxx.npy file
# python gen_results.py --config ./config/cbam_resnet50_config.json --model_name cbam_resnet50 --checkpoint_path cbam_resnet50_test_2021Jul24_19.18

# test hrnet_w64 get xxx.npy file
# python gen_results.py --config ./config/hrnet_w64_config.json --model_name hrnet_w64 --checkpoint_path hrnet_test_2021Aug01_17.13

# test resmasking
# python gen_results.py --config ./config/resmasking_config.json --model_name resmasking --checkpoint_path resmasking_test_2021Jul26_14.33

# test resmasking_dropout1
# python gen_results.py --config ./config/resmasking_dropout1_config.json --model_name resmasking_dropout1 --checkpoint_path resmasking_dropout1_test_2021Aug01_17.13

# test resnest269e
# python gen_results.py --config ./config/swin_config.json --model_name swin_large_patch4_window7_224 --checkpoint_path swin_large_patch4_window7_224_test_2021Aug02_21.36

# test swin
# python gen_results.py --config ./config/swin_config.json --model_name swin_large_patch4_window7_224 --checkpoint_path swin_large_patch4_window7_224_test_2021Aug02_21.36