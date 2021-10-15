
# 人脸情绪识别挑战赛-第3名-W03KFgNOc-源代码、模型以及说明文档
1. 队名：W03KFgNOc
2. 排名：3
3. 正确率: 0.75564
4. 队员：[yyMoming](https://github.com/yyMoming),[xkwang](https://github.com/xk-wang),[RichardoMu](https://github.com/RichardoMrMu)。
5. 比赛链接：[人脸情绪识别挑战赛](http://challenge.xfyun.cn/topic/info?type=facial-emotion-recognition)
# emotion 
该项目分别训练八个模型并生成csv文件，并进行融合
## 构建conda环境
```shell
conda create -n emotion python==3.8.0
conda activate emotion
cd {project_path}
pip install -r requirements.txt
```
## 训练
打开`train.sh`，可以看到训练的命令行，依次注释和解注释随后运行`train.sh`。
因为是训练八个模型，分别是`efficientnet_b2b`, `efficientnet_b3b`, `cbam_resnet50`, `resmasking`,`resmasking_dropout1`,`resnest269e`,`swin`,`hrnet_w64`,所以要训练和测试，需要分别进行8次。

1. 训练efficientnet_b2b

```shell
python main_fer2013.py --config ./config/efficientnet_b2b_config.json
```
2. 训练efficientnet_b3b

```shell
python main_fer2013.py --config ./config/efficientnet_b3b_config.json
```
3. 训练cbam_resnet50

```shell
python main_fer2013.py --config ./config/cbam_resnet50_config.json
```

4. 训练hrnet_w64

```shell
python main_fer2013.py --config ./config/hrnet_w64_config.json
```
5. 训练resmasking

```shell
python main_fer2013.py --config ./config/resmasking_config.json
```
6. 训练resmasking_dropout1

```shell
python main_fer2013.py --config ./config/resmasking_dropout1_config.json
```
7. 训练resnest269e

```shell
python main_fer2013.py --config ./config/resnest269e_config.json
```

8. 训练swin

```shell
python main_fer2013.py --config ./config/swin_config.json
```
checkpoint保存在`{project_path}/checkpoint`目录下，可以在`log`文件夹下查看训练的日志。
## 预测
具体内容在`test.sh`文件中。各个模型我们存放在百度云盘 https://pan.baidu.com/s/1mM-APWoLV5P3nvrzmG--Jg  提取码 1gyh

下载后复制到user_data/model_data下面即可运行下面的命令进行预测。

1. 预测efficientnet_b2b

```shell
python gen_results.py --config ./config/efficientnet_b2b_config.json --model_name efficientnet_b2b --checkpoint_path efficientnet_b2b_2021Jul25_17.08
```
2. 预测efficientnet_b3b

```shell
python gen_results.py --config ./config/efficientnet_b3b_config.json --model_name efficientnet_b3b --checkpoint_path efficientnet_b3b_2021Jul25_20.08
```
3. 测试cbam_resnet50

```shell
python gen_results.py --config ./config/cbam_resnet50_config.json --model_name cbam_resnet50 --checkpoint_path cbam_resnet50_test_2021Jul24_19.18
```
4. 测试hrnet_w64

```shell
python gen_results.py --config ./config/hrnet_w64_config.json --model_name hrnet_w64 --checkpoint_path hrnet_test_2021Aug01_17.13
```
5. 测试resmasking

```shell
python gen_results.py --config ./config/resmasking_config.json --model_name resmasking --checkpoint_path resmasking_test_2021Jul26_14.33
```
6. 测试resmasking_dropout1

```shell
python gen_results.py --config ./config/resmasking_dropout1_config.json --model_name resmasking_dropout1 --checkpoint_path resmasking_dropout1_test_2021Aug01_17.13
```
7. 测试resnest269e

```shell
python gen_results.py --config ./config/resnest269e_config.json --model_name resnest269e --checkpoint_path resnest269e_test_2021Aug02_11.39
```

8. 测试swin

```shell
python gen_results.py --config ./config/swin_config.json --model_name swin_large_patch4_window7_224 --checkpoint_path swin_large_patch4_window7_224_test_2021Aug02_21.36
```
请注意，这里的`model_name`是确定的，`checkpoint_path`是你训练得到模型的名字，如果你自己训练了其中的一些模型，请将对应的名称修改为训练得到模型的名称。

## 集成

上述8个模型的预测结果统一放在user_data/tmp_data里面，下面使用集成方法对上述八个模型的结果进行整合。

```shell
python gen_ensemble.py
```
我们将上述八个模型的结果进行集成，最终生成的文件放在prediction_result下面的result.csv文件中。

