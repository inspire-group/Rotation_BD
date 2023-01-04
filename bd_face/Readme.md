





This is the file for running experiments on face recognition dataset


## Prepare for the experiment 

1. Download the [Data](https://github.com/inspire-group/Rotation_BD/releases/download/facedata/face.zip)
2. put into `./data` folder. 
3. unzip face.zip

Go to experiment folder to run our experiment.
```
cd Experiment 
```

Download the [VGGFace Pretrained Model](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz) (About 540M).
Unzip it by `tar -xvf vgg_face_torch.tar.gz` and put it in this folder.

`pip install torchfile` for loading model

## Training Backdoored Models
Multiple Class Attacks:

```
df=SS ## other defending methods can be find in defend.py 

backdoor_angle=90

rotation_augmentation=1 ##random rotation augmentation with [-15, +15] degree

inject_portion=0.01 
```

```
python bd_train.py --trigger_type ${backdoor_angle} --trans ${rotation_augmentation} --inject_portion ${inject_portion} --epochs 50
```

Command example:
```
python bd_train.py --trigger_type 90 --trans 1 --inject_portion 0.01  --epochs 50
```

## Defending methods 

```
python defend.py --defense ${df} --trigger_type ${backdoor_angle}  --pr ${inject_portion} 
```

Command example:
```
python defend.py --defense SS --trigger_type 90  --pr 0.01
```


