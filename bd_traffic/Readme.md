
This is the file for running experiments on traffic sign dataset


## Prepare for the experiment 

Download the [Data](https://github.com/tongwu2020/Rotation_BD/releases/download/dataset/bdt_data.zip) and put into this folder.


## Training Backdoored Models

Multiple Class Attacks:

```
df=NC ## other defending methods can be find in defend.py 

backdoor_angle=90

rotation_augmentation=15 ##random rotation augmentation with [-15, +15] degree

inject_portion=0.01 
```

```
python poison_train.py -aug ${rotation_augmentation} -rb_angle ${backdoor_angle} -pr ${inject_portion}
```

## Defending methods 

```
python defend.py -aug ${rotation_augmentation} -rb_angle ${backdoor_angle} -defense ${df} -pr ${inject_portion}
```


## Evaluate physical noise  


```
python physical_test_noise.py -rb_angle ${backdoor_angle} -pr ${inject_portion} -noise "gn"
```

Other noise can be found in the `./physical_test_noise.py`
