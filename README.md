# RETIA: Relation-Entity Twin-Interact Aggregation for Temporal Knowledge Graph Extrapolation

This is the released codes of the following paper:

Kangzheng Liu, Feng Zhao, Guandong Xu, Xianzhi Wang, and Hai Jin. RETIA: Relation-Entity Twin-Interact Aggregation for Temporal Knowledge Graph Extrapolation. ICDE 2023.

![RETIA](https://github.com/Liudaxian1/FIG/blob/main/RETIA.png)

## Citation

If this repository is helpful for your current work, please cite our paper here:

```shell
@inproceedings{DBLP:conf/icde/Liu0X0023,
  author       = {Kangzheng Liu and
                  Feng Zhao and
                  Guandong Xu and
                  Xianzhi Wang and
                  Hai Jin},
  title        = {{RETIA:} Relation-Entity Twin-Interact Aggregation for Temporal Knowledge
                  Graph Extrapolation},
  booktitle    = {39th {IEEE} International Conference on Data Engineering, {ICDE} 2023,
                  Anaheim, CA, USA, April 3-7, 2023},
  pages        = {1761--1774},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/ICDE55515.2023.00138},
  doi          = {10.1109/ICDE55515.2023.00138},
  timestamp    = {Thu, 27 Jul 2023 17:17:25 +0200},
  biburl       = {https://dblp.org/rec/conf/icde/Liu0X0023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Environment dependencies

```shell
python==3.6.5
torch==1.9.0+cu102
dgl-cu102==0.8.0.post1
tqdm==4.62.3
rdflib==5.0.0
numpy==1.19.5
pandas==1.1.5
```

## General Training

First, train the model based on the information of an invariant historical range (i.e., the size of the training set). The training parameters for different datasets are presented as follows:

### YAGO


```shell
cd src
python main.py -d YAGO --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7
```

### WIKI

```shell
cd src
python main.py -d WIKI --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7
```

### ICEWS14

```shell
cd src
python main.py -d ICEWS14 --train-history-len 9 --test-history-len 9 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 14 --discount 1 --add-static-graph
```

### ICEWS18

```shell
cd src
python main.py -d ICEWS18 --train-history-len 4 --test-history-len 4 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 10 --discount 1 --add-static-graph
```

### ICEWS05-15

```shell
cd src
python main.py -d ICEWS05-15 --train-history-len 9 --test-history-len 9 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 10 --discount 1 --add-static-graph
```

## Offline Testing

Directly evaluate the performance of the model obtained by General Training. The testing parameters for different datasets are presented as follows:

### YAGO

```shell
python main.py -d YAGO --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --test
```

### WIKI

```shell
python main.py -d WIKI --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --test
```

### ICEWS14

```shell
python main.py -d ICEWS14 --train-history-len 9 --test-history-len 9 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 14 --discount 1 --add-static-graph --test
```

### ICEWS18

```shell
python main.py -d ICEWS18 --train-history-len 4 --test-history-len 4 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 10 --discount 1 --add-static-graph --test
```

### ICEWS05-15

```shell
python main.py -d ICEWS05-15 --train-history-len 9 --test-history-len 9 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 10 --discount 1 --add-static-graph --test
```

## Online Continuous Training

Then, continuously train the model based on the newly emerging historical information at the validation or test set timestamps. The online continuous training parameters for different datasets are presented as follow:

### YAGO

Continuously train the model based on the newly emerging history in the validation set:

```shell
python main.py -d YAGO --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --test-valid
```

Continuously train and test the model based on the newly emerging history in the test set:

```shell
python main.py -d YAGO --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --test-test
```

### WIKI

Continuously train the model based on the newly emerging history in the validation set:

```shell
python main.py -d WIKI --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --test-valid
```

Continuously train and test the model based on the newly emerging history in the test set:

```shell
python main.py -d WIKI --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --test-test
```

### ICEWS14

Continuously train the model based on the newly emerging history in the validation set:

```shell
python main.py -d ICEWS14 --train-history-len 9 --test-history-len 9 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 14 --discount 1 --add-static-graph --test-valid
```

Continuously train and test the model based on the newly emerging history in the test set:

```shell
python main.py -d ICEWS14 --train-history-len 9 --test-history-len 9 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 14 --discount 1 --add-static-graph --test-test
```

### ICEWS18

Continuously train the model based on the newly emerging history in the validation set:

```shell
python main.py -d ICEWS18 --train-history-len 4 --test-history-len 4 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 10 --discount 1 --add-static-graph --test-valid
```

Continuously train and test the model based on the newly emerging history in the test set:

```shell
python main.py -d ICEWS18 --train-history-len 4 --test-history-len 4 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 10 --discount 1 --add-static-graph --test-test
```

### ICEWS05-15

Continuously train the model based on the newly emerging history in the validation set:

```shell
python main.py -d ICEWS05-15 --train-history-len 9 --test-history-len 9 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 10 --discount 1 --add-static-graph --test-valid
```

Continuously train and test the model based on the newly emerging history in the test set:

```shell
python main.py -d ICEWS05-15 --train-history-len 9 --test-history-len 9 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 10 --discount 1 --add-static-graph --test-test
```

## Reproduce the results in our paper

We provide the general training models for all datasets. The trained models can be downloaded at https://github.com/Liudaxian1/TrainedModels/tree/main/RETIAGeneral_Models. Then, put the trained models into the corresponding folders in the "./models" folder. Note that Github only allows large files to be downloaded one by one (bigger than 25MB). Therefore, go into the last level of each folder in the web to download, to ensure the integrity of the model files.

## Contacts

Contact us with the following email address: FrankLuis@hust.edu.cn.
