# RETIA: Relation-Entity Twin-Interact Aggregating for Temporal Knowledge Graph Extrapolating

This is the released codes of the following paper:

Kangzheng Liu, Feng Zhao, Guandong Xu, and Hai Jin. RETIA: Relation-Entity Twin-Interact Aggregating for Temporal Knowledge Graph Extrapolating.

![RETIA](https://github.com/Liudaxian1/FIG/blob/main/RETIA.png)

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

First, train the model based on the information of an invariant historical range (e.i., the size of the training set). The training parameters for different datasets are presented as follows:

### YAGO


```shell
python main.py -d YAGO --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7
```

### WIKI

```shell
python main.py -d WIKI --train-history-len 3 --test-history-len 3 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7
```

### ICEWS14

```shell
python main.py -d ICEWS14 --train-history-len 9 --test-history-len 9 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 14 --discount 1 --add-static-graph
```

### ICEWS18

```shell
python main.py -d ICEWS18 --train-history-len 4 --test-history-len 4 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --gpu 0 --ft_lr=0.001 --norm_weight 1 --task-weight 0.7 --weight 0.5 --angle 10 --discount 1 --add-static-graph
```

### ICEWS05-15

```shell
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

We provide general training models for all datasets.

## Contacts

Contact us with the following email address: FrankLuis@hust.edu.cn.
