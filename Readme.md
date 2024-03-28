# How Does Distribution Matching Help Domain Generalization: An Information-theoretic Analysis

Our supplementary material enables the replication of two experiments:
* Colored MNIST
* DomainBed

## Colored MNIST

Below are the steps to reproduce the results in Table 1:

```sh
cd colored_mnist
python train_coloredmnist.py --algorithm idm
```

The reported results of ERM, IRM, V-REx and Fishr are from [Fishr repository](https://github.com/alexrame/fishr).

The final hyper-parameters selected for IDM and IGA:

| Parameter | Distribution | IDM | IGA |
| :-------- | :------------------ | :-: | :-: |
| hidden dimension | $2^{\mathrm{Uniform}(6,9)}$ | 433 | 138 |
| weight decay | $10^{\mathrm{Uniform}(-2,-5)}$ | 0.00034 | 0.001555 |
| learning rate | $10^{\mathrm{Uniform}(-2.5,-3.5)}$ | 0.000449 | 0.001837 |
| warmup iterations | $\mathrm{Uniform}(50,250)$ | 154 | 118 |
| regularization strength | $10^{\mathrm{Uniform}(4,8)}$ | 2888595.180638 | 17320494.495665 |

## DomainBed

We implement IDM in [algorithms.py](./DomainBed/domainbed/algorithms.py) and set the hyper-parameters in [hparams_registry.py](./DomainBed/domainbed/hparams_registry.py).

Below are the steps to reproduce the results in Table 2:

```sh
cd DomainBed
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/data/dir/\
       --output_dir=/my/sweep/output/path\
       --command_launcher multi_gpu
       --datasets ColoredMNIST\
       --algorithms IDM
```

Please refer to [DomainBed repository](https://github.com/facebookresearch/DomainBed) for how to setup the DomainBed environment and download the datasets.
