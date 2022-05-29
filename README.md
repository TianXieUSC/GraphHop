# GraphHop: An Enhanced Label Propagation Method for Node Classification

This respository contains the PyTorch implementation of GraphHop for the task of semi-supervised classification of nodes in a graph, as described in our paper:

Tian Xie, Bin Wang, C.-C. Jay Kuo, GraphHop: An Enhanced Label PropagationMethod for Node Classification. [[paper]](https://arxiv.org/abs/2101.02326)


## Dependencies
* torch == 1.5.0
* numpy == 1.18.1
* scipy == 1.4.1
* [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## RUN
```
sh run_model.sh
```
You may change the hyperparameters inside the shell script.

## Citation
If you are use this code for your research, please cite our paper.

```
@ARTICLE{9737682,
  author={Xie, Tian and Wang, Bin and Kuo, C.-C. Jay},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={GraphHop: An Enhanced Label Propagation Method for Node Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2022.3157746}}
```
