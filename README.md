# GraphHop: An Enhanced Label PropagationMethod for Node Classification

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
@misc{xie2021graphhop,
      title={GraphHop: An Enhanced Label Propagation Method for Node Classification}, 
      author={Tian Xie and Bin Wang and C. -C. Jay Kuo},
      year={2021},
      eprint={2101.02326},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```