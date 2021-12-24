# Datitos


## TP2

### Notebooks

* [Notebook: Model version 1](https://github.com/adrianmarino/datitos/blob/master/tp2/tp2.ipynb)
* [Notebook: Model + optimizaiton](https://github.com/adrianmarino/datitos/blob/master/tp2/tp2-optimization.ipynb)

#### Parallel Training

You can run a training into N workers. Each worker can be seen as a trial executor job. Each job train a model with a set of specific hyper params. All hyperparams -score pairs are stored into a maridb db. Finally you can load optuna study to get best hyperparams with hiest score. You can run a worker as next:

```bash
$ conda activate datitos 
$ python bin/train.py --device gpu --studt study1
```

To run 10 worker repeat previous command into 10 distinct shells (bash/szh).

On the other hand, you cna run worker that use cpu or gpu. Normally a good configuration could be 2 cpu + 8 gpu workers. 
This could be limited by the type of cpu, gpu and gpiu and ram memory.
