# Datitos


## TP2

### Notebooks

* [Notebook: Simple Model](https://github.com/adrianmarino/datitos/blob/master/tp2/tp2.ipynb)
* [Notebook: Model + Cross Validation + Hyper params optimizaiton](https://github.com/adrianmarino/datitos/blob/master/tp2/tp2-optimization.ipynb)

#### Parallel Training

You can run a training into N workers. Each worker can be seen as a trial executor job. Each job train a model with a set of specific hyper params. All hyperparams -score pairs are stored into a maridb db. Finally you can load optuna study to get best hyperparams with hiest score. You can run a worker as next:

```bash
$ conda activate datitos 
$ python bin/train.py --device gpu --study study1
```

To run 10 workers repeat previous command into 10 distinct shell sessions (bash/szh).

On the other hand, you can run workers that use CPU or GPU. Normally a good configuration could be 2 CPU + 8 GPU workers. 
This could be limited by the type of CPU, GPU and GPU and RAM memory. CPU workers parallelze k fold cross validation to decrese response time. GPU workers cant parallelize cv.

To monitor workers you can use any of next tools:

* [btop](https://github.com/aristocratos/btop)
* [htop](https://github.com/htop-dev/htop)
* [tmux](https://github.com/tmux/tmux)
