# Datitos


## TP2

### Notebooks

* [Notebook: Simple Model](https://github.com/adrianmarino/datitos/blob/master/tp2/tp2.ipynb)
* [Notebook: Model + Cross Validation + Hyper params optimizaiton](https://github.com/adrianmarino/datitos/blob/master/tp2/tp2-optimization.ipynb)

#### Parallel Training

You can run a training into N workers. Each worker can be seen as a trial executor job. Each job train a model with a set of specific hyper params. All hyperparams -score pairs are stored into a maridb db. Finally you can load optuna study to get best hyperparams with hiest score. You can run a worker as next:

**GPU**

```bash
$ conda activate datitos 
$ python bin/train.py --device gpu \
                      --study study3 \
                      --cuda-process-memory-fraction 0.1 \
                      --folds 10 \
                      --trials 300 \
                      --db-url mysql://root:1234@localhost/example \
                      --timeout 5000
```

To run 10 workers repeat previous command into 10 distinct shell sessions (bash/szh).

On the other hand, you can run workers that use CPU or GPU. Normally a good configuration could be N GPU workers and maybe 1 CPU worker, because CPU workers are high CPU consuming processes. 
This could be limited by the type of CPU, GPU and GPU and RAM memory. CPU workers parallelze k fold cross validation to decrese response time. GPU workers cant parallelize cv.

**CPU**

```bash
$ conda activate datitos 
$ python bin/train.py --device cpu \
                      --study study3 \
                      --folds 10 \
                      --trials 300 \
                      --db-url mysql://root:lv3jg6@localhost/example \
                      --timeout 5000
```

To monitor workers you can use any of next tools:

* [btop](https://github.com/aristocratos/btop)
* [htop](https://github.com/htop-dev/htop)
* [tmux](https://github.com/tmux/tmux)

See script help:

```bash
$ python bin/train.py --help

Usage: train.py [OPTIONS]

Options:
  --device TEXT                   Device used to train and optimize model.
                                  Values: gpu, cpu.
  --study TEXT                    The study name.
  --trials INTEGER                Max trials count.
  --timeout INTEGER               maximum time spent optimizing hyper
                                  parameters in seconds.
  --db-url TEXT                   Mariadb/MySQL connection url.
  --cuda-process-memory-fraction FLOAT
                                  Setup max memory user per CUDA procees.
                                  Percentage expressed between 0 and 1
  --folds INTEGER                 Number of train dataset splits to apply
                                  cross validation.
  --help                          Show this message and exit.
```
