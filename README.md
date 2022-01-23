# Datitos

[Datitos: Aprendizaje profundo](https://datitos.github.io/curso-aprendizaje-profundo/)

## Requeriments

* **miniconda** / **anaconda**
* **MariaDB** / **MySQL**
* **Apache Airflow**: See [airflow-systemd](https://github.com/adrianmarino/airflow-systemd) to install Apache Airflow as systemd daemon.
* **Optuna Dashboard**: See [optuna-dashboard-systemd](https://github.com/adrianmarino/optuna-dashboard-systemd) to install Optuna Dashbord as systemd daemon.

## TP2

[T.P. N°2 - Aprendizaje Profundo 2021 by Datitos](https://www.kaggle.com/c/tp-n2-aprendizaje-profundo-2021-by-datitos-v2)

### Notebooks

* [MLP Model](https://github.com/adrianmarino/datitos/blob/master/tp2/tp2-1-mlp.ipynb)
* [MLP Model + Cross Validation + Hyper params optimizaiton](https://github.com/adrianmarino/datitos/blob/master/tp2/tp2-2-mlp-cv-hp-opt.ipynb)
* [LightGBM Model](https://github.com/adrianmarino/datitos/blob/master/tp2/tp2-3-lightgbm.ipynb)

### Best results

* **Public Score**: **0.90581**
* **Private Score**: 0.89456
* **File**: study16-predict-2022-01-23_07-44-35.csv


### Training Workflow

To automate complete train process(training, reports generation, kaggle file) already exist an [Airflow](https://airflow.apache.org/) [Dag](https://github.com/adrianmarino/datitos/blob/master/dags/fifa_dag.py). An airflow dag is a data workflow that runs N parallel training processes and then run reports generation and kaggle result generation steps.

![image](images/dag.png)


### Parallel Training

You can run a training into N workers. Each worker can be seen as a trial executor job. Each job train a model with a set of specific hyper params. All hyperparams -score pairs are stored into a maridb db. Finally you can load optuna study to get best hyperparams with hiest score. You can run a worker as next:

**Notes**

* Each trainig process is a `train_model.py` execution.
* Optimization report step run `bin/optmimization_report.py` script.
* Test model step run `bin/test_model.py` script.
* See below to undestant how do each script. 


**GPU**

```bash
$ conda activate datitos 
$ python bin/train_model.py --device gpu \
                            --study study3 \
                            --cuda-process-memory-fraction 0.1 \
                            --folds 5 \
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
                      --folds 5 \
                      --trials 300 \
                      --db-url mysql://root:1234@localhost/example \
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


### Optimization report

Generate next plots for specified optuna study:


#### Validation accuracy distribution

![image](report/study16-acc_dist.png)

#### Optimizartion Contour diagram

![image](report/study16-contour.png)

#### Optimizartion EDF

![image](report/study16-edf.png)

#### Optimizartion history

![image](report/study16-optimization_history.png)

#### Optimizartion paralel coordinates

![image](report/study16-parallel_coordinate.png)

#### Feature importance

![image](report/study16-param_importances.png)

#### Slice

![image](report/study16-slice.png)

#### Optimization trials accurary distribution 

![image](report/study16-trials_metric_dist.png)


#### Command


```bash
$ conda activate datitos
$ python bin/optmimization_report.py \
    --study study6 \
    --db-url mysql://root:1234@localhost/example \
    --device gpu \
    --seeds-count 3 \
    --folds 2
```

See script help:

```bash
$ python  bin/optmimization_report.py --help

Usage: optmimization_report.py [OPTIONS]

Options:
  --device TEXT          Device used to train and optimize model. Values: gpu,
                         cpu.
  --study TEXT           The study name.
  --db-url TEXT          Mariadb/MySQL connection url.
  --report-path TEXT     Path where save optimization plots.
  --seeds-count INTEGER  seeds count used calculate acuracy distribution
  --folds INTEGER        Number of train dataset splits to apply cross
                         validation.
  --help                 Show this message and exit.
  ```

### Test model

It script runs N model training instances using hyperparameters of optimization trial with best accurary. Then gets model with highest accuracy and predict under kaggle test file. Finally genera kaggle file to upload. 

```bash
$ conda activate datitos
$ python  bin/test_model.py \
    --study study6 \
    --db-url mysql://root:1234@localhost/example \
    --device gpu
```

See script help:

```bash
$ python bin/test_model.py --help

Usage: test_model.py [OPTIONS]

Options:
  --device TEXT       Device used to train and optimize model. Values: gpu,
                      cpu.
  --study TEXT        The study name.
  --db-url TEXT       Mariadb/MySQL connection url.
  --result-path TEXT  path where test predictions are saved.
  --help              Show this message and exit.
```
