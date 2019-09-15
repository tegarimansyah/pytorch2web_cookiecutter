# {{cookiecutter.project_name}}

## Important Command

```bash
$ make install
$ jupyter notebook
$ make train
$ make serve
$ make tensorboard
```

## Folder Structure

```
{{cookiecutter.project_name}}
├── Makefile
├── README.md
├── app
│   ├── __init__.py
│   ├── architecture
│   │   ├── __init__.py        -> Data initiation (Datatrain, Datatest)
│   │   ├── NN.py              -> NN Architecture (LeNet)
│   │   └── train.py           -> Training Mechanism
│   ├── main.py                -> FastAPI Instance
│   └── predict                -> HTTP handler for prediction
│       ├── __init__.py
│       ├── service
│       │   ├── __init__.py
│       │   └── prediction.py   -> Logic for prediction
│       └── views.py            -> Routing for /predict
├── data                        -> Dataset Folder
│   └── README.md
└── requirements.txt
```
