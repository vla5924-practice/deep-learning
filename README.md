# Глубокое обучение

Лабораторные работы


## Настройка окружения

Создание [виртуального окружения](https://docs.python.org/3/library/venv.html):

```sh
python3 -m venv .venv --prompt deep-learning
source .venv/bin/activate   # Linux bash/zsh
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

Установка зависимостей должна быть произведена один раз:

```sh
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
```


## Загрузка данных

Используемые датасеты:

* [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
* [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
