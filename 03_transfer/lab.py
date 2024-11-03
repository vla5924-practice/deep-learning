# %% [markdown]
# # Глубоко обучение. Лабораторная работа №3
# 
# **Цель:** изучить и реализовать метод переноса знаний (transfer learning) в глубоких нейронных сетях для решения задачи классификации изображений. Предлагается применение метода переноса знаний к известным моделям глубокого обучения, обученных на наборе данных ImageNet. Существуют различные подходы к переносу знаний:
# 
# - Обучение всех весов нейронной сети
# - Обучение весов слоев, которые были добавлены или изменены
# - Обучение нескольких последних слоев
# - Обучение части слоев нейронной сети
# 
# **Требования:**
# 
# 1. Необходимо выполнить следующие задачи:
#     - Загрузить и проверить данные, включая демонстрацию избранных изображений и меток классов для подтверждения корректности загрузки и совпадения размерностей.
#     - Загрузить 4 нейронные сети (можно использовать torchvision), обученные на наборе данных ImageNet. Требуется для каждой модели провести 2 эксперимента, используя разные конфигурация переноса знаний. Модифицировать последний слой и реализовать обучение на наборе данных Garbage Classification. Настроить гиперпараметры обучения.
#     - Построить F1-score от количества эпох для всех моделей на валидационных данных. Построить сравнительную столбчатую диаграмму точностей: модель и тип эксперимента (с кратким указанием параметров) по горизонтали, F1 score на тестовых данных по вертикали.
# 
# 2. Проверка корректности:
#     - Разделите датасет на тренировочную, валидационную и тестовую выборки самостоятельно в соотношении 70/15/15.
#     - Для оценки качества следует использовать Macro [F1-score](https://en.wikipedia.org/wiki/F-score), поскольку датасет не сбалансирован.
# 
# 3. Можно использовать любые сверточные архитектуры или архитектуры на базе механизма внимания (transformer, ViT).

# %%
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from time import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models

# %%
# Гиперпараметры
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# %% [markdown]
# ## Подготовка данных
# 
# В качестве входных данных должен быть выбран датасет Garbage Classification с сайта https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

# %%
data_dir = '/work/garbage'
classes = os.listdir(data_dir)
print(classes)

# %% [markdown]
# ### Загрузка и нормализация данных
# 
# Для создания загрузчика были использованы обучающие материалы с сайта https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html, адаптированные для текущей задачи.

# %%
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
print("Dataset size:", len(dataset))

# %%
def show_sample(img, label):
    print("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))

img, label = dataset[999]
show_sample(img, label)

# %% [markdown]
# ### Разделение выборки
# 
# Входные данные разделены на три независмые выборки с одинаковым распределением по классам:
# 
# * Тренировочная выборка (70%) для обучения модели
# * Валидационная выборка (15%) для выбора лучшей модели
# * Тестовая выборка (15%) для демонстрации результата обучения 

# %%
targets = dataset.targets
indices = [*range(len(dataset))]

train_indices, temp_indices = train_test_split(indices, test_size=0.3, stratify=targets)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, stratify=[targets[i] for i in temp_indices])

train_dataset = Subset(dataset, train_indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = Subset(dataset, val_indices)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))

# %% [markdown]
# ## Настройка нейросетей
# 
# Для проведения экспериментов были выбраны четыре архитектуры сверточных нейросетей:
# 
# * ResNet-50 https://arxiv.org/pdf/1512.03385
# * MobileNetV2 https://arxiv.org/pdf/1801.04381
# * EfficientNet-B0 https://arxiv.org/pdf/1905.11946
# * Densenet-121 https://arxiv.org/pdf/1608.06993

# %% [markdown]
# ### ResNet-50
# 
# Предлагается провести два эксперимента с разными моделями на базе указанной архитектуры:
# 
# 1. Обучение всех весов.
# 2. Добавление слоев `Dropout + Linear + ReLU` перед последним линейным слоем и обучение весов только добавленных слоев.

# %%
def resnet50_classic(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet50_modified(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
    )
    return model

# %% [markdown]
# ### MobileNetV2
# 
# Предлагается провести два эксперимента с разными моделями на базе указанной архитектуры:
# 
# 1. Обучение всех весов.
# 2. Добавление блока, аналогичного блокам _Bottleneck_ в архитектуре сети, перед последним линейным слоем и обучение весов только добавленных слоев.

# %%
def mobilenet_v2_classic(num_classes: int):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def mobilenet_v2_modified(num_classes: int):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    conv_block = nn.Sequential(
        nn.Conv2d(in_channels=1280, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
    )
    old_feature = model.features[-1]
    model.features[-1] = nn.Sequential(old_feature, conv_block)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model

# %% [markdown]
# ### EfficientNet-B0
# 
# Предлагается провести два эксперимента с разными моделями на базе указанной архитектуры:
# 
# 1. Обучение всех весов.
# 2. Добавление блока `Dropout + Linear + ReLU + BatchNorm` (для снижения риска переобучения на небольших датасетах) перед последним линейным слоем и обучение весов только добавленных слоев.

# %%
def efficientnet_b0_classic(num_classes: int):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def efficientnet_b0_modified(num_classes: int):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),
    )
    return model

# %% [markdown]
# ### Densenet-121
# 
# Предлагается провести два эксперимента с разными моделями на базе указанной архитектуры:
# 
# 1. Обучение всех весов.
# 2. Обучение всех весов, кроме весов первых двух блоков _Dense Block_ в архитектуре сети.

# %%
def densenet121_classic(num_classes: int):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def densenet121_modified(num_classes: int):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    for i, param in enumerate(model.parameters()):
       if i < 6 + 12:
           param.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# %% [markdown]
# ## Обучение моделей
# 
# В качестве функции потерь будет использована кросс-энтропия (`torch.nn.CrossEntropyLoss`). Обновление параметров будет происходить с помощью метода Adam (`torch.optim.Adam`). Выбор лучшей модели будет осуществляться на основании максимального значения Macro F1 score (`sklearn.metrics.f1_score`), полученного на валидационной выборке. 

# %%
def model_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def eval_f1(model, dataloader: DataLoader):
    device = model_device(model)
    f1_all = []
    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        f1 = f1_score(labels.cpu(), preds.cpu(), average="macro")
        f1_all.append(f1)
    return sum(f1_all) / len(f1_all)


def train_and_evaluate(model, name, train_loader, val_loader, num_epochs, learning_rate):
    device = model_device(model)
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0
    val_f1_stat = []
    for epoch in range(num_epochs):
        model.train()
        start_time = time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_time = time() - start_time
        start_time = time()
        val_f1 = eval_f1(model, val_loader)
        val_time = time() - start_time
        if val_f1 > best_f1:
            torch.save(model.state_dict(), f"best_{name}.pth")
            best_f1 = val_f1
        val_f1_stat.append(val_f1)
        print(f"Epoch {epoch}: Train time {train_time:.2f} s, Validation time {val_time:.2f} s, Validation F1 {val_f1:.2f}", flush=True)
    return val_f1_stat

# %%
test_f1_stat = {}

assert torch.cuda.is_available(), "CUDA backend must be available"
device = torch.device("cuda")
print(device)

# %%
def run_experiment(model_factory):
    global test_f1_stat
    model = model_factory(len(classes)).to(device)
    name = model_factory.__name__
    val_f1_stat = train_and_evaluate(model, name, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)
    model.load_state_dict(torch.load(f"best_{name}.pth", weights_only=True))
    test_f1 = eval_f1(model, test_loader)
    print("Test F1:", test_f1)
    plt.plot(range(NUM_EPOCHS), val_f1_stat, marker="o")
    plt.title(f"{name} validation")
    plt.xlabel("epoch")
    plt.xticks([i for i in range(NUM_EPOCHS)])
    plt.ylabel("F1 score")
    plt.grid()
    plt.show()
    test_f1_stat[name] = test_f1

# %% [markdown]
# Проведены эксперименты, описанные в предыдущем разделе. Значение Macro F1 score в процессе обучения представлено на соответствующих графиках.
# 
# ### ResNet-50

# %%
run_experiment(resnet50_classic)

# %%
run_experiment(resnet50_modified)

# %% [markdown]
# ### MobileNetV2

# %%
run_experiment(mobilenet_v2_classic)

# %%
run_experiment(mobilenet_v2_modified)

# %% [markdown]
# ### EfficientNet-B0

# %%
run_experiment(efficientnet_b0_classic)

# %%
run_experiment(efficientnet_b0_modified)

# %% [markdown]
# ### Densenet-121

# %%
run_experiment(densenet121_classic)

# %%
run_experiment(densenet121_modified)

# %% [markdown]
# ## Вывод результатов
# 
# На диаграмме ниже представлены значения Macro F1 score на тестовой выборке для моделей, выбранных в каждом из экспериментов.

# %%
model_names = [*test_f1_stat.keys()]
f1_scores = [*test_f1_stat.values()]
plt.bar(model_names, f1_scores, color="blue")
plt.xlabel("experiment")
plt.ylabel("F1 score")
plt.title("Test F1")
plt.xticks(rotation=45, ha="right")
plt.show()
