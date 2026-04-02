## Задача 2

- Тип задачи: Сегментация
- Датасет: Massachusetts Roads Dataset
- Метрика качества: Dice

Краткое описание задачи
Задача заключается в разработке модели машинного обучения для семантической сегментации дорожной сети на основе спутниковых снимков из набора данных Massachusetts Roads Dataset. Целью является автоматическое выделение пикселей, принадлежащих дорогам, на аэрофотоснимках с максимизацией метрики Dice Coefficient.

## Стркутура директории

24943_ALPATOVA_II_LAB2/

├── data/

│   └── /massachusetts-roads-dataset/tiff/

├── scripts/

│   ├── train.py

│   ├── test.py

│   └── val.py

├── src/

│   ├── data.py

│   └── model.py

├── requirements.txt

├── loss_graph.png

├── best_model.pth

└── last_model.pth



## Подготовка датасета

- Скачать zip-архив по ссылке: https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset
- Расархировать в папку data.

## Как воспроизвести

Команда обучения:
```
python scripts/train.py
```
Команда валидации:
```
python scripts/val.py
```
Команда тестирования:
```
python scripts/test.py
```
## Development

- Создание и активация виртуального окружения:

```bash
python -m venv .venv
source .venv/bin/activate
```

- Зависимости
```bash
pip install -r requirements.txt
```

## Результаты обучения

| Epoch | Train IoU | Val IoU | Train Dice | Val Dice | Train Loss |
|:-----:|:---------:|:-------:|:----------:|:--------:|:----------:|
| 1/10  |  12.90%   | 26.82%  |   22.29%   |  42.22%  |   0.5922   |
| 2/10  |  24.97%   | 40.72%  |   38.92%   |  57.83%  |   0.4434   |
| 3/10  |  35.41%   | 48.65%  |   51.22%   |  65.38%  |   0.3631   |
| 4/10  |  41.25%   | 51.85%  |   57.44%   |  68.21%  |   0.3280   |
| 5/10  |  44.24%   | 53.53%  |   60.43%   |  69.63%  |   0.3106   |
| 6/10  |  45.53%   | 53.56%  |   61.57%   |  69.65%  |   0.3043   |
| 7/10  |  46.96%   | 54.40%  |   63.06%   |  70.37%  |   0.2954   |
| 8/10  |  47.59%   | 55.92%  |   63.58%   |  71.66%  |   0.2918   |
| 9/10  |  48.25%   | 55.17%  |   64.15%   |  70.99%  |   0.2883   |
| 10/10 |  48.96%   | 55.44%  |   64.79%   |  71.21%  |   0.2837   |
|

> **Итоги:**
> - **Лучший Dice на валидации:** 71.66%


### Test Results

`IoU: 56.95%` • `Dice: 72.33%`


<img width="1121" height="585" alt="image" src="https://github.com/user-attachments/assets/fef72b2b-73ad-4f3f-a29a-61762c9acd10" />




