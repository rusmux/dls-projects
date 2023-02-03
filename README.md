# Проекты курса Deep Learning School

В данном репозитории находятся проекты, выполненные мною в ходе продвинутого потока школы
МФТИ [Deep Learning School](https://dls.samcs.ru).

[Диплом](certificate.pdf).

## Стек проектов

### Работа с данными

Numpy, Pandas, OpenCV, SciPy, scikit-image, PIL

### Машинное обучение

PyTorch, scikit-learn, CatBoost, LightGBM, Optuna

### Визуализация

Matplotlib, Seaborn, Plotly, IPyWidgets

## Проекты

- [CNNs](notebooks/CNNs.ipynb) – Базовая работа с CNN. Реализация логистической регрессии и архитектуры LeNet.
  Демонстрация
  работы сверточных ядер.

- [Churn Prediction](notebooks/Churn-Prediction-2.ipynb) – Предсказание оттока клиентов мобильного оператора в виде
  соревнования на платформе Kaggle. В качестве метрики качества использовался ROC-AUC. Было проведено сравнение
  различных моделей: логистическая регрессия, дерево решений, ансамбль деревьев, градиентный бустинг. Были
  протестированы библиотеки scikit-learn, LightGBM, CatBoost. Для подбора гиперпараметров использовался Random
  Search и байесовские методы библиотеки Optuna.

- [Simpsons Classification](notebooks/Simpsons-Classification.ipynb) – Классификация персонажей мультсериала "Симпсоны"
  в виде соревнования на платформе Kaggle. В качестве метрики качества использовался F1-score. Для решения задачи
  использовался fine-tuning модели ResNet-50, предобученной на датасете ImageNet1k.

- [Lesion Segmentation](notebooks/Lesion-Segmentation.ipynb) – Сегментация медицинских снимков. Были реализованы
  архитектуры моделей SegNet и U-Net и было проведено их обучение с различными функциями потерь: BCE Loss, Focal Loss,
  DICE Loss.

- [Autoencoders](notebooks/Autoencoders.ipynb) – Реализация архитектур обычного, вариационного и условного
  автоэнкодера для
  генерации новых картинок из латентного пространства.

- [GANs](notebooks/GANs.ipynb) – Обучение GAN-моделей для генерации лиц людей. Для обучения использовалась часть
  датасета Flickr Faces. Для оценки качества модели использовалась Leave-one-out кросс-валидация.

- [Face Recognition](notebooks/Face-Recognition.ipynb) – Распознавания лиц. Для решения использовалась модель
  Inception-ResNet, которая была обучена с функциями потерь Cross Entropy, Triplet Loss и ArcFace. Для обучения был
  написан свой собственный Trainer. Для оценки качества модели использовался Identification Rate. Также был реализован поиск других изображений одного человека с
  помощью Cosine Similarity, поиск похожих лиц, и нахождение некачественных фотографий.
