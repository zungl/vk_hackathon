## MindStorm
#### VK - Машинное обучение на графах (ОФЛАЙН-КЕЙС)

<!-- ![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here) -->
[![presentation](https://img.shields.io/badge/Microsoft_PowerPoint-B7472A?style=for-the-badge&logo=microsoft-powerpoint&logoColor=white&label=%D0%9F%D1%80%D0%B5%D0%B7%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8F)](https://docs.yandex.ru/docs/view?url=ya-disk-public%3A%2F%2F%2BdRQmyK%2FfIWnt3qkyREc5P%2FBIdIsBo3BhN6GRqv69roJxRp%2B0KFrFKbQc0iOS4yaq%2FJ6bpmRyOJonT3VoXnDag%3D%3D&name=MindStorm%20-%20VK%20hackathon.pptx)
[![video](https://img.shields.io/badge/ScreenCast-black?style=for-the-badge&logo=Files&logoColor=white)](https://disk.yandex.ru/i/Abx8CVluuczQCQ)
<hr>

## Директории
`/data` - Директория для файлов с данными<br>
`/utils` - Кастомный модуль с утилитами<br>
С функциями для поиска оптимальных гиперпараметров Feature_selection, Featrue_importance и обучения модели

## Ноутбуки
`get_data.ipynb` - Получение исходных данных<br>
`preproc&eda.ipynb` - Пердобработка и анализ исходнгых данных<br>
`get_session_embeddings.ipynb` - Эмбэддинги для пользовательских сессий<br>
`combined model.ipynb` - Модель использующая все данные<br>
`logreg.ipynb` - <br>
## Краткое описание решения:
Для решения задачи классификации пользователей социальной сети ВКонтакте был использован подход построения моделей ансамблевого прогнозирования. Результирующая модель является линейной комбинацей моделей более низкого уровня каждая из которых, построена на специальном наборе данных:

• Выявление пользователей склонных к благотоворительности за счёт вектора собственных признаков на основе моделей бустинга<br>
• за счёт окружения пользователя (его друзей), посредством аггрегации их признаков<br>
• модели генерации векторов состояния пользователя. Для этого был построен граф переходов между сессиями, по которому впоследствии были получены векторные предстваления по каждой из сессии. Затем последовательности сессий пользователя были определны как вектор, характеризующий пользователя.<br>
## Уникальность:

Построенная модель учитывает большое количество факторов, который были добыты с помощью применения продвинутых технологий обработки графов.

## 🛠 Технологии

[![python](https://img.shields.io/pypi/pyversions/Torch?color=green&label=python&style=for-the-badge)](https://python.org/)
[![jupyter](https://img.shields.io/badge/Jupyter-black?style=for-the-badge&logo=Jupyter)]()
[![lightGBM](https://img.shields.io/badge/lightGBM-black?style=for-the-badge&logo=lightGBM)]()
[![KGembeddings](https://img.shields.io/badge/KGembeddings-black?style=for-the-badge&logo=KGembeddings)]()
[![PyKeen](https://img.shields.io/badge/PyKeen-black?style=for-the-badge&logo=PyKeen)]()
[![Pytorch](https://img.shields.io/badge/Pytorch-black?style=for-the-badge&logo=Pytorch)]()
[![Distmult](https://img.shields.io/badge/Distmult-black?style=for-the-badge&logo=Distmult)]()
[![NodeToVec](https://img.shields.io/badge/NodeToVec-black?style=for-the-badge&logo=NodeToVec)]()
[![GGVec](https://img.shields.io/badge/GGVec-black?style=for-the-badge&logo=GGVec)]()
[![NetworkX](https://img.shields.io/badge/NetworkX-black?style=for-the-badge&logo=NetworkX)]()
[![NodeToVec](https://img.shields.io/badge/NodeToVec-black?style=for-the-badge&logo=NodeToVec)]()
[![Feature_selection](https://img.shields.io/badge/Feature_selection-black?style=for-the-badge&logo=Feature_selection)]()
[![Featrue_importance](https://img.shields.io/badge/Featrue_importance-black?style=for-the-badge&logo=Featrue_importance)]()
[![Blending](https://img.shields.io/badge/Blending-black?style=for-the-badge&logo=Blending)]()
