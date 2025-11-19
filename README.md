В ходе работы были выполнены следующие задачи:
* Получены mel-spectrograms для набора голосовых команд.
* Обучены нейронные сети с бинарными весами.
  * Реализован метод из статьи [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/pdf/1603.05279).
  * Проведено сравнение наивной реализации и кода из репозиториев https://github.com/allenai/XNOR-Net, https://github.com/jiecaoyu/XNOR-Net-PyTorch
  * Выполнен рефакторинг: разрозненная логика заменена кастомным оператором с реализованными forward и backward, а также созданными на его основе слоями BinLinear и BinConv2d для PyTorch.
  * Исследована стабильность обучения для: наивной реализации из статьи, реализации из официального репозитория, реализованного кастомного оператора, варианта с квантизацией (bit=2) из семинара.
* Разработан PyTorch C++ extension для упаковки бинарных матриц в формат uint64_t и выполнения бинарного матричного умножения на основе операций XNOR и popcount.
* Проведено сравнение производительности:
  * C++ extension c умножением бинарных матриц
  * C++ extension с наивной реализацией матричного умножения
  * операций матричного умножения в PyTorch (float32, int32, int8).

Описание структуры репозитория

/BinaryNet
* **BinaryNet/binary_layers** - кастомные бинарные слои (линейный и свертка)
* **BinaryNet/models** - модели с бинарными весами
* **BinaryNet/utils** - датасет для конвертации ауди файлов в mel spectrograms, ClassificationTrainer

/experiments
* **dataset_and_fft_features.ipynb** - mel spectrograms
* **test_models_on_filtered_dataset.ipynb** - тестирование простой сверточной сетки без бинаризации
* **test_binary_models_on_filtered_dataset.ipynb** - эксперименты с обучением моделей с бинарными весами (с использованием различных подходов) на датасете из 10 голосовых команд
* **test_matmult_time.ipynb** - сравнение времени работы наивной реализации матричного умножения и pytorch.matmult
* **test_popcount.ipynb** - тестирование упаковки бинарных матриц в формат uint64_t и матричного умножения на основе операций XNOR и popcount

/cpp_src
C++ extension с матричным умножением бинарных матриц на основе операций XNOR и popcount

Результаты

<img
  src="/experiments/models/SimpleCNN/learning.jpg"
  title="Результаты обучения на фильтрованном датасете без бинаризации">

<img
  src="/experiments/models/NaiveXNORSimpleCNN/learning.jpg"
  title="Результаты обучения на фильтрованном датасете с бинаризацией с помощью наивной реализацией XNOR-Net">

<img
  src="/experiments/models/XNORSimpleCNN/learning.jpg"
  title="Результаты обучения на фильтрованном датасете с бинаризацией с помощью реализацией XNOR-Net из официального репозитория">

<img
  src="/experiments/models/SimpleCNNQuintized/learning.jpg"
  title="Результаты обучения на фильтрованном датасете с бинаризацией с помощью LSQ">
