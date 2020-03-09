#Variational Autoencoders

В данном проекте реализован [вариационный автоэнкодер](https://arxiv.org/abs/1312.6114). 

## MetaSaver 

Класс ```MetaSaver``` используется для декорирования исполняемых функций в наследуемых классах.

- Каждый вызов```.run_experiment(attributes_to_save, wrapper)``` создает директорию в папке, указанной в ```base_directory``` со следующим 
форматом: ```ДД-ММ-ГГГГ--ЧЧ-ММ-СС--ПОСТФИКС```, выполняет метод, указанный в ```wrapper``` и сохраняет все атрибуты, указанные в ```attributes_to_save```.

- ```MetaSaver``` реализует метод ```.load_model(dir)```, главной особенностью которого является обработка случая ```dir="last"```, который выполняет поиск по всем поддиректориям и возвращает последнюю обученную модель.

Пример использования:

> ```vae = VAE(num_epochs=10, dim_z=8, train_set=some_dataset, postfix='test', base_directory='./experiments')```
> ```vae.run_experiment(['train_loss_epoch']).load_model('last').generate_from_noise(some_data)```       

##Docker 
Для воспроизводимости эксперимента, весь код подготовлен к запуску внутри Docker-контейнера.
Для сборки образа необходимо выполнить:
```docker build -t vae .```

Запуск: ```docker run vae```

Перед запуском убедитесь что Docker-deamon запущен

Файл ```X_train.csv``` с тренировочными данными необходимо положить в текущую директорию.


##Tensorboard
Для удобства отслеживания метрик во время обучения, некоторые парамеры обучения (в частности, reconstruction loss), доступны в [дэшборде](localhost:6006).   