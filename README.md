# recsys

Вы можете запустить этот код на своей машине для  этого нужно запустить файл project/scripts/run_model.py или файл с экспериментами project/scripts/experiments.ipynb.

Чтобы их запустить нужно сделать следующее
1. Для начало нужно скачать либы с файла requirements.txt, с помощью pip install -r requirements.txt
2. Получите путь до своего файла (пример такого пути /home/vladislav/python/recsys/) получить этот путь до папки recsys можно с помощью команды pwd
3. Перейдите в консоле в папку с этими файлами recsys/project/scripts/
4. Скачав этот файл, скачав либы из requirements и получив путь до папки recsys, можно запустить файл для тренировки модели с помощью команды     PYTHONPATH=(путь до recsys папки) python3 run_model.py, пример команды (PYTHONPATH=/home/vladislav/python/recsys/ python3 run_model.py), чтобы запустить файл с экспериментами введите в терминале команду PYTHONPATH=(путь до recsys папки) jupyter-lab


О коде и эксперименте смотрите recsys.docx файл
