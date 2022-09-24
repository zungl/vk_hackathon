# Директория для файлов с данными

## Папки:
`clean_data`  - предобработанные данные<br>
`submissions` - файлы с разметкой тестовой части данных<br>

## Исходные файлы:

```python
## Получение исходных данных
!wget https://ai-data.obs.ru-moscow-1.hc.sbercloud.ru/МО_на_графах.zip -O МО_на_графах.zip

from zipfile import ZipFile
for file in tqdm_notebook(zf.infolist()[2:]): 
    zf.extract(file) 
```
Файлы из архива <МО_на_графах.zip>:

`FINAL_ALL_SEQUENCES_TRAINTEST.tsv`<br>
`FINAL_FEATURES_FRIENDS.tsv`<br>
`FINAL_FEATURES_TRAINTEST.tsv`<br>
`FINAL_SEQUENCES_MATRIX.tsv`<br>
`FINAL_TARGETS_DATES_TRAINTEST.tsv`<br>