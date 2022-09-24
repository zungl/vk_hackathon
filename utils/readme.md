## Кастомный модуль с утилитами
С функциями для поиска оптимальных гиперпараметров Feature_selection, Featrue_importance и обучения модели
<hr>

```python
## Пример использования пайплайна
import utils

help(utils.pipeline)

pipline_response = utils.pipeline(
    features_df            =features_df.copy(),
    train_df               =train_df,
    val_df                 =val_df,
    left_on                ='CLIENT_ID',
    right_on               ='CLIENT_ID',
    tech_cols              =['CLIENT_ID'],
    target_column          ='TARGET',
    drop_singular_cut_off  =0.9,
    drop_corr_cut_off      =0.75,
    n_trials               =50,
    model_filename         ='model_friends.txt'
)
```

```python
## autopca
utils.group_columns(df=features_df.copy(), ids='CLIENT_ID', n_components=5)
## Удаление признаков
utils.drop_singular(df, cut_off=0.9)
utils.drop_corr(df, cut_off=0.9)

## other
utils.backward_selection(argList, model, x_train, y_train, x_test, y_test)
utils.get_top_features(alg, target_name, params, x_train, y_train, x_test, y_test)
utils.objective(trial, x_train, y_train, x_test, y_test, alg)
```