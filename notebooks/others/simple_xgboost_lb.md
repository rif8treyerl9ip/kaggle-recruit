https://www.kaggle.com/festa78/simple-xgboost-lb-0-495


#### 実装(設計)方法

- 辞書に複数Dataframeを持たせる方法

以上のように辞書のキーと値に変数とdataframeを持たせることで、
各dataframeに簡単にアクセスでき、可読性が著しく向上する。

```
data = {
    'tra':
    pd.read_csv('../../input/air_visit_data.csv'),
    'as':
    pd.read_csv('../../input/air_store_info.csv'),
    'hs':
    pd.read_csv('../../input/hpg_store_info.csv'),
    'ar':
    pd.read_csv('../../input/air_reserve.csv'),
    'hr':
    pd.read_csv('../../input/hpg_reserve.csv'),
    'id':
    pd.read_csv('../../input/store_id_relation.csv'),
    'tes':
    pd.read_csv('../../input/sample_submission.csv'),
    'hol':
    pd.read_csv('../../input/date_info.csv') \
    .rename(columns={
        'calendar_date': 'visit_date'
    })
}

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])
```

例えば、以下の部分では二つのデータフレームに共通した処理を簡単に施すことができている。
関数と異なり、上から下に流れていく処理を目で追えばよいので可読性がよく感じる。

```
for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(
        lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df].groupby(
        ['air_store_id', 'visit_datetime'], as_index=False)[[
            'reserve_datetime_diff', 'reserve_visitors'
        ]].sum().rename(columns={
            'visit_datetime': 'visit_date'
        })
    print(data[df].head())
```

- ベースとなるdataframeを作成し、そこに集計した特徴量を横方向に繰り返し結合していく

```
#sure it can be compressed...
tmp = data['tra'].groupby(
['air_store_id', 'dow'],
as_index=False)['visitors'].min().rename(columns={
    'visitors': 'min_visitors'
})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])```
```

#### 前処理

- 文字列をsplitして該当部分のみをとってくる操作
- 'air_00a91d42b08b08d9_2017-04-23'のような文字列から、アンスコで分割して一番後ろの日付のみをとってくる

```
data['tes']['visit_date'] = data['tes']['id'].map(
    lambda x: str(x).split('_')[2]) # 2017-05-31部分の取得
```

- 'air_00a91d42b08b08d9_2017-04-23'のような文字列から、アンスコで分割して前二つをとってくる

```
data['tes']['air_store_id'] = data['tes']['id'].map(
    lambda x: '_'.join(x.split('_')[:2]))
```


- labelencoding、単純に自分が見たことない実装方法だっただけ

```
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])
```

train.columnsとしなくともdataframeの各イテレータ？は列名らしい。
seriesが格納されると思ったけどなぜだろうか。
```
col = [
    c for c in train
    if c not in ['id', 'air_store_id', 'visit_date', 'visitors']
]
```

あとでnanの理由を突き止める、あとー1で補完した理由も

- scikit-learnAPIは知らないが、xgboostのpythonAPI(恐らく)だと内部でnp.float32に変換されるらしいので、先に変換しておく。

```
for c, dtype in zip(train.columns, train.dtypes):
    if dtype == np.float64:
        train[c] = train[c].astype(np.float32)

for c, dtype in zip(test.columns, test.dtypes):
    if dtype == np.float64:
        test[c] = test[c].astype(np.float32)
```


#### コードリーディングの方法
vscodeデュアルモニター
片方はdataframeの変化を都度表示する

#### 考え方の部分

- 訓練データの目的変数の最小値・中央値・平均値・最大値・カウントを特徴量として与えようという考え。
一見して気温や緯度・経度が役に立つ特徴量でなさそうなことを考えると、目的変数が持つ情報を使うのはアリ
テストデータには日付に由来する特徴量を入れることができないので、曜日に由来する特徴量を持ってくることでリークを防いでいる。


##### 感想

- 逆算して読もうとしても、読めないときはどうしても実装が読み取れない。というか慣れで読めるようになってきたので本当に慣れなんだろう。
- ワイドフォーマットとロングフォーマットの話があったが、ここで双方の変換を行うメリットはあまりなさそうに見え、コードがかなり複雑になりそう。


vscode便利機能
上記の可読性の項で思いついたのだが、
https://nodoame.net/archives/10866
を参考にsettings.jsonを

`"workbench.colorCustomizations": {
    "editor.lineHighlightBackground": "#404040"
}`
のように編集すると、現在カーソルがある行が明るくなり作業しやすい。