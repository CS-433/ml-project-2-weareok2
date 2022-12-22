## Folder structure
```
–twitter-datasets/
 |train_pos_full.txt
 |train_neg_full.txt
 |test_data.csv
–transf/
 |build_vocab.sh
 |cut_vocab.sh
 |pickle_vocab.py
 |transf.py
 |_transf.py
 |transfPred.py
```

## Preprocessing
To produce first the dictionary of tokens.
```bash
build_vocab.sh
cut_vocab.sh
python3 pickle_vocab.py
```

## Training
Require pytorch library and a GPU.
```bash
python transf.py embedding_dimension \
    number_of_layers \
    number_of_heads \
    learning_rate \
    batch_size \
    output_file_for_trained_model
```

## Prediction
Require pytorch library.
```bash
python transfPrediction.py output_file input_trained_model
```
