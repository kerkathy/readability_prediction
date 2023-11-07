# Readability Prediction

Download OneStopEnglish data.
```
git clone https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git
```

Genereate train, validation, testing data pair splits, with ratio 3:1:1.
```
./dataset_to_csv.py
```

Move it into weak_signal directory.
```
mkdir multitask-learning-transformers/weak_signal/data/ose
mv *pair.csv multitask-learning-transformers/weak_signal/data/ose
```