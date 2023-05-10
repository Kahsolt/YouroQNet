@ECHO OFF

REM extra data
python mk_data.py

REM stats & plots
python mk_stats.py

REM heuristical tokenizer
python mk_vocab.py --ngram
python mk_vocab.py --kgram
