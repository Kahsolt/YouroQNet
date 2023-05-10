@ECHO OFF

REM extra data
ECHO mk_data
python mk_data.py

REM stats & plots
ECHO mk_stats
python mk_stats.py

REM heuristical tokenizer
ECHO mk_vocab
python mk_vocab.py --ngram
python mk_vocab.py --kgram

ECHO Done.
