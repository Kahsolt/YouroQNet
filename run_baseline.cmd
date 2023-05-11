@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION

IF /I "%1"=="eval" GOTO eval

:train
ECHO start train sk...

python run_baseline_sk.py --analyzer char  --feature tfidf
python run_baseline_sk.py --analyzer word  --feature tfidf
python run_baseline_sk.py --analyzer 2gram --feature tfidf
python run_baseline_sk.py --analyzer 3gram --feature tfidf
python run_baseline_sk.py --analyzer kgram --feature tfidf

python run_baseline_sk.py --analyzer char  --feature fasttext
python run_baseline_sk.py --analyzer word  --feature fasttext
python run_baseline_sk.py --analyzer sent  --feature fasttext
python run_baseline_sk.py --analyzer 2gram --feature fasttext
python run_baseline_sk.py --analyzer 3gram --feature fasttext
python run_baseline_sk.py --analyzer kgram --feature fasttext

ECHO start train vq...

SET id=0
SET MODEL=
SET FINAL=

REM mode config naming patterns:
REM   dnn-[type:str]-[dim:int]-[agg:str]
REM   cnn-[type:str]-[dim:int]-[agg:str]
REM   rnn-[type:str]-[dim:int]-[agg:str]-[dir:str]
REM where the valid string choices are:
REM            DNN      CNN       RNN
REM   type = std/res   1d/2d    gru/lstm
REM   agg  = avg/max  avg/max  avg/max/fin
REM   dir  =                     uni/bi

:vq0
SET MODEL=dnn-std-80-max
GOTO train_vq 

:vq1
SET MODEL=dnn-res-32-avg
GOTO train_vq

:vq2
SET MODEL=cnn-1d-80-avg
GOTO train_vq

:vq3
SET MODEL=cnn-2d-80-avg
GOTO train_vq

:vq4
SET MODEL=rnn-lstm-80-avg-uni
GOTO train_vq

:vq5
SET MODEL=rnn-gru-80-avg-bi
SET FINAL=1
GOTO train_vq

:train_vq
python run_baseline_vq.py --analyzer char   --model !MODEL!
python run_baseline_vq.py --analyzer 2gram  --model !MODEL!
python run_baseline_vq.py --analyzer 3gram  --model !MODEL!
python run_baseline_vq.py --analyzer kgram  --model !MODEL!
python run_baseline_vq.py --analyzer 2gram+ --model !MODEL!
python run_baseline_vq.py --analyzer 3gram+ --model !MODEL!
python run_baseline_vq.py --analyzer kgram+ --model !MODEL!

SET /A id = !id! + 1 > NUL
IF "!FINAL!"=="1" (
  GOTO train_vq_done
) ELSE (
  GOTO vq!id!
)
:train_vq_done


:eval
ECHO start eval sk...
python run_baseline_sk.py --eval

ECHO start eval vq...
python run_baseline_vq.py --eval

ECHO Done.
