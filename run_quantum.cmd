@ECHO OFF

IF /I "%1"=="eval" GOTO eval

:train
python run_quantum.py --analyzer char
python run_quantum.py --analyzer 2gram
python run_quantum.py --analyzer 3gram
python run_quantum.py --analyzer kgram

:eval
python run_quantum.py --eval
