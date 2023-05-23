@ECHO OFF

IF /I "%1"=="inspect" GOTO inspect

:train
python run_quantum.py --analyzer char   --model YouroQ
python run_quantum.py --analyzer kgram  --model YouroQ
python run_quantum.py --analyzer kgram+ --model YouroQ

python run_quantum.py --analyzer char   --model YouroM
python run_quantum.py --analyzer kgram  --model YouroM
python run_quantum.py --analyzer kgram+ --model YouroM

:inspect
python run_quantum.py --inspect
