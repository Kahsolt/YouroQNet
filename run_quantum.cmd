@ECHO OFF

IF /I "%1"=="inspect" GOTO inspect

:train
python run_quantum.py --analyzer char   --model Youro
python run_quantum.py --analyzer 2gram  --model Youro
python run_quantum.py --analyzer 3gram  --model Youro
python run_quantum.py --analyzer kgram  --model Youro
python run_quantum.py --analyzer 2gram+ --model Youro
python run_quantum.py --analyzer 3gram+ --model Youro
python run_quantum.py --analyzer kgram+ --model Youro

:inspect
python run_quantum.py --inspect
