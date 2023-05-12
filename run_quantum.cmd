@ECHO OFF

IF /I "%1"=="eval" GOTO eval

:train
python run_quantum.py --analyzer char   --model Youro
python run_quantum.py --analyzer 2gram  --model Youro
python run_quantum.py --analyzer 3gram  --model Youro
python run_quantum.py --analyzer kgram  --model Youro
python run_quantum.py --analyzer 2gram+ --model Youro
python run_quantum.py --analyzer 3gram+ --model Youro
python run_quantum.py --analyzer kgram+ --model Youro

:eval
python run_quantum.py --eval
