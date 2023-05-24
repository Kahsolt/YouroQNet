@ECHO OFF

IF /I "%1"=="inspect" GOTO inspect

:train
python run_quantum.py --analyzer char   --model YouroQ --debug_step
python run_quantum.py --analyzer kgram  --model YouroQ --debug_step
python run_quantum.py --analyzer kgram+ --model YouroQ --debug_step

python run_quantum.py --analyzer char   --model YouroM --debug_step
python run_quantum.py --analyzer kgram  --model YouroM --debug_step
python run_quantum.py --analyzer kgram+ --model YouroM --debug_step

:inspect
python run_quantum.py --inspect
