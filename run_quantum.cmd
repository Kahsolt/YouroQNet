@ECHO OFF

IF /I "%1"=="inspect" GOTO inspect

:train
python run_quantum.py --analyzer char   --model YouroQ --onehot --debug_step
python run_quantum.py --analyzer kgram  --model YouroQ --onehot --debug_step
python run_quantum.py --analyzer kgram+ --model YouroQ --onehot --debug_step

python run_quantum.py --analyzer char   --model YouroM --onehot --debug_step
python run_quantum.py --analyzer kgram  --model YouroM --onehot --debug_step
python run_quantum.py --analyzer kgram+ --model YouroM --onehot --debug_step

:inspect
python run_quantum.py --inspect
