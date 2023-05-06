@ECHO OFF

IF /I "%1"=="eval" GOTO eval

:train
python run_quantum.py --feature tfidf --analyzer char
python run_quantum.py --feature tfidf --analyzer word

python run_quantum.py --feature fasttext --analyzer char
python run_quantum.py --feature fasttext --analyzer word
python run_quantum.py --feature fasttext --analyzer sent

:eval
python run_quantum.py --eval
