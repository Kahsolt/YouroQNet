@ECHO OFF

IF /I "%1"=="eval" GOTO eval

:train
python run_baseline_sk.py --analyzer char --feature tfidf
python run_baseline_sk.py --analyzer word --feature tfidf

python run_baseline_sk.py --analyzer char --feature fasttext
python run_baseline_sk.py --analyzer word --feature fasttext
python run_baseline_sk.py --analyzer sent --feature fasttext

python run_baseline_vq.py --analyzer char  --model DNN
python run_baseline_vq.py --analyzer 2gram --model DNN
python run_baseline_vq.py --analyzer 3gram --model DNN
python run_baseline_vq.py --analyzer kgram --model DNN

python run_baseline_vq.py --analyzer char  --model CNN
python run_baseline_vq.py --analyzer 2gram --model CNN
python run_baseline_vq.py --analyzer 3gram --model CNN
python run_baseline_vq.py --analyzer kgram --model CNN

python run_baseline_vq.py --analyzer char  --model RNN
python run_baseline_vq.py --analyzer 2gram --model RNN
python run_baseline_vq.py --analyzer 3gram --model RNN
python run_baseline_vq.py --analyzer kgram --model RNN

:eval
python run_baseline_sk.py --eval
python run_baseline_vq.py --eval
