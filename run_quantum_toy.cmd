@ECHO OFF

:bin_clf
REM train & view embed
python vis_youroqnet_toy.py -B 1 %*
REM view grid sample
python vis_qc_func.py


:ablation_bias
REM this experiment is extremely sensitive to rand seed :(
SET RAND_SEED_SAVED=%RAND_SEED%
SET RAND_SEED=114514
python vis_youroqnet_toy.py -B 1 --bias pos
SET RAND_SEED=%RAND_SEED_SAVED%

REM this is not that sensitive, should be fine :)
python vis_youroqnet_toy.py -B 1 --bias neg


:tri_clf
REM train & view embed
python vis_youroqnet_toy.py --tri --debug_step -B 4 --n_repeat 4
