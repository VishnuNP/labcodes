@echo off
echo Running and comparing..
nvcc RKCKpaper_parallel.cu -o niemeyer_sim
niemeyer_sim.exe
python2 niemeyer_vs_odeint.py
echo DONE
pause