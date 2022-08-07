for running on GPU:

salloc -p gpu
ssh gpu001

conda activate /home/nadavsc/Desktop/envs/targets
cd Desktop/Multi-View-Foams
export PYTHONPATH=$PYTHONPATH:$PWD
cd model2
export PYTHONPATH=$PYTHONPATH:$PWD
python -m train.py

scancel {job_id}
