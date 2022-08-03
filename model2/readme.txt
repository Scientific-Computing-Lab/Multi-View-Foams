for running on GPU:

salloc -p gpu
ssh gpu001

cd Desktop/projects/targets
conda activate /home/nadavsc/Desktop/envs/targets
export PYTHONPATH=$PYTHONPATH:$PWD
cd model2
export PYTHONPATH=$PYTHONPATH:$PWD
python -m train.py

scancel {job_id}
