import os
import re




directory = '../../../data/raw/figure3' #unprocessed data directory
lnames = set()
n_lakes = 0

qsub = ""
for filename in os.listdir(directory):
    #parse lakename from file
    m = re.search(r'^nhd_(\d+)_.*', filename)
    if m is None:
        continue
    name = m.group(1)
    if name not in lnames:
        #for each unique lake
        lnames.add(name)
        l = name
        header = "#!/bin/bash -l\n#PBS -l walltime=22:00:00,nodes=1:ppn=24:gpus=2,mem=16gb \n#PBS -m abe \n#PBS -N %s \n#PBS -o %s.stdout \n#PBS -q k40 \n"%(l,l)
        script = "source takeme.sh\n"
        script2 = "source activate pytorch4"
        script3 = "python experiment_correlation_check.py %s"%(l)
        all= "\n".join([header,script,script2,script3])
        qsub = "\n".join(["qsub job_%s.sh"%(l),qsub])
        with open('job_{}.sh'.format(l), 'w') as output:
            output.write(all)

with open('qsub_script.sh', 'w') as output2:
    output2.write(qsub)
