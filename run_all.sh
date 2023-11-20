rm -f *.out
rm -f *.run

for i in {1..9}
do

echo "#!/bin/bash -l
#SBATCH -J grawiti_$i
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00

module load gcc python
source /home/juleuenb/venvs/c3mp/bin/activate
srun python3 2_study_paths.py --maxcomplexity 100000000 --maxpaths 1000 --restrict $i
deactivate" > submission_$i.run

sbatch submission_$i.run

done
