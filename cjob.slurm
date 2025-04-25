#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
#
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=CNN_Forward_Pass_C #Set the job name to "JobExample2"
#SBATCH --time=00:05:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=1           #Request 8 tasks/cores per node
#SBATCH --mem=8G                     #Request 8GB per node 
#SBATCH --output=forward_pass_execution_c_%j.out      #Send stdout/err to "output.[jobID]" 
#
##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=email_address    #Send all emails to email_address 
#
##First Executable Line
#
#module load GCC/11.3.0         # load Intel software stack 
# Compile code
gcc inference.c -lm -o c_compiled

#Run Executable
./c_compiled
