#!/bin/bash
#SBATCH --job-name=TMLDOSmin_Qabs1e1_des10by10_chi7d9_emit10_maxeval20000_gpr100_sixpoles_rand1
#SBATCH --output=TM_LDOSmin_dipole_Qabs1e1_des10by10_chi7d9_emit10_maxeval20000_gpr100_sixpoles_randstart1.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20-00:00:00
#SBATCH --mem-per-cpu=11000
#SBATCH --error=TM_errorminQabs1e1_des10by10_chi7d9_emit10_maxeval20000_gpr100_sixpoles_rand1.err
 
module load anaconda3/2023.9
conda activate PB_env

# defines the program name to be run
prog=rampQabs_TM_DOSmin_PhC.py

# input parameters for the calculation
wavelength=1.0
pow10Qabs_start=1.0
pow10Qabs_end=1.0
pow10Qabs_num=1
Num_Poles=6
geometry='Cavity'

ReChi=7.9
ImChi=0.0

# number of gridpoints per length equal to 1
gpr=100

# size of design region (in x and y)
design_x=10.0
design_y=10.0

# size of cavity/vacuum region in the design region
vacuum_x=0.0
vacuum_y=0.0
dist_x=0.0

# size of region of dipoles (in x and y)
emitter_x=10.0
emitter_y=10.0

# more simulation parameters (for convergence)
pml_thick=0.5
pml_sep=1.0

# initialization of design region with material
init_type='rand'
init_file='run_Qabs1e1_maxeval1000_des10by10_chi7d9_twelveshort3_TM_rand1.txt'

# number of iterations after which to output the current design
output_base=50

# maximum number of iterations
maxeval=20000

# name of ouput file
name='TMmin_Qabs1e1_des10by10_chi7d9_emit10_maxeval20000_gpr100_sixpoles_rand1'
 

# run the program
python3 $prog -wavelength $wavelength -pow10Qabs_start $pow10Qabs_start -pow10Qabs_end $pow10Qabs_end -pow10Qabs_num $pow10Qabs_num -ReChi $ReChi -ImChi $ImChi -gpr $gpr -design_x $design_x -design_y $design_y -vacuum_x $vacuum_x -vacuum_y $vacuum_y -emitter_x $emitter_x -emitter_y $emitter_y -pml_thick $pml_thick -pml_sep $pml_sep -init_type $init_type -init_file $init_file -output_base $output_base -name $name -maxeval $maxeval -dist_x $dist_x -Num_Poles $Num_Poles -geometry $geometry >> TM_LDOSmin_Qabs1e1_des10by10_chi7d9_emit10_maxeval20000_gpr100_sixpoles_rand1.txt

