from decimal import Decimal
import os

app_name="1"
# app_name="2a"

dir="/home/rcf-proj2/ma2/azizim/juliarun/AAAI/classification/1"
# dir="/home/rcf-proj2/ma2/azizim/juliarun/AAAI/classification/2a"

exe_name="a1.jl"
# exe_name="2a.jl"


data_set_name="cen" # cen df com
sample='s1';#or s1


lam_num=[0,1,1,1,10]
lam_denum=[1,10000,10,1,1]



if data_set_name=="cen":
    if sample=='s1':
        tr_start=2901
        tr_N=1000
        tes_start=4001
        tes_N=1000
    elif sample=='s2':
        tr_start=16001
        tr_N=2000
        tes_start=10001
        tes_N=1000
    elif sample=='s3':
        tr_start=3001
        tr_N=2000
        tes_start=11001
        tes_N=2000
    elif sample=='s4':
        tr_start=8001
        tr_N=4000
        tes_start=1
        tes_N=4000
    elif sample=='s5':
        tr_start=12001
        tr_N=5000
        tes_start=3001
        tes_N=4000
elif data_set_name=="df":
    if sample=='s1':
        tr_start=1
        tr_N=1000
        tes_start=1001
        tes_N=1000
    elif sample=='s2':
        tr_start=2001
        tr_N=1000
        tes_start=3501
        tes_N=500
    elif sample=='s3':
        tr_start=4001
        tr_N=2000
        tes_start=6001
        tes_N=2000
    elif sample=='s4':
        tr_start=8001
        tr_N=4000
        tes_start=12001
        tes_N=5000
    elif sample=='s5':
        tr_start=17001
        tr_N=5000
        tes_start=23001
        tes_N=7000
elif data_set_name=="com":
    if sample=='s1':
        tr_start=1
        tr_N=1000
        tes_start=1101
        tes_N=1000
    elif sample=='s2':
        tr_start=4103
        tr_N=2000
        tes_start=2102
        tes_N=2000
    elif sample=='s3':
        tr_start=6104
        tr_N=1000
        tes_start=8105
        tes_N=1500
    elif sample=='s4':
        tr_start=7001
        tr_N=3000
        tes_start=1000
        tes_N=3000
    elif sample=='s5':
        tr_start=1
        tr_N=5000
        tes_start=5001
        tes_N=4000


for ind in range(len(lam_num)):
    lambda_num=lam_num[ind]
    lambda_denum=lam_denum[ind]
    if lambda_num==0:
        fair=0
    else:
        fair=1
    time_lim=80000
    mip_gap_num=5
    nothing=1
    depth='none'

    lam=lambda_num/lambda_denum
    tmp = '%.2E' % Decimal(str(lam))
    lam="lam"+tmp[4:]
    lam="lam_"+str(lambda_num)+"_"+str(lambda_denum)

    if fair==0:
        lam='lam0'

    args=[sample,tr_N,tr_start,fair,lambda_num,nothing,tes_N,tes_start,time_lim,mip_gap_num,lambda_denum,data_set_name,depth]
    tmp_arg=""
    for i in range(13):
        tmp_arg+=str(args[i])
        tmp_arg+=" "

    command="julia "+exe_name+" "+tmp_arg


    output_name=app_name+"_"+data_set_name+"_"+sample+"_f"+str(fair)+"_"+lam+"_time"+str(time_lim)+"s"+".txt"
    output_file=app_name+"_"+data_set_name+"_"+sample+"_f"+str(fair)+"_"+lam+"_time"+str(time_lim)+"s"+".slurm"


    S="#!/bin/bash\n"
    # S+="#SBATCH --ntasks=100\n"
    S+="#SBATCH --ntasks=5\n"
    S+="#SBATCH --cpus-per-task=20\n"
    S+="#SBATCH --mem=64GB\n"
    S+="#SBATCH --time=24:00:00\n"
    # S+="#SBATCH --mem-per-cpu=16G\n"
    S+="#SBATCH --export=NONE\n"
    S+="#SBATCH --output="+output_name+"\n"
    S+="\n"
    S+="cd "+dir+"\n"
    S+="\n"
    S+="\n"
    S+="source /home/rcf-proj2/ma2/azizim/software/julia/0.6.0/setup.sh\n"
    # S+="export JULIA_PKGDIR=/home/rcf-proj2/sa3/juliaPkg\n"
    S+="source /usr/usc/gurobi/default/setup.sh\n"
    S+="\n"
    S+="\n"
    S+="#julia file_name.jl sample(s1 s2 ...)/N_tr/start_tr/fair/lambda/nothing/N_tes/start_tes/time_limit/MIP_GAP/lambda_denominator/data_set/depth\n"
    S+=command

    dest_dir="/Users/sina/Desktop/AAAI/classification/"+app_name+"/slurm files/"
    # try:
    #     os.remove(dest_dir+output_file)
    # except OSError:
    #     pass
    f= open(dest_dir+output_file,"w+")
    f.write(S)
    f.close()
