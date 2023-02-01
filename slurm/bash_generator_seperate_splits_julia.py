import os
import sys


path = '/Users/sina/Documents/GitHub/FairStrongTrees/'
approach_name = 'MIP_DT_DIDI'


data_group = 'limited-adult-2'# german_binary german compas adult default limited_adult
KamiranVersion = 0
samples = [1,2,3,4,5]
depths = [3]
lambdas = [x / 100.0 for x in range(0, 100, 2)]
time_limit = 10800




def put_qmark(s):
        s = "\""+s+"\""
        return s




def generate():
        global time_limit, depths, samples, approach_name, data_group, lambdas
        slurm_file = f'slurm_{approach_name}_{time_limit}_{data_group}.sh'
        dir=f"/project/vayanou_651/FairStrongTrees/Code/{approach_name}/"

        sample_list = []
        depth_list=[]
        lambdas_list = []
        for s in samples:
                for d in depths:
                        for l in lambdas:
                                depth_list.append(d)
                                sample_list.append(s)
                                lambdas_list.append(l)


        S="#!/bin/bash\n"
        # S+="#SBATCH --ntasks=100\n"
        S+="#SBATCH --ntasks=1\n"
        S+="#SBATCH --cpus-per-task=4\n"
        S+="#SBATCH --mem-per-cpu=14GB\n"
        S+="#SBATCH --time=05:00:00\n"
        S+="#SBATCH --export=NONE\n"
        S+="#SBATCH --constraint=\"xeon-2640v4\"\n"
        S+=f"#SBATCH --array=0-{len(sample_list)-1}\n"
        S+="\n"
        S+="\n"
        S+=f"cd {dir}"
        S+="\n"
        S+="\n"
        S+="module load gcc\n"
        S+="module load gurobi\n"
        S+="module load julia\n"
        S+="\n"
        S+="export JULIA_DEPOT_PATH=/project/vayanou_651/julia/pkgs"+"\n"
        S+="\n"

        S+="depth_list=" + put_qmark(" ".join(str(item) for item in depth_list) + "\n")
        S+="\n"
        S+="sample_list=" + put_qmark(" ".join(str(item) for item in sample_list) + "\n")
        S+="\n"
        S+="lambdas_list=" + put_qmark(" ".join(str(item) for item in lambdas_list) + "\n")
        S+="\n"
        S+='depth_list=($depth_list)'+ "\n"
        S+='sample_list=($sample_list)'+ "\n"
        S+='lambdas_list=($lambdas_list)'+ "\n"

        S+="\n"
        S+="\n"

        command_sample = '${sample_list[$SLURM_ARRAY_TASK_ID]}'
        command_depth = '${depth_list[$SLURM_ARRAY_TASK_ID]}'
        command_lambda = '${lambdas_list[$SLURM_ARRAY_TASK_ID]}'
        command = f'julia MIP_DT_DIDI.jl {data_group} {KamiranVersion} {command_sample} {command_depth} {command_lambda} {time_limit}'

        S+=command
        S+="\n"



        dest_dir=path
        f= open(dest_dir+slurm_file,"w+")
        f.write(S)
        f.close()
        print(slurm_file)




def main():
        generate()




if __name__ == "__main__":
    main()
