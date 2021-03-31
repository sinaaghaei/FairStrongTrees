import os
import sys

path = '/Users/sina/Documents/GitHub/FairStrongTrees/'
approach_name = 'FairOCT' #
samples = [1,2,3,4,5]
depths = [1, 2]
time_limit = 7200
datasets = ['compas']
protected_feature = ['race_factor']
condition_feature = ['priors_buckets']
bounds = [0.01, 0.05, 0.1 , 0.2, 0.3, 0.5]
fairness_type = ["CSP", "None"]


def put_qmark(s):
        s = "\""+s+"\""
        return s


def generate():
        global time_limit, depths, samples, approach_name, datasets, protected_feature, bounds, fairness_type
        slurm_file = 'slurm_'+approach_name + '_' + str(time_limit)+".sh"
        dir="/scratch2/saghaei/FairStrongTrees/"+approach_name+"/"

        dataset_reg_list=[]
        dataset_enc_list=[]
        depth_list=[]
        sample_list = []
        fairness_type_list= []
        bounds_list = []
        protected_feature_list = []
        condition_feature_list = []
        for dset_index, dset in enumerate(datasets):
                for s in samples:
                    for d in depths:
                        for f in fairness_type:
                                if f == "None":
                                            dataset_enc_list.append(dset + '_enc' + '.csv')
                                            dataset_reg_list.append(dset + '.csv')
                                            depth_list.append(d)
                                            sample_list.append(s)
                                            fairness_type_list.append(f)
                                            bounds_list.append(1)
                                            protected_feature_list.append(protected_feature[dset_index])
                                            condition_feature_list.append(condition_feature[dset_index])
                                else:
                                    for bound in bounds:
                                            dataset_enc_list.append(dset + '_enc' + '.csv')
                                            dataset_reg_list.append(dset + '.csv')
                                            depth_list.append(d)
                                            sample_list.append(s)
                                            fairness_type_list.append(f)
                                            bounds_list.append(bound)
                                            protected_feature_list.append(protected_feature[dset_index])
                                            condition_feature_list.append(condition_feature[dset_index])

        S="#!/bin/bash\n"
        # S+="#SBATCH --ntasks=100\n"
        S+="#SBATCH --ntasks=1\n"
        S+="#SBATCH --cpus-per-task=4\n"
        S+="#SBATCH --mem-per-cpu=4GB\n"
        S+="#SBATCH --time=03:00:00\n"
        S+="#SBATCH --export=NONE\n"
        S+="#SBATCH --constraint=\"xeon-2640v4\"\n"
        S+="#SBATCH --array=0-59\n"
        S+="\n"
        S+="\n"
        S+="module load gcc\n"
        S+="module load gurobi\n"
        S+="module load python\n"
        S+="\n"
        S+="export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}"+"\n"
        S+="\n"

        S+="dataset_enc_list=" + put_qmark(" ".join(str(item) for item in dataset_enc_list) + "\n")
        S+="\n"
        S+="dataset_reg_list=" + put_qmark(" ".join(str(item) for item in dataset_reg_list) + "\n")
        S+="\n"
        S+="depth_list=" + put_qmark(" ".join(str(item) for item in depth_list) + "\n")
        S+="\n"
        S+="sample_list=" + put_qmark(" ".join(str(item) for item in sample_list) + "\n")
        S+="\n"
        S+="fairness_type_list=" + put_qmark(" ".join(str(item) for item in fairness_type_list) + "\n")
        S+="\n"
        S+="bounds_list=" + put_qmark(" ".join(str(item) for item in bounds_list) + "\n")
        S+="\n"
        S+="protected_feature_list=" + put_qmark(" ".join(str(item) for item in protected_feature_list) + "\n")
        S+="\n"
        S+="condition_feature_list=" + put_qmark(" ".join(str(item) for item in condition_feature_list) + "\n")
        S+="\n"
        S+='dataset_enc_list=($dataset_enc_list)'+ "\n"
        S+='dataset_reg_list=($dataset_reg_list)'+ "\n"
        S+='depth_list=($depth_list)'+ "\n"
        S+='sample_list=($sample_list)'+ "\n"
        S+='fairness_type_list=($fairness_type_list)'+ "\n"
        S+='bounds_list=($bounds_list)'+ "\n"
        S+='protected_feature_list=($protected_feature_list)'+ "\n"
        S+='condition_feature_list=($condition_feature_list)'+ "\n"

        S+="\n"
        S+="\n"
        command = 'python FlowOCTReplication.py ' + '-r ' +'${dataset_reg_list[$SLURM_ARRAY_TASK_ID]}' + ' -f ' +'${dataset_enc_list[$SLURM_ARRAY_TASK_ID]}' + " -d " + '${depth_list[$SLURM_ARRAY_TASK_ID]}' + " -t " + str(time_limit) + " -l " + str(0)+ " -i " + '${sample_list[$SLURM_ARRAY_TASK_ID]}'+ " -c " + str(1) + " -a " + '${fairness_type_list[$SLURM_ARRAY_TASK_ID]}'+ " -b " + '${bounds_list[$SLURM_ARRAY_TASK_ID]}'+" -e " + '${protected_feature_list[$SLURM_ARRAY_TASK_ID]}'+" -g " + str(2) + " -h " + '${condition_feature_list[$SLURM_ARRAY_TASK_ID]}'
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
