import os
import sys

path = '/Users/sina/Documents/GitHub/FairStrongTrees/'
approach_name = 'FairOCT' #
samples = [1,2,3,4,5]
depths = [2]
time_limit = 10800
datasets = ['german']# german compas adult
protected_feature = ['age']# age race sex
condition_feature = ['credit_history']# credit_history priors_count education
bounds = [x / 100.0 for x in range(1, 56, 1)]#[x / 100.0 for x in range(1, 56, 1)] #[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
fairness_type = ['None', 'SP']


_lambda = 0
calibration_mode = 1
positive_class = 2



def put_qmark(s):
        s = "\""+s+"\""
        return s




def generate():
        global time_limit, depths, samples, approach_name, datasets, protected_feature, bounds, fairness_type, condition_feature
        slurm_file = f'slurm_{approach_name}_{time_limit}.sh'
        dir=f"/project/vayanou_651/FairStrongTrees/{approach_name}/"

        data_train_reg_list=[]
        data_train_enc_list=[]

        data_test_reg_list=[]
        data_test_enc_list=[]

        data_calibration_reg_list=[]
        data_calibration_enc_list=[]

        depth_list=[]
        sample_list = []
        fairness_type_list= []
        bounds_list = []
        protected_feature_list = []
        condition_feature_list = []
        for dset_index, dset in enumerate(datasets):
                for s in samples:
                        if calibration_mode == 1:
                                train_file_reg = f'{dset}_train_calibration_{s}.csv'
                                train_file_enc = f'{dset}_train_calibration_enc_{s}.csv'
                        else:
                                train_file_reg = f'{dset}_train_{s}.csv'
                                train_file_enc = f'{dset}_train_enc_{s}.csv'
                        test_file_reg = f'{dset}_test_{s}.csv'
                        test_file_enc = f'{dset}_test_enc_{s}.csv'
                        calibration_file_reg = f'{dset}_calibration_{s}.csv'
                        calibration_file_enc = f'{dset}_calibration_enc_{s}.csv'
                        for d in depths:
                                for f in fairness_type:
                                        if f == "None":
                                                    data_train_reg_list.append(train_file_reg)
                                                    data_train_enc_list.append(train_file_enc)
                                                    data_test_reg_list.append(test_file_reg)
                                                    data_test_enc_list.append(test_file_enc)
                                                    data_calibration_reg_list.append(calibration_file_reg)
                                                    data_calibration_enc_list.append(calibration_file_enc)
                                                    depth_list.append(d)
                                                    sample_list.append(s)
                                                    fairness_type_list.append(f)
                                                    bounds_list.append(1)
                                                    protected_feature_list.append(protected_feature[dset_index])
                                                    condition_feature_list.append(condition_feature[dset_index])
                                        else:
                                            for bound in bounds:
                                                    data_train_reg_list.append(train_file_reg)
                                                    data_train_enc_list.append(train_file_enc)
                                                    data_test_reg_list.append(test_file_reg)
                                                    data_test_enc_list.append(test_file_enc)
                                                    data_calibration_reg_list.append(calibration_file_reg)
                                                    data_calibration_enc_list.append(calibration_file_enc)
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
        S+="#SBATCH --time=04:00:00\n"
        S+="#SBATCH --export=NONE\n"
        S+="#SBATCH --constraint=\"xeon-2640v4\"\n"
        S+=f"#SBATCH --array=0-{len(data_train_enc_list)-1}\n"
        S+="\n"
        S+="\n"
        S+=f"cd {dir}"
        S+="\n"
        S+="\n"
        S+="module load gcc\n"
        S+="module load gurobi\n"
        S+="module load python\n"
        S+="\n"
        S+="export PYTHONPATH=/project/vayanou_651/python/pkgs:${PYTHONPATH}"+"\n"
        S+="\n"

        S+="data_train_enc_list=" + put_qmark(" ".join(str(item) for item in data_train_enc_list) + "\n")
        S+="\n"
        S+="data_train_reg_list=" + put_qmark(" ".join(str(item) for item in data_train_reg_list) + "\n")
        S+="\n"
        S+="data_test_enc_list=" + put_qmark(" ".join(str(item) for item in data_test_enc_list) + "\n")
        S+="\n"
        S+="data_test_reg_list=" + put_qmark(" ".join(str(item) for item in data_test_reg_list) + "\n")
        S+="\n"
        S+="data_calibration_enc_list=" + put_qmark(" ".join(str(item) for item in data_calibration_enc_list) + "\n")
        S+="\n"
        S+="data_calibration_reg_list=" + put_qmark(" ".join(str(item) for item in data_calibration_reg_list) + "\n")
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
        S+='data_train_enc_list=($data_train_enc_list)'+ "\n"
        S+='data_train_reg_list=($data_train_reg_list)'+ "\n"
        S+='data_test_enc_list=($data_test_enc_list)'+ "\n"
        S+='data_test_reg_list=($data_test_reg_list)'+ "\n"
        S+='data_calibration_enc_list=($data_calibration_enc_list)'+ "\n"
        S+='data_calibration_reg_list=($data_calibration_reg_list)'+ "\n"
        S+='depth_list=($depth_list)'+ "\n"
        S+='sample_list=($sample_list)'+ "\n"
        S+='fairness_type_list=($fairness_type_list)'+ "\n"
        S+='bounds_list=($bounds_list)'+ "\n"
        S+='protected_feature_list=($protected_feature_list)'+ "\n"
        S+='condition_feature_list=($condition_feature_list)'+ "\n"

        S+="\n"
        S+="\n"
        command = 'python FlowOCTReplication.py '+ ' --train_file_reg ' +'${data_train_reg_list[$SLURM_ARRAY_TASK_ID]}'+ ' --train_file_enc ' +'${data_train_enc_list[$SLURM_ARRAY_TASK_ID]}'+ ' --test_file_reg ' +'${data_test_reg_list[$SLURM_ARRAY_TASK_ID]}'+ ' --test_file_enc ' +'${data_test_enc_list[$SLURM_ARRAY_TASK_ID]}'+ ' --calibration_file_reg ' +'${data_calibration_reg_list[$SLURM_ARRAY_TASK_ID]}'+ ' --calibration_file_enc ' +'${data_calibration_enc_list[$SLURM_ARRAY_TASK_ID]}' + " --depth " + '${depth_list[$SLURM_ARRAY_TASK_ID]}' + " --timelimit " + str(time_limit) + " -i " + str(_lambda)+ " --sample " + '${sample_list[$SLURM_ARRAY_TASK_ID]}'+ " --calibration_mode " + str(calibration_mode) + " --fairness_type " + '${fairness_type_list[$SLURM_ARRAY_TASK_ID]}'+ " --fairness_bound " + '${bounds_list[$SLURM_ARRAY_TASK_ID]}'+" --protected_feature " + '${protected_feature_list[$SLURM_ARRAY_TASK_ID]}'+" --positive_class " + str(positive_class) + " --conditional_feature " + '${condition_feature_list[$SLURM_ARRAY_TASK_ID]}'
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
