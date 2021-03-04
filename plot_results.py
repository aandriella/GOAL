import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statannot import add_stat_annotation
#from pandas.compat import StringIO
import pandas as pd
import csv
import os

def compute_assistance(directory_name, output_filename):

    def compute_average_assistance(file_name):
        print("FILENAME:", file_name)
        data = pd.read_csv(file_name)
        attempt = (data['attempt'])
        if attempt.empty == True:
            return 0
        sum=0
        print("attempt vector ", attempt)
        for i in range(len(attempt)):
           print("id:", i, "val:", attempt[i])
           sum += attempt[i]
        return sum/len(attempt)

    #create a file for outputting our data
    with open(output_filename, 'a+') as f:
        f.write(output_filename)
        f.write("\n")
        f.close()
    #get into directory e.g. terapist-patient-interaction
    directory = directory_name
    files_in_directory = os.listdir(directory_name)
    #for each user id
    for dir_name in files_in_directory:
        files_in_directory = os.listdir(directory_name+"/"+dir_name)
        for dir_dir_name in files_in_directory:
            files_files_in_directory = os.listdir(directory_name+"/"+dir_name+"/"+dir_dir_name)
            if "1" in files_files_in_directory or \
             "2" in files_files_in_directory  or \
             "3" in files_files_in_directory:
                for dir_dir_dir_name in files_files_in_directory:
                    files_files_files_in_directory = os.listdir(directory_name+"/"+dir_name+"/"+dir_dir_name+"/"+dir_dir_dir_name)

                    for dir_dir_dir_dir_name in files_files_files_in_directory:
                        if dir_dir_dir_dir_name == "log_spec.csv":
                            print(dir_dir_dir_dir_name)
                            # process
                            attempt = compute_average_assistance(directory_name+"/"+dir_name+"/"+dir_dir_name+"/"+dir_dir_dir_name+"/"+dir_dir_dir_dir_name)
                            print(directory_name+"/"+dir_name+"/"+dir_dir_name+"/"+dir_dir_dir_name+"/"+dir_dir_dir_dir_name)
                            with open(output_filename, 'a+') as f:
                                f.write(str(attempt))
                                f.write("\n")
                                f.close()

def compute_lev_assistance(directory_name, output_filename):

    def compute_assistance_vector(file_name):
        print("FILENAME:", file_name)
        data = pd.read_csv(file_name)
        assistance_levs = (data['agent_assistance'])
        if assistance_levs.empty == True:
            return 0
        counter_levels=[0,0,0,0,0]
        print("agent_assistance vector ", assistance_levs)
        for i in range(len(assistance_levs)):
            print("id:", i, "val:", assistance_levs[i])
            if assistance_levs[i] == 0:
               counter_levels[0] += 1
            elif assistance_levs[i] == 1:
               counter_levels[1] += 1
            elif assistance_levs[i] == 2:
                counter_levels[2] += 1
            elif assistance_levs[i] == 3:
                counter_levels[3] += 1
            else:
                counter_levels[4] += 1

        return counter_levels

    #create a file for outputting our data
    with open(output_filename, 'a+') as f:
        f.write(output_filename)
        f.write("\n")
        f.close()
    #get into directory e.g. terapist-patient-interaction
    directory = directory_name
    files_in_directory = os.listdir(directory_name)
    #for each user id
    for dir_name in files_in_directory:
        files_in_directory = os.listdir(directory_name+"/"+dir_name)
        for dir_dir_name in files_in_directory:
            files_files_in_directory = os.listdir(directory_name+"/"+dir_name+"/"+dir_dir_name)
            if "1" in files_files_in_directory or \
             "2" in files_files_in_directory  or \
             "3" in files_files_in_directory:
                for dir_dir_dir_name in files_files_in_directory:
                    files_files_files_in_directory = os.listdir(directory_name+"/"+dir_name+"/"+dir_dir_name+"/"+dir_dir_dir_name)

                    for dir_dir_dir_dir_name in files_files_files_in_directory:
                        if dir_dir_dir_dir_name == "bn_variables.csv":
                            print(dir_dir_dir_dir_name)
                            # process
                            ass_level = compute_assistance_vector(directory_name+"/"+dir_name+"/"+dir_dir_name+"/"+dir_dir_dir_name+"/"+dir_dir_dir_dir_name)
                            print(directory_name+"/"+dir_name+"/"+dir_dir_name+"/"+dir_dir_dir_name+"/"+dir_dir_dir_dir_name)
                            with open(output_filename, 'a+') as f:
                                f.write(str(ass_level[0])+","+str(ass_level[1])+","+str(ass_level[2])+","+
                                            str(ass_level[3])+","+str(ass_level[4]))
                                f.write("\n")
                                f.close()


def compute_boxplot(x_val, y_val, order, box_pairs, p_values, input_filename, output_filename):

    df = pd.read_csv(input_filename)
    x = x_val
    y = y_val
    order = order
    ax = sns.boxplot(data=df, x=x, y=y, order=order)
    add_stat_annotation(ax, data=df, x=x, y=y, order=order,
                        box_pairs=box_pairs,
                        perform_stat_test=False, pvalues=p_values,
                        test=None, text_format='star', loc='inside', verbose=2);
    plt.xlabel("Therapist")
    plt.ylabel("STD_DEV")
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')

    plt.show()
#plot therapist evaluation
# x = "session"
# y = "eval"
# order = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
# box_pairs = [('S1','S2'), ('S2', 'S3'),('S3','S4'),('S4', 'S5'),('S5','S6')]
# p_values = [0.001, 0.01, 0.001, 0.001, 0.07]
#
# input_filename = "../therapist_evaluation.csv"
# output_filename = "Pictures/therapist_evaluation.png"
# compute_boxplot(x_val=x, y_val=y, order=order, box_pairs=box_pairs, p_values=p_values, input_filename=input_filename, output_filename=output_filename)

#plot NASA TLX therapist vs robot

# x = "Task Workload"
# y = "Value"
# hue = "assistant"
# hue_order=['Human Therapist', 'Robot Therapist', '']
# box_pairs=[
#     (("mental", "Human Therapist"), ("mental", "Robot Therapist")),
#     (("physical", 'Human Therapist'), ("physical", "Robot Therapist")),
#     (("temporal", 'Human Therapist'), ("temporal", "Robot Therapist")),
#     (("performance", 'Human Therapist'), ("performance", "Robot Therapist")),
#     (("effort", 'Human Therapist'), ("effort", "Robot Therapist")),
#     (("frustration", 'Human Therapist'), ("frustration", "Robot Therapist")),
#     (("total", 'Human Therapist'), ("total", "Robot Therapist")),
#     ]
# input_filename = "statistical_analysis/nasa_r_vs_t.csv"
# df = pd.read_csv(input_filename)
# ax = sns.boxplot(data=df, x=x, y=y, hue=hue,)
# add_stat_annotation(ax, data=df, x=x, y=y, hue=hue, box_pairs=box_pairs,
#                     perform_stat_test=False, pvalues=[0.004, 0.62, 0.5, 0.41, 0.005, 0.32, 0.03],
#                     test=None, loc='inside', verbose=2)
# plt.xticks(rotation=45)
# plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
# plt.savefig('Pictures/nasa_r_vs_t.png', dpi=600, bbox_inches='tight')

# x = "assistant"
# y = "performance"
# order = ['Human', 'GOAL Simulator', 'Robot']
# box_pairs = [('Human','Robot'), ('GOAL Simulator', 'Robot')]
# p_values = [0.02, 0.1]
#
# input_filename = "statistical_analysis/tpi_vs_rpi_sim.csv"
# output_filename = "Pictures/tpi_rpi_sim_performance.png"
# compute_boxplot(x_val=x, y_val=y, order=order, box_pairs=box_pairs, p_values=p_values, input_filename=input_filename, output_filename=output_filename)

x = "assistant"
y = "std_dev"
order = ['Human', 'Robot']
box_pairs = [('Human','Robot')]
p_values = [0.001]

input_filename = "statistical_analysis/rpi_tpi_std_dev.csv"
output_filename = "Pictures/tpi_rpi_std_dev.png"
compute_boxplot(x_val=x, y_val=y, order=order, box_pairs=box_pairs, p_values=p_values, input_filename=input_filename, output_filename=output_filename)
#
# sum_attempt = compute_lev_assistance("/home/pal/carf_ws/src/carf/robot-patient-interaction",
#                                  #"/home/pal/Documents/Framework/GenerativeMutualShapingRL/therapist-patient-interaction",
#                                  "/home/pal/Documents/Framework/GenerativeMutualShapingRL/statistical_analysis/assistance_levels_robot.txt")
# print("attempt:",sum_attempt)

#sns.set_theme(style="whitegrid")

# input_filename = "statistical_analysis/average_assistance.csv"
# df = pd.read_csv(input_filename)
# ax = sns.barplot(y="level", x="percentage", hue="Therapist", data=df, orient='h')
# plt.xlabel("Percentage")
# plt.ylabel("Assistance Level")
# plt.savefig('Pictures/avg_assistance.png', dpi=600, bbox_inches='tight')
