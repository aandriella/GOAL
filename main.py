#read the log from the caregiver_in_the_loop or robot_in_the_loop
#update the initial beliefs of the Bayesian Network set by the therapist
#return the updated model for user_action, react_time, robot_assistance and robot_feedback
#run a simulation and generate episodes

#compute R(s) and pi according to the episodes
#return pi -> given a state provide an action
#store policy, id

#######In the case of robot ###############
#run entropy check to show the caregiver the states with high entropy (different from the initial model)
#ask him if he wants to make changes to the initial model
#show how the policies rehape during each iteraction

import sys
import random
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
import math
import operator
import pickle
import bnlearn
import argparse

from episode import Episode
from cognitive_game_env import CognitiveGame
from environment import Environment
import maxent as M
import plot as P
import solver as S
import optimizer as O
import img_utils as I
import value_iteration as vi

import simulation as Sim
import bn_functions as bn_functions
from bn_variables import Agent_Assistance, Agent_Feedback, Attempt, User_Action, User_React_time, Game_State
import utils as utils


def build_2dtable(table_vals, rows, cols):
    plt.figure()
    ax = plt.gca()
    # plt.plot([10,10,14,14,10],[2,4,4,2,2],'r')
    col_labels = ["att_user" for i in range(cols)]
    row_labels = ["g_s" for i in range(rows)]
    # the rectangle is where I want to place the table
    the_table = plt.table(cellText=table_vals,
                          colWidths=[0.08] * cols,
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          loc='center')
    plt.text(12, 3.4, 'Table Title', size=8)
    plt.show()


def setup_mdp(initial_state, terminal_state, task_length,
              n_max_attempt, action_space, state_space,
              user_action, timeout, episode):
    """This function initialise the mdp
    Args:
        :initial_state: initial state of the mdp
        :final_states: final states of the mdp
        :n_solution: n of tokens to place in the correct locations
        :n_token: total number of tokens
        :n_attempt: max attempts for token
        :n_user_action: -2: timeout, -1 max_attempt, 0, wrong move, 1 correct move
        :timeout: max time for user to move a token
        :trans_filename: the transition probabilities generated with the simulator
    Return:
        :word: reference to the mdp
        :rewards
        :terminals vector (index)
    """
    world = CognitiveGame(initial_state, terminal_state, task_length,
                          n_max_attempt, action_space, state_space,
                          user_action, timeout, episode)

    terminals = list()
    # set up the reward function
    reward = np.zeros(world.n_states)
    for idx, final_state in enumerate(terminal_state):
        index_terminal_state = world.episode_instance.state_from_point_to_index(state_space, final_state)
        reward[index_terminal_state] = 1
        terminals.append(index_terminal_state)

    return world, reward, terminals


def get_entropy(policies, state_space, action_space):
    diff = dict([(s, [0] * len(action_space)) for s in state_space])
    entropy = dict([(s, 0) for s in state_space])
    for pi in policies:
        for s in state_space:
            index = (pi[s])
            diff[s][index] += 1.0 / len(policies)

    for s in state_space:
        E = 0
        for i in range(len(action_space)):
            if diff[s][i] > 0:
                E -= diff[s][i] * math.log(diff[s][i])
        entropy[s] = E

    # highlight high and low entropy states
    entropy_sort = sorted(entropy.items(), key=operator.itemgetter(1))
    s_constraints = [entropy_sort[i][0] for i in range(-1, -6, -1)]
    s_preferences = [entropy_sort[i][0] for i in range(5)]

    return  s_constraints,  s_preferences


def maxent(world, terminal, trajectories):
    """
    Maximum Entropy Inverse Reinforcement Learning
     Args:
        :word: reference to the mdp
        :terminal: the terminal vector
        :trajectories: The trajectories generated with the simulator
    Return:
        estimation of the reward based on the MEIRL
    """
    # set up features: we use one feature vector per state
    #features = world.state_features()
    features = world.assistive_feature(trajectories)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.1))

    # actually do some inverse reinforcement learning
    reward = M.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount=0.1)
    #reward = M.irl(world.p_transition, features, terminal, trajectories, optim, init)

    return reward

def build_policy_from_therapist(bn_model_path, action_space, n_game_state, n_attempt):
    DAG = bnlearn.import_DAG(bn_model_path)
    therapist_policy = [[[ 0 for k in range(len(action_space))] for j in range(n_attempt)] for i in range(n_game_state)]
    for g in range(n_game_state):
        for a in range(n_attempt):
            q_origin = bnlearn.inference.fit(DAG, variables=['agent_assistance'], evidence={
                    'game_state': g,
                    'attempt': a})
            therapist_policy[g][a] = q_origin.values.tolist()
    return therapist_policy

def merge_agent_policy(policy_from_data, policy_from_therapist):
    merged_policy = policy_from_therapist[:]
    for index in range(len(policy_from_therapist)):
        merged_policy[index]
        merged_policy[index] = list(map(lambda x:sum(x), ))

def merge_user_log(tpi_folder_pathname, file_output, user_id, with_feedback, rpi_folder_pathname=None,  column_to_remove=None):
    tpi_absolute_path = tpi_folder_pathname+"/"+str(+user_id)+"/"+str(with_feedback)
    if rpi_folder_pathname!=None:
        rpi_absolute_path = rpi_folder_pathname+"/"+str(+user_id)+"/"+str(with_feedback)
    else:
        rpi_absolute_path=None

    episodes_length, tpi_episode_length, rpi_episode_length = [], [], []

    df = pd.DataFrame()
    if len(tpi_absolute_path)==0:
        print("Error no folders in path ", tpi_absolute_path)

    if column_to_remove!=None:
        df = df.drop(column_to_remove, axis=1)
    #df_removed = df.drop(["user_memory", "user_reactivity"], axis=1)
    tpi_sessions_directory = os.listdir(tpi_absolute_path)
    tpi_episode_length = [0]*(len(tpi_sessions_directory)+1)

    files = None
    for i in range(len(tpi_sessions_directory)):
        file_folder = tpi_absolute_path+"/"+tpi_sessions_directory[i]
        print("File folder: ", file_folder)
        files = os.listdir(file_folder)

        for k in range(len(files)):
            if files[k] == "bn_variables.csv":
                df_ = pd.read_csv(file_folder+"/"+files[k])
                df = df.append(df_)
                tpi_episode_length[i+1] = tpi_episode_length[i]+(df_.shape[0]-1)+1

    if rpi_folder_pathname!=None and len(rpi_absolute_path)!=None:
        rpi_sessions_directory = os.listdir(rpi_absolute_path)
        rpi_episode_length = [0] * len(rpi_sessions_directory)

        files = None
        for i in range(len(rpi_sessions_directory)):
            file_folder = rpi_absolute_path + "/" + rpi_sessions_directory[i]
            print("File folder: ", file_folder)
            files = os.listdir(file_folder)

            for k in range(len(files)):
                if files[k] == "bn_variables.csv":
                    df_ = pd.read_csv(file_folder + "/" + files[k])
                    df = df.append(df_)
                    rpi_episode_length[i]  = (sum(rpi_episode_length)+ (df_.shape[0] - 1) + 1)
        rpi_episode_length = [rpi_episode_length[i]+tpi_episode_length[-1] for i in range(len(rpi_episode_length))]
    else:
        print("You are not considering the data collected from the interaction with the robot")


    df.to_csv(file_output, index=False)
    episodes_length = tpi_episode_length+rpi_episode_length

    return df, (episodes_length)

def compute_agent_policy(training_set_filename, state_space, action_space, episode_length):
    #read columns of interest (game_state, attempt, user_prev_action)
    ep = Episode()
    df = pd.read_csv(training_set_filename)
    agent_policy_counter = [[0 for  a in action_space] for s in state_space]
    agent_policy_prob = [[0 for  a in action_space] for s in state_space]
    row_t_0 = 0
    for index, row in df.iterrows():
        if index == 0 or index in episode_length:
            state_point = (row['game_state'], row['attempt']+1, 0)
            state_index = ep.state_from_point_to_index(state_space, state_point)
            action_point = (row['agent_assistance'])
            action_index = action_point
            agent_policy_counter[state_index][action_index] += 1
            row_t_0 = row['user_action']
        else:
            state_point = (row['game_state'], row['attempt']+1, row_t_0)
            state_index = ep.state_from_point_to_index(state_space, state_point)
            action_point = (row['agent_assistance'])
            action_index = action_point
            agent_policy_counter[state_index][action_index] += 1
            row_t_0 = row['user_action']
    for s in range(len(state_space)):
        agent_policy_prob[s] = list(map(lambda x:x/(sum(agent_policy_counter[s])+0.001), agent_policy_counter[s]))

    return agent_policy_prob

def main():

    #################GENERATE SIMULATION################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--bn_model_folder', '--bn_model_folder', type=str,help="folder in which all the user and the agent models are stored ",
                        default="/home/pal/Documents/Framework/GenerativeMutualShapingRL/BN_Models")
    parser.add_argument('--bn_agent_model_filename', '--bn_agent_model', type=str,help="file path of the agent bn model",
                        default="/home/pal/Documents/Framework/bn_generative_model/bn_agent_model/agent_test.bif")
    parser.add_argument('--epoch', '--epoch', type=int,help="number of epochs in the simulation", default=200)
    parser.add_argument('--run', '--run', type=int, help="number of runs in the simulation", default=50)
    parser.add_argument('--output_policy_filename', '--p', type=str,help="output policy from the simulation",
                        default="policy.pkl")
    parser.add_argument('--output_reward_filename', '--r', type=str, help="output reward from the simulation",
                        default="reward.pkl")
    parser.add_argument('--output_value_filename', '--v', type=str, help="output value function from the simulation",
                        default="value_function.pkl")
    parser.add_argument('--therapist_patient_interaction_folder', '--tpi_path', type=str,help="therapist-patient interaction folder",
                        default="/home/pal/carf_ws/src/carf/caregiver_in_the_loop/log")
    parser.add_argument('--agent_patient_interaction_folder', '--api_path', type=str,help="agent-patient interaction folder",
                        default="/home/pal/carf_ws/src/carf/robot_in_the_loop/log")
    parser.add_argument('--user_id', '--id', type=int,help="user id", required=True)
    parser.add_argument('--with_feedback', '--f', type=eval, choices=[True, False], help="offering sociable", required=True)
    parser.add_argument('--session', '--s', type=int, help="session of the agent-human interaction", required=True)

    args = parser.parse_args()


    # READ PARAMS FROM COMMAND LINE
    user_id = args.user_id
    with_feedback = args.with_feedback
    session = args.session
    epochs = args.epoch
    runs = args.run
    # initialise the agent
    bn_user_model_filename = args.bn_model_folder  +"/"+str(user_id)+"/"+str(with_feedback)+"/user_model.bif"
    bn_agent_model_filename = args.bn_model_folder+"/"+str(user_id)+"/"+str(with_feedback)+"/agent_model.bif"
    learned_policy_filename = args.output_policy_filename
    learned_reward_filename = args.output_reward_filename
    learned_value_f_filename = args.output_value_filename
    therapist_patient_interaction_folder = args.therapist_patient_interaction_folder
    agent_patient_interaction_folder = args.agent_patient_interaction_folder
    scaling_factor = 1

    #import user and agent model
    bn_user_model_action = bnlearn.import_DAG(bn_user_model_filename)
    bn_agent_model_behaviour = bnlearn.import_DAG(bn_agent_model_filename)

    #setup by the caregiver

    # define state space struct for the irl algorithm
    episode_instance = Episode()
    # DEFINITION OF THE MDP
    # define state space struct for the irl algorithm
    attempt = [i for i in range(1, Attempt.counter.value + 1)]
    # +1 (3,_,_) absorbing state
    game_state = [i for i in range(0, Game_State.counter.value + 1)]
    user_action = [i for i in range(0, User_Action.counter.value)]
    state_space = (game_state, attempt, user_action)
    states_space_list = list(itertools.product(*state_space))
    state_space_index = [episode_instance.state_from_point_to_index(states_space_list, s) for s in states_space_list]
    agent_assistance_action = [i for i in range(Agent_Assistance.counter.value)]
    action_space = (agent_assistance_action)
    action_space_list = action_space
    terminal_state = [(Game_State.counter.value, i, user_action[j]) for i in range(1, Attempt.counter.value + 1) for j in
                      range(len(user_action))]
    initial_state = (1, 1, 0)


    #output folders
    output_folder_data_path = os.getcwd() + "/results/" + str(user_id) +"/"+str(with_feedback)+"/"+str(session)
    if not os.path.exists(os.getcwd() + "/results"+"/"+str(user_id)):
        os.mkdir(os.getcwd() + "/results"+"/"+str(user_id))
    if not os.path.exists(os.getcwd() + "/results"+"/"+str(user_id) +"/"+str(with_feedback)):
        os.mkdir(os.getcwd() + "/results" + "/" +str(user_id) +"/"+str(with_feedback))
    if not os.path.exists(output_folder_data_path):
        os.mkdir(output_folder_data_path)


#1. CREATE INITIAL USER COGNITIVE MODEL FROM DATA
    df_from_data, episode_length = merge_user_log(tpi_folder_pathname=therapist_patient_interaction_folder,
                                                  file_output=output_folder_data_path+"/summary_bn_variables_from_data.csv",
                                                  user_id=user_id,
                                                  with_feedback=with_feedback,
                                                  rpi_folder_pathname=None,#agent_patient_interaction_folder,
                                                column_to_remove=None)

    #2. CREATE POLICY FROM DATA
    agent_policy_from_data = compute_agent_policy(training_set_filename=output_folder_data_path+"/summary_bn_variables_from_data.csv",
                         state_space=states_space_list,
                         action_space=action_space_list, episode_length=episode_length)
    det_agent_policy_from_data = list(map(lambda x:np.argmax(x), agent_policy_from_data))
    sns.heatmap(np.reshape(det_agent_policy_from_data, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(output_folder_data_path + "/pi_from_data"+str(user_id)+"_"+str(with_feedback)+"_"+str(session)+".jpg")


    # 3. RUN THE SIMULATION
    bn_model_user_action_from_data_and_therapist = None
    bn_model_agent_behaviour_from_data_and_therapist = None



    if os.path.exists(output_folder_data_path):
        bn_model_user_action_from_data_and_therapist = Sim.build_model_from_data(
            csv_filename=output_folder_data_path + "/summary_bn_variables_from_data.csv", dag_filename=bn_user_model_filename,
            dag_model=bn_user_model_action)
        bn_model_agent_behaviour_from_data_and_therapist = Sim.build_model_from_data(
            csv_filename=output_folder_data_path + "/summary_bn_variables_from_data.csv", dag_filename=bn_agent_model_filename,
            dag_model=bn_agent_model_behaviour)
    else:
        assert ("You're not using the user information")
        question = input("Are you sure you don't want to load user's belief information?")

    game_performance_per_episode, agent_assistance_per_episode, episodes = \
        Sim.simulation(bn_model_user_action=bn_model_user_action_from_data_and_therapist,
                       bn_model_agent_behaviour = bn_model_agent_behaviour_from_data_and_therapist,
                       var_user_action_target_action=['user_action'],
                       var_agent_behaviour_target_action=['agent_assistance'],
                       game_state_bn_name="game_state",
                       attempt_bn_name="attempt",
                       agent_assistance_bn_name="agent_assistance",
                       agent_policy = [],
                       state_space=states_space_list,
                       action_space=action_space_list,
                       epoch=epochs,
                       run=runs,
                       task_complexity=5,
                       max_attempt_per_object=4,
                       alpha_learning=0.1)

    plot_game_performance_path = output_folder_data_path+"/game_performance_" + "epoch_" + str(epochs) + ".jpg"
    plot_agent_assistance_path = output_folder_data_path+"/agent_assistance_" + "epoch_" + str(epochs) + ".jpg"

    utils.plot2D_game_performance(plot_game_performance_path, epochs, scaling_factor, game_performance_per_episode)
    utils.plot2D_assistance(plot_agent_assistance_path, epochs, scaling_factor, agent_assistance_per_episode)

    # add episodes from different policies
    # for e in range(len(episodes)):
    #     episodes_from_different_policies.append(Episode(episodes[e]._t))

    cognitive_game_world, reward, terminals = setup_mdp(initial_state=initial_state, terminal_state=terminal_state,
                                                        task_length=Game_State.counter.value, n_max_attempt=Attempt.counter.value,
                                                        action_space=action_space_list, state_space=states_space_list,
                                                        user_action=user_action, timeout=15, episode=episodes)

    # state_tuple_indexed = [states_space_list.index(tuple(s)) for s in (states_space_list)]
    # states_space_list_string = [[str(states_space_list[j*12+i]) for i in range(12)] for j in range(4)]
    # build_2dtable(states_space_list_string, 4, 12)



    # R(s) and pi(s) generated from the first sim
    maxent_R = maxent(world=cognitive_game_world, terminal=terminals, trajectories=episodes)
    maxent_V, maxent_P = vi.value_iteration(cognitive_game_world.p_transition, maxent_R, gamma=0.99, error=1e-2,
                                                              deterministic=False)
    print(maxent_P)
    with open(output_folder_data_path+"/"+learned_policy_filename, 'wb') as f:
        pickle.dump(maxent_P, f, protocol=2)
    with open(output_folder_data_path+"/"+learned_reward_filename, 'wb') as f:
        pickle.dump(maxent_R, f, protocol=2)
    with open(output_folder_data_path+"/"+learned_value_f_filename, 'wb') as f:
        pickle.dump(maxent_V, f, protocol=2)

    # if n>0:
    #
    #     s_constraints, s_preferences = get_entropy([policies_from_sim[-1], det_agent_policy_from_data], state_space_index, action_space_index)
    #     print("S_preferences: ", s_preferences)
    #     print("S_constrains: ", s_constraints)
    #     for state_index in s_constraints:
    #         action = np.argmax(agent_policy_from_data[state_index])
    #         for action_index in range(len(maxent_P_real_sim[state_index])):
    #             if action == action_index:
    #                 maxent_P_real_sim[state_index][action_index] = (0.9)
    #             else:
    #                 maxent_P_real_sim[state_index][action_index] = 0.02
    #         maxent_P_real_sim[state_index] = list(map(lambda x:x/sum(maxent_P_real_sim[state_index]), maxent_P_real_sim[state_index]))

    sns.heatmap(np.reshape(maxent_R, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(output_folder_data_path + "/maxent_R.jpg")
    plt.show()
    sns.heatmap(np.reshape(maxent_V, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(output_folder_data_path + "/maxent_V.jpg")
    plt.show()
    maxent_P_det = list(map(lambda x: np.argmax(x), maxent_P))
    sns.heatmap(np.reshape(maxent_P_det, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(output_folder_data_path + "/maxent_P.jpg")
    plt.show()



if __name__ == '__main__':
    main()