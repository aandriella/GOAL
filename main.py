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
import datetime
import pickle
import bnlearn

from cognitive_game_env import CognitiveGame
from episode import Episode
from environment import Environment
import src.maxent as M
import src.plot as P
import src.solver as S
import src.optimizer as O
import src.img_utils as I
import src.value_iteration as vi

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
    s_preferences = [entropy_sort[i][0] for i in range(-1, -6, -1)]
    s_constraints = [entropy_sort[i][0] for i in range(5)]

    return s_preferences, s_constraints


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
    features = world.assistive_feature(trajectories)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.1))

    # actually do some inverse reinforcement learning
    reward = M.irl(world.p_transition, features, terminal, trajectories, optim, init)

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

def merge_user_log(folder_pathname, user_id, with_feedback, column_to_remove):
    absolute_path = folder_pathname+"/"+str(+user_id)+"/"+str(with_feedback)
    df = pd.DataFrame()
    if len(absolute_path)==0:
        print("Error no folders in path ", absolute_path)
    # else:
    #     df = pd.read_csv(absolute_path+"/1/bn_variables.csv")

    if column_to_remove!=None:
        df = df.drop(column_to_remove, axis=1)
    #df_removed = df.drop(["user_memory", "user_reactivity"], axis=1)
    sessions_directory = os.listdir(absolute_path)
    episode_length = [0]*(len(sessions_directory)+1)

    for i in range(len(sessions_directory)):
        file_folder = absolute_path+"/"+sessions_directory[i]
        print("File folder: ", file_folder)
        files = os.listdir(file_folder)

        for k in range(len(files)):
            if files[k] == "bn_variables.csv":
                df_ = pd.read_csv(file_folder+"/"+files[k])
                df = df.append(df_)
                episode_length[i+1] = episode_length[i]+(df_.shape[0]-1)+1
    df.to_csv(absolute_path + "/summary_bn_variables.csv", index=False)
    return df, episode_length

def compute_agent_policy(folder_pathname, user_id, with_feedback, state_space, action_space, episode_length):
    #read columns of interest (game_state, attempt, user_prev_action)
    ep = Episode()
    df = pd.read_csv(folder_pathname+"/"+str(user_id)+"/"+str(with_feedback)+"/summary_bn_variables.csv")
    agent_policy_counter = [[0 for  a in action_space] for s in state_space]
    agent_policy_prob = [[0 for  a in action_space] for s in state_space]
    row_t_0 = 0
    for index, row in df.iterrows():
        if index == 0 or index in episode_length:
            state_point = (row['game_state'], row['attempt'], 0)
            state_index = ep.state_from_point_to_index(state_space, state_point)
            action_point = (row['agent_assistance'])
            action_index = action_point
            agent_policy_counter[state_index][action_index] += 1
            row_t_0 = row['user_action']
        else:
            state_point = (row['game_state'], row['attempt'], row_t_0)
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
    # SIMULATION PARAMS
    epochs = 10
    scaling_factor = 1
    # initialise the agent
    bn_model_user_action_filename = '/home/pal/Documents/Framework/bn_generative_model/bn_persona_model/persona_model_test.bif'
    bn_model_agent_behaviour_filename = '/home/pal/Documents/Framework/bn_generative_model/bn_agent_model/agent_assistive_model.bif'
    learned_policy_filename = ""
    bn_model_user_action = bnlearn.import_DAG(bn_model_user_action_filename)
    bn_model_agent_behaviour = bnlearn.import_DAG(bn_model_agent_behaviour_filename)

    #setup by the caregiver
    user_pref_assistance = 2
    agent_behaviour = "challenge"

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
    agent_feedback_action = [i for i in range(Agent_Feedback.counter.value)]
    action_space = (agent_assistance_action)
    action_space_list = action_space#list(itertools.product(*action_space))
    action_space_index = action_space_list#[episode_instance.state_from_point_to_index(action_space_list, a) for a in action_space_list]
    terminal_state = [(Game_State.counter.value, i, user_action[j]) for i in range(1, Attempt.counter.value + 1) for j in
                      range(len(user_action))]
    initial_state = (1, 1, 0)
    agent_policy = [0 for s in state_space]



    #####################INPUT AND OUTPUT FOLDER ####################################
    input_folder_data = "/home/pal/Documents/Framework/GenerativeMutualShapingRL/data"
    user_id = 1
    with_feedback = True

    output_folder_data = os.getcwd() + "/results/" + str(user_id)
    if not os.path.exists(output_folder_data):
        os.mkdir(output_folder_data)
        if not os.path.exists(output_folder_data+"/"+str(with_feedback)):
            os.mkdir(output_folder_data+"/"+with_feedback)

    #1. CREATE INITIAL USER COGNITIVE MODEL FROM DATA
    df_from_data, episode_length = merge_user_log(folder_pathname=input_folder_data,
                                        user_id=user_id, with_feedback=with_feedback, column_to_remove=None)
    #2. CREATE POLICY FROM DATA
    agent_policy_from_data = compute_agent_policy(folder_pathname=input_folder_data,
                         user_id=user_id, with_feedback=with_feedback, state_space=states_space_list,
                         action_space=action_space_list, episode_length=episode_length)

    # 3. RUN THE SIMULATION
    log_directory = input_folder_data+"/"+str(user_id)+"/"+str(with_feedback)
    bn_model_user_action_from_data_and_therapist = None
    bn_model_agent_behaviour_from_data_and_therapist = None
    if os.path.exists(log_directory):
        bn_model_user_action_from_data_and_therapist = Sim.build_model_from_data(csv_filename=log_directory+"/summary_bn_variables.csv", dag_filename=bn_model_user_action_filename, dag_model=bn_model_user_action)
        bn_model_agent_behaviour_from_data_and_therapist = Sim.build_model_from_data(csv_filename=log_directory+"/summary_bn_variables.csv", dag_filename=bn_model_agent_behaviour_filename, dag_model=bn_model_agent_behaviour)
    else:
        assert ("You're not using the user information")
        question = input("Are you sure you don't want to load user's belief information?")

    diff = dict([(s, [0] * len(action_space_index)) for s in state_space_index])
    entropy = dict([(s, 0) for s in state_space_index])
    N = 5

    for i in range(N):
        game_performance_per_episode, react_time_per_episode, agent_assistance_per_episode, agent_feedback_per_episode, episodes_list = \
            Sim.simulation(bn_model_user_action=bn_model_user_action_from_data_and_therapist,
                           bn_model_agent_behaviour = bn_model_agent_behaviour_from_data_and_therapist,
                           var_user_action_target_action=['user_action'],
                           var_agent_behaviour_target_action=['agent_assistance'],
                           game_state_bn_name="game_state",
                           attempt_bn_name="attempt",
                           agent_assistance_bn_name="agent_assistance",
                           agent_feedback_bn_name="agent_feedback",
                           user_pref_assistance=user_pref_assistance,
                           agent_behaviour=agent_behaviour,
                           agent_policy = agent_policy_from_data,
                           state_space=states_space_list,
                           action_space=action_space_list,
                           epochs=epochs, task_complexity=5, max_attempt_per_object=4, alpha_learning=0.1)

        plot_game_performance_path = output_folder_data+"/REAL_SIM_game_performance_" + "epoch_" + str(epochs) + ".jpg"
        plot_agent_assistance_path = output_folder_data+"/REAL_SIM_agent_assistance_" + "epoch_" + str(epochs) + ".jpg"
        plot_agent_feedback_path = output_folder_data+"/REAL_SIM_agent_feedback_" + "epoch_" + str(epochs) + ".jpg"

        utils.plot2D_game_performance(plot_game_performance_path, epochs, scaling_factor, game_performance_per_episode)
        utils.plot2D_assistance(plot_agent_assistance_path, epochs, scaling_factor, agent_assistance_per_episode)
        utils.plot2D_feedback(plot_agent_feedback_path, epochs, scaling_factor, agent_feedback_per_episode)

        cognitive_game_world, reward, terminals = setup_mdp(initial_state=initial_state, terminal_state=terminal_state,
                                                            task_length=Game_State.counter.value, n_max_attempt=Attempt.counter.value,
                                                            action_space=action_space_list, state_space=states_space_list,
                                                            user_action=user_action, timeout=15, episode=episodes_list)

        state_tuple_indexed = [states_space_list.index(tuple(s)) for s in (states_space_list)]
        states_space_list_string = [[str(states_space_list[j*12+i]) for i in range(12)] for j in range(4)]
        build_2dtable(states_space_list_string, 4, 12)

        # R(s) and pi(s) generated from the first sim
        maxent_R_real_sim = maxent(world=cognitive_game_world, terminal=terminals, trajectories=episodes_list)
        maxent_V_real_sim, maxent_P_real_sim = vi.value_iteration(cognitive_game_world.p_transition, maxent_R_real_sim, gamma=0.9, error=1e-3,
        deterministic=True)

        learned_policy_filename = output_folder_data + "/" + "learned_policy.pkl"
        with open(learned_policy_filename, 'wb') as f:
            pickle.dump(maxent_P_real_sim, f, protocol=2)


        for s in state_space_index:
            index = maxent_P_real_sim[s]
            diff[s][index] += 1.0 / N
        sns.heatmap(np.reshape(maxent_P_real_sim, (4, 12)), cmap="Spectral", annot=True, cbar=False)
        plt.savefig(output_folder_data + "maxent_P_iter_"+str(i)+".jpg")

    for s in state_space_index:
        E = 0
        for i in range(len(action_space_index)):
            if diff[s][i] > 0:
                E -= diff[s][i] * math.log(diff[s][i])
        entropy[s] = E

    # highlight high and low entropy states
    entropy_sort = sorted(entropy.items(), key=operator.itemgetter(1))
    s_preferences = [episode_instance.state_from_index_to_point(states_space_list, entropy_sort[i][0]) for i in range(-1, -6, -1)]
    s_constraints = [episode_instance.state_from_index_to_point(states_space_list, entropy_sort[i][0]) for i in range(27)][22:]

    print("S_preferences: ", s_preferences)
    print("S_constrains: ", s_constraints)


    plt.figure(figsize=(12, 4), num="maxent_rew")
    sns.heatmap(np.reshape(maxent_R_real_sim, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(output_folder_data + "real_sim_maxent_R.jpg")
    plt.figure(figsize=(12, 4), num="maxent_V")
    sns.heatmap(np.reshape(maxent_V_real_sim, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(output_folder_data + "real_sim_maxent_V.jpg")
    plt.figure(figsize=(12, 4), num="maxent_P")
    sns.heatmap(np.reshape(maxent_P_real_sim, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(output_folder_data + "real_sim_maxent_P.jpg")




if __name__ == '__main__':
    main()