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
import matplotlib.pyplot as plt
import itertools
import os
import math
import operator
import datetime
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
    features = world.state_features()#assistive_feature(trajectories)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.1))

    # actually do some inverse reinforcement learning
    reward = M.irl(world.p_transition, features, terminal, trajectories, optim, init)

    return reward


def main():
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    #################GENERATE SIMULATION################################
    # SIMULATION PARAMS
    epochs = 20
    scaling_factor = 1
    # initialise the agent
    bn_model_user_action_filename = '/home/pal/Documents/Framework/bn_generative_model/bn_persona_model/persona_model_test.bif'
    bn_model_user_action = bnlearn.import_DAG(bn_model_user_action_filename)

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
    user_action = [i for i in range(-1, User_Action.counter.value - 1)]
    state_space = (game_state, attempt, user_action)
    states_space_list = list(itertools.product(*state_space))
    state_space_index = [episode_instance.state_from_point_to_index(states_space_list, s) for s in states_space_list]
    agent_assistance_action = [i for i in range(Agent_Assistance.counter.value)]
    agent_feedback_action = [i for i in range(Agent_Feedback.counter.value)]
    action_space = (agent_feedback_action, agent_assistance_action)
    action_space_list = list(itertools.product(*action_space))
    action_space_index = [episode_instance.state_from_point_to_index(action_space_list, a) for a in action_space_list]
    terminal_state = [(Game_State.counter.value, i, user_action[j]) for i in range(1, Attempt.counter.value + 1) for j in
                      range(len(user_action))]
    initial_state = (1, 1, 0)
    agent_policy = [0 for s in state_space]

    #1. RUN THE SIMULATION WITH THE PARAMS SET BY THE CAREGIVER
    game_performance_per_episode, react_time_per_episode, agent_assistance_per_episode, agent_feedback_per_episode, episodes_list = \
    Sim.simulation(bn_model_user_action=bn_model_user_action,
                   var_user_action_target_action=['user_action'],
                   game_state_bn_name="game_state",
                   attempt_bn_name="attempt",
                   agent_assistance_bn_name="agent_assistance",
                   agent_feedback_bn_name="agent_feedback",
                   user_pref_assistance=user_pref_assistance,
                   agent_behaviour=agent_behaviour,
                   agent_policy=[],
                   state_space=states_space_list,
                   action_space=action_space_list,
                   epochs=epochs, task_complexity=5, max_attempt_per_object=4, alpha_learning=0.1)

    #2. GIVEN THE EPISODES ESTIMATE R(S) and PI(S)

    format = "%a%b%d-%H:%M:%S %Y"
    today_id = datetime.datetime.today()
    full_path = os.getcwd() + "/results/" + str(today_id) +"/"
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    plot_game_performance_path = "SIM_game_performance_"+"epoch_" + str(epochs) + ".jpg"
    plot_agent_assistance_path = "SIM_agent_assistance_"+"epoch_"+str(epochs)+".jpg"
    plot_agent_feedback_path = "SIM_agent_feedback_"+"epoch_"+str(epochs)+".jpg"

    utils.plot2D_game_performance(full_path +plot_game_performance_path, epochs, scaling_factor, game_performance_per_episode)
    utils.plot2D_assistance(full_path + plot_agent_assistance_path, epochs, scaling_factor, agent_assistance_per_episode)
    utils.plot2D_feedback(full_path + plot_agent_feedback_path, epochs, scaling_factor, agent_feedback_per_episode)

    world, reward, terminals = setup_mdp(initial_state=initial_state, terminal_state=terminal_state, task_length=Game_State.counter.value,
                                         n_max_attempt=Attempt.counter.value, action_space=action_space_list, state_space=states_space_list,
                                         user_action=user_action, timeout=15, episode = episodes_list)

    state_tuple_indexed = [states_space_list.index(tuple(s)) for s in (states_space_list)]

    #Dirty way to represent the state space in a graphical way
    states_space_list_string = [[str(states_space_list[j*12+i]) for i in range(12)] for j in range(3)]
    build_2dtable(states_space_list_string, 3, 12)

    #R(s) and pi(s) generated from the first sim
    maxent_R_sim = maxent(world, terminals, episodes_list)
    maxent_V_sim, maxent_P_sim = vi.value_iteration(world.p_transition, maxent_R_sim, gamma=0.9, error=1e-3, deterministic=False)
    plt.figure(figsize=(12, 4), num="maxent_rew")
    sns.heatmap(np.reshape(maxent_R_sim, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(full_path + "sim_maxent_R.jpg")
    plt.figure(figsize=(12, 4), num="maxent_V")
    sns.heatmap(np.reshape(maxent_V_sim, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(full_path + "sim_maxent_V.jpg")
    plt.figure(figsize=(12, 4), num="maxent_P")
    sns.heatmap(np.reshape(maxent_P_sim, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(full_path + "sim_maxent_P.jpg")
    #####################################################################################

    #3.WE GOT SOME REAL DATA UPDATE THE BELIEF OF THE BN
    log_directory = "/home/pal/Documents/Framework/bn_generative_model/bn_persona_model/cognitive_game.csv"

    if os.path.exists(log_directory):
        bn_model_user_action_from_data = Sim.build_model_from_data(csv_filename=log_directory, dag_filename=bn_model_user_action_filename, dag_model=bn_model_user_action)
    else:
        assert ("You're not using the user information")
        question = input("Are you sure you don't want to load user's belief information?")

    game_performance_per_episode, react_time_per_episode, agent_assistance_per_episode, agent_feedback_per_episode, episodes_list = \
        Sim.simulation(bn_model_user_action=bn_model_user_action,
                       var_user_action_target_action=['user_action'],
                       game_state_bn_name="game_state",
                       attempt_bn_name="attempt",
                       agent_assistance_bn_name="agent_assistance",
                       agent_feedback_bn_name="agent_feedback",
                       user_pref_assistance=user_pref_assistance,
                       agent_behaviour=agent_behaviour,
                       agent_policy = maxent_P_sim,
                       state_space=states_space_list,
                       action_space=action_space_list,
                       epochs=epochs, task_complexity=5, max_attempt_per_object=4, alpha_learning=0.1)

    plot_game_performance_path = "REAL_SIM_game_performance_" + "epoch_" + str(epochs) + ".jpg"
    plot_agent_assistance_path = "REAL_SIM_agent_assistance_" + "epoch_" + str(epochs) + ".jpg"
    plot_agent_feedback_path = "REAL_SIM_agent_feedback_" + "epoch_" + str(epochs) + ".jpg"

    utils.plot2D_game_performance(full_path + plot_game_performance_path, epochs, scaling_factor, game_performance_per_episode)
    utils.plot2D_assistance(full_path + plot_agent_assistance_path, epochs, scaling_factor, agent_assistance_per_episode)
    utils.plot2D_feedback(full_path + plot_agent_feedback_path, epochs, scaling_factor, agent_feedback_per_episode)

    # R(s) and pi(s) generated from the first sim
    maxent_R_real_sim = maxent(world, terminals, episodes_list)
    maxent_V_real_sim, maxent_P_real_sim = vi.value_iteration(world.p_transition, maxent_R_real_sim, gamma=0.9, error=1e-3,
                                                    deterministic=True)
    plt.figure(figsize=(12, 4), num="maxent_rew")
    sns.heatmap(np.reshape(maxent_R_real_sim, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(full_path + "real_sim_maxent_R.jpg")
    plt.figure(figsize=(12, 4), num="maxent_V")
    sns.heatmap(np.reshape(maxent_V_real_sim, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(full_path + "real_sim_maxent_V.jpg")
    plt.figure(figsize=(12, 4), num="maxent_P")
    sns.heatmap(np.reshape(maxent_P_real_sim, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(full_path + "real_sim_maxent_P.jpg")

    # Compute entropy between two policies
    policies = [maxent_P_sim, maxent_P_real_sim]
    s_preferences, s_constraints = get_entropy(policies, state_space_index, action_space_index)
    print("Preferences:", s_preferences, " Constraints:", s_constraints)


if __name__ == '__main__':
    main()