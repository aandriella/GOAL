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
    epochs = 10
    scaling_factor = 1
    # initialise the agent
    bn_model_caregiver_assistance = bnlearn.import_DAG('/home/pal/Documents/Framework/bn_generative_model/bn_agent_model/agent_assistive_model.bif')
    bn_model_caregiver_feedback = None#bnlearn.import_DAG('/home/pal/Documents/Framework/bn_generative_model/bn_agent_model/agent_feedback_model.bif')
    bn_model_user_action = bnlearn.import_DAG('/home/pal/Documents/Framework/bn_generative_model/bn_persona_model/user_action_model.bif')
    bn_model_user_react_time = bnlearn.import_DAG('/home/pal/Documents/Framework/bn_generative_model/bn_persona_model/user_react_time_model.bif')

    # initialise memory, attention and reactivity varibles
    persona_memory = 0;
    persona_attention = 0;
    persona_reactivity = 1;

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

    # attempt = [i for i in range(1, Attempt.counter.value + 1)]
    # # +1a (3,_,_) absorbing state
    # game_state = [i for i in range(0, Game_State.counter.value + 1)]
    # user_action = [i for i in range(-1, User_Action.counter.value - 1)]
    # state_space = (game_state, attempt, user_action)
    # states_space_list = list(itertools.product(*state_space))
    # agent_assistance_action = [i for i in range(Agent_Assistance.counter.value)]
    # agent_feedback_action = [i for i in range(Agent_Feedback.counter.value)]
    # action_space = (agent_assistance_action, agent_feedback_action)
    # action_space_list = list(itertools.product(*action_space))


    ##############BEFORE RUNNING THE SIMULATION UPDATE THE BELIEF IF YOU HAVE DATA####################
    log_directory = "/home/pal/carf_ws/src/carf/caregiver_in_the_loop/log/1/0"
    if os.path.exists(log_directory):
        bn_functions.update_episodes_batch(bn_model_user_action, bn_model_user_react_time, bn_model_caregiver_assistance,
                                           bn_model_caregiver_feedback, folder_filename=log_directory,
                                           with_caregiver=True)
    else:
        assert ("You're not using the user information")
        question = input("Are you sure you don't want to load user's belief information?")

    game_performance_per_episode, react_time_per_episode, agent_assistance_per_episode, agent_feedback_per_episode, episodes_list = \
        Sim.simulation(bn_model_user_action=bn_model_user_action, var_user_action_target_action=['user_action'],
                       bn_model_user_react_time=bn_model_user_react_time,
                       var_user_react_time_target_action=['user_react_time'],
                       user_memory_name="memory", user_memory_value=persona_memory,
                       user_attention_name="attention", user_attention_value=persona_attention,
                       user_reactivity_name="reactivity", user_reactivity_value=persona_reactivity,
                       task_progress_name="game_state", game_attempt_name="attempt",
                       agent_assistance_name="agent_assistance", agent_feedback_name="agent_feedback",
                       bn_model_agent_assistance=bn_model_caregiver_assistance,
                       var_agent_assistance_target_action=["agent_assistance"],
                       bn_model_agent_feedback=bn_model_caregiver_feedback, var_agent_feedback_target_action=["agent_feedback"],
                       agent_policy=None,
                       state_space=states_space_list, action_space=action_space_list,
                       epochs=epochs, task_complexity=5, max_attempt_per_object=4)


    format = "%a%b%d-%H:%M:%S %Y"
    today_id = datetime.datetime.today()
    full_path = os.getcwd() + "/results/" + str(today_id) +"/"
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    plot_game_performance_path = "BEFORE_game_performance_"+"epoch_" + str(epochs) + "_persona_memory_" + str(persona_memory) + "_persona_attention_" + str(persona_attention) + "_persona_reactivity_" + str(persona_reactivity) + ".jpg"
    plot_agent_assistance_path = "BEFORE_agent_assistance_"+"epoch_"+str(epochs)+"_persona_memory_"+str(persona_memory)+"_persona_attention_"+str(persona_attention)+"_persona_reactivity_"+str(persona_reactivity)+".jpg"
    plot_agent_feedback_path = "BEFORE_agent_feedback_"+"epoch_"+str(epochs)+"_persona_memory_"+str(persona_memory)+"_persona_attention_"+str(persona_attention)+"_persona_reactivity_"+str(persona_reactivity)+".jpg"

    utils.plot2D_game_performance(full_path +plot_game_performance_path, epochs, scaling_factor, game_performance_per_episode)
    utils.plot2D_assistance(full_path + plot_agent_assistance_path, epochs, scaling_factor, agent_assistance_per_episode)
    utils.plot2D_feedback(full_path + plot_agent_feedback_path, epochs, scaling_factor, agent_feedback_per_episode)

    world, reward, terminals = setup_mdp(initial_state=initial_state, terminal_state=terminal_state, task_length=Game_State.counter.value,
                                         n_max_attempt=Attempt.counter.value, action_space=action_space_list, state_space=states_space_list,
                                         user_action=user_action, timeout=15, episode = episodes_list)

    state_tuple_indexed = [states_space_list.index(tuple(s)) for s in (states_space_list)]

    states_space_list_string = [[str(states_space_list[j*12+i]) for i in range(12)] for j in range(3)]


    build_2dtable(states_space_list_string, 3, 12)

    # exp_V, exp_P = vi.value_iteration(world.p_transition, reward, gamma=0.9, error=1e-3, deterministic=True)
    # plt.figure(figsize=(12, 4), num="state_space")
    # sns.heatmap(np.reshape(state_tuple_indexed, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    # plt.savefig(full_path+"state_space.jpg")
    # #PLOTS EXPERT
    # plt.figure(figsize=(12, 4), num="exp_rew")
    # sns.heatmap(np.reshape(reward, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    # plt.savefig(full_path+"exp_rew.jpg")
    # plt.figure(figsize=(12, 4), num="exp_V")
    # sns.heatmap(np.reshape(exp_V, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    # plt.savefig(full_path+"exp_V.jpg")
    # plt.figure(figsize=(12, 4), num="exp_P")
    # sns.heatmap(np.reshape(exp_P, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    # plt.savefig(full_path+"exp_P.jpg")

    maxent_R = maxent(world, terminals, episodes_list)
    maxent_V, maxent_P = vi.value_iteration(world.p_transition, maxent_R, gamma=0.9, error=1e-3, deterministic=True)
    plt.figure(figsize=(12, 4), num="maxent_rew")
    sns.heatmap(np.reshape(maxent_R, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(full_path + "maxent_rew.jpg")
    plt.figure(figsize=(12, 4), num="maxent_V")
    sns.heatmap(np.reshape(maxent_V, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(full_path + "maxent_V.jpg")
    plt.figure(figsize=(12, 4), num="maxent_P")
    sns.heatmap(np.reshape(maxent_P, (4, 12)), cmap="Spectral", annot=True, cbar=False)
    plt.savefig(full_path + "maxent_P.jpg")

    #Compute entropy between two policies
    # policies = [exp_P, maxent_P]
    # entropy = get_entropy(policies, state_space_index, action_space_index)

    game_performance_per_episode, react_time_per_episode, agent_assistance_per_episode, agent_feedback_per_episode, episodes_list = \
        Sim.simulation(bn_model_user_action=bn_model_user_action, var_user_action_target_action=['user_action'],
                       bn_model_user_react_time=bn_model_user_react_time,
                       var_user_react_time_target_action=['user_react_time'],
                       user_memory_name="memory", user_memory_value=persona_memory,
                       user_attention_name="attention", user_attention_value=persona_attention,
                       user_reactivity_name="reactivity", user_reactivity_value=persona_reactivity,
                       task_progress_name="game_state", game_attempt_name="attempt",
                       agent_assistance_name="agent_assistance", agent_feedback_name="agent_feedback",
                       bn_model_agent_assistance=bn_model_caregiver_assistance,
                       var_agent_assistance_target_action=["agent_assistance"],
                       bn_model_agent_feedback=bn_model_caregiver_feedback,
                       var_agent_feedback_target_action=["agent_feedback"],
                       agent_policy=maxent_P,
                       state_space=states_space_list, action_space=action_space_list,
                       epochs=epochs, task_complexity=5, max_attempt_per_object=4)

    plot_game_performance_path = "AFTER_game_performance_" + "epoch_" + str(epochs) + "_persona_memory_" + str(
        persona_memory) + "_persona_attention_" + str(persona_attention) + "_persona_reactivity_" + str(
        persona_reactivity) + ".jpg"
    plot_agent_assistance_path = "AFTER_agent_assistance_" + "epoch_" + str(epochs) + "_persona_memory_" + str(
        persona_memory) + "_persona_attention_" + str(persona_attention) + "_persona_reactivity_" + str(
        persona_reactivity) + ".jpg"
    plot_agent_feedback_path = "AFTER_agent_feedback_" + "epoch_" + str(epochs) + "_persona_memory_" + str(
        persona_memory) + "_persona_attention_" + str(persona_attention) + "_persona_reactivity_" + str(
        persona_reactivity) + ".jpg"

    utils.plot2D_game_performance(full_path + plot_game_performance_path, epochs, scaling_factor, game_performance_per_episode)
    utils.plot2D_assistance(full_path + plot_agent_assistance_path, epochs, scaling_factor, agent_assistance_per_episode)
    utils.plot2D_feedback(full_path + plot_agent_feedback_path, epochs, scaling_factor, agent_feedback_per_episode)


if __name__ == '__main__':
    main()