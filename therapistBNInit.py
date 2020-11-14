import random
import pandas as pd
import os
import argparse
from bn_variables import User_Action, Game_State, Attempt, Agent_Assistance


class Therapist_BN_Model():
    def __init__(self, bn_model_template_folder, user_model_template, agent_model_template, bn_model_output, user_id, with_feedback):
        self.bn_model_folder = bn_model_folder
        self.user_model_template = bn_model_template_folder+"/"+user_model_template
        self.agent_model_template = bn_model_template_folder+"/"+agent_model_template
        if not os.path.exists(bn_model_output+"/"+str(user_id)):
            os.mkdir(bn_model_output+"/"+str(user_id))
        if not os.path.exists(bn_model_output+"/"+str(user_id)+ "/" + str(with_feedback)):
            os.mkdir(bn_model_output+"/"+str(user_id)+"/"+str(with_feedback))
        self.user_model_filename = bn_model_output+"/"+str(user_id)+"/"+str(with_feedback)+"/user_model.bif"
        self.agent_model_filename = bn_model_output+"/"+str(user_id)+"/"+str(with_feedback)+"/agent_model.bif"
        self.user_model = None
        self.agent_model = None
        self.attempt_preferences = [[0 for i in range(Attempt.counter.value)] for j in
                                     range(User_Action.counter.value)]
        self.game_preferences = [[0 for i in range(Game_State.counter.value)] for j in
                                  range(User_Action.counter.value)]
        self.user_action_preferences_on_agent_assistance = [[0 for i in range(User_Action.counter.value)]
                                              for j in range(Agent_Assistance.counter.value)]
        self.agent_assistance_preferences_on_attempt_game = [[0 for ass in range(Agent_Assistance.counter.value)]
                                        for g in range(Game_State.counter.value)
                                         for a in range(Attempt.counter.value)]

    #1 create two files where to include the user model and agent model, respectively.
    def create_template(self, filename_out, filename_in):
        dest = open(filename_out, "w")
        orig = open(filename_in, "r")
        contents = orig.readlines()
        if orig.mode == 'r':
            dest.writelines(contents)


    def get_attempt_given_user_action(self, attempt_preferences_for_correct_move):
        attempt_preferences_for_wrong_move = [(5 - attempt_preferences_for_correct_move[a]) / 2 for a in
                                              range(Attempt.counter.value)]
        attempt_preferences_for_timeout = [(5 - attempt_preferences_for_correct_move[a]) / 2 for a in
                                           range(Attempt.counter.value)]
        #normalise values
        for elem in range(len(self.attempt_preferences)):
            attempt_preferences_for_correct_move = [attempt_preferences_for_correct_move[i] * User_Action.counter.value for
                                                    i in
                                                    range(len(attempt_preferences_for_correct_move))]
            attempt_preferences_for_wrong_move = [attempt_preferences_for_wrong_move[i] * User_Action.counter.value for i in
                                                  range(len(attempt_preferences_for_wrong_move))]
            attempt_preferences_for_timeout = [attempt_preferences_for_timeout[i] * User_Action.counter.value for i in
                                               range(len(attempt_preferences_for_timeout))]
            self.attempt_preferences = [attempt_preferences_for_correct_move, attempt_preferences_for_wrong_move,
                                   attempt_preferences_for_timeout]

        return self.attempt_preferences

    def get_game_state_given_user_action(self, game_preferences_for_correct_move):
        game_preferences_for_wrong_move = [(5 - game_preferences_for_correct_move[a]) / 2 for a in
                                           range(Game_State.counter.value)]
        game_preferences_for_timeout = [(5 - game_preferences_for_correct_move[a]) / 2 for a in
                                        range(Game_State.counter.value)]

        for elem in range(len(self.game_preferences)):
            game_preferences_for_correct_move = [game_preferences_for_correct_move[i] * User_Action.counter.value for i in
                                                 range(len(game_preferences_for_correct_move))]
            game_preferences_for_wrong_move = [game_preferences_for_wrong_move[i] * User_Action.counter.value for i in
                                               range(len(game_preferences_for_wrong_move))]
            game_preferences_for_timeout = [game_preferences_for_timeout[i] * User_Action.counter.value for i in
                                            range(len(game_preferences_for_timeout))]
            self.game_preferences = [game_preferences_for_correct_move, game_preferences_for_wrong_move,
                                game_preferences_for_timeout]

        return self.game_preferences

    def get_user_action_given_agent_assistance(self, assistance_preferences_for_correct_move):
        assistance_preferences_for_wrong_move = [(5 - assistance_preferences_for_correct_move[i]) / 2 for i in
                                                 range(len(assistance_preferences_for_correct_move))]
        assistance_preferences_for_timeout = [(5 - assistance_preferences_for_correct_move[i]) / 2 for i in
                                              range(len(assistance_preferences_for_correct_move))]
        #normalise
        assistance_preferences_for_correct_move = [assistance_preferences_for_correct_move[i] * User_Action.counter.value for
                                                   i in
                                                   range(len(assistance_preferences_for_correct_move))]
        assistance_preferences_for_wrong_move = [assistance_preferences_for_wrong_move[i] * User_Action.counter.value for i in
                                                 range(len(assistance_preferences_for_wrong_move))]
        assistance_preferences_for_timeout = [assistance_preferences_for_timeout[i] * User_Action.counter.value for i in
                                              range(len(assistance_preferences_for_timeout))]
        for elem in range(Agent_Assistance.counter.value):

            den = (assistance_preferences_for_correct_move[elem] + assistance_preferences_for_wrong_move[elem] +
                   assistance_preferences_for_timeout[elem])
            self.user_action_preferences_on_agent_assistance[elem] = [assistance_preferences_for_correct_move[elem] / den,
                                                        assistance_preferences_for_wrong_move[elem] / den,
                                                        assistance_preferences_for_timeout[elem] / den]

        return self.user_action_preferences_on_agent_assistance


    def get_agent_assistance_given_attempt_and_game_state(self, agent_assistance_preferences_for_correct_move_game_attempt):
        it = 0
        for g in range(Game_State.counter.value):
            for a in range(Attempt.counter.value):
                print("it:", it)
                # get the preference
                agent_assistance_preferences_for_correct_move_game_attempt = [[(agent_assistance_preferences_for_correct_move_game_attempt[i][j] * Agent_Assistance.counter.value) for j in
                                           range(len(agent_assistance_preferences_for_correct_move_game_attempt[i]))] for i in
                                          range(len(agent_assistance_preferences_for_correct_move_game_attempt))]
                self.agent_assistance_preferences_on_attempt_game[it] = list(
                    map(lambda x: x / sum(agent_assistance_preferences_for_correct_move_game_attempt[a]), agent_assistance_preferences_for_correct_move_game_attempt[a]))
                it += 1
        return self.agent_assistance_preferences_on_attempt_game

    def read_therapist_data(self, therapist_questionnaire_filename, user_id):
        n_attempt = 4
        n_game_state = 3
        n_assistance = 6
        n_user_action = 3
        attempt_given_user = [0 for i in range(n_attempt)]
        game_state_given_user = [0 for i in range(n_game_state)]
        user_given_assistance = [0 for i in range(n_assistance)]
        assistance_given_attempt_game = [[ 0 for j in range(n_assistance)]for i in range(n_attempt)]
        #from 5 to 7 game_state | user_action
        #from 8 to 13 user_action | agent_assistance
        #from 14 to 37 where every 6 which to attempt
        data = pd.read_csv(therapist_questionnaire_filename)
        patient_info = (data.loc[data['User_ID'] == user_id])
        print("attempt|user_action:",patient_info.values[0][1:n_attempt+1])
        print("game_state|user_action:", patient_info.values[0][n_attempt+1:n_attempt+n_game_state+1])
        print("user_action|agent_assistance:", patient_info.values[0][n_attempt+n_game_state+1:n_attempt+n_game_state+n_assistance+1])
        print("agent_assistance|attempt,game_state:", patient_info.values[0][n_attempt+n_game_state+n_assistance+1:n_attempt+n_game_state+n_assistance+1+(n_assistance*n_attempt)])

        attempt_given_user = patient_info.values[0][1:n_attempt+1]
        game_given_user = patient_info.values[0][n_attempt+1:n_attempt+n_game_state+1]
        user_given_assistance = patient_info.values[0][n_attempt+n_game_state+1:n_attempt+n_game_state+n_assistance+1]
        assistance_given_attempt_game_in_row = patient_info.values[0][n_attempt+n_game_state+n_assistance+1:n_attempt+n_game_state+n_assistance+1+(n_assistance*n_attempt)]
        for i in range(n_attempt):
            for j in range(n_assistance):
                assistance_given_attempt_game[i][j] = assistance_given_attempt_game_in_row[(i*n_assistance)+j]
        print("agent_assistance|attempt,game_state:", assistance_given_attempt_game)

        return attempt_given_user, game_given_user, user_given_assistance, assistance_given_attempt_game


parser = argparse.ArgumentParser()
parser.add_argument('--bn_template', '--bn_model_folder', type=str,
                    help="folder in which all the user and the agent models are stored ",
                    default="/home/pal/Documents/Framework/bn_generative_model/bn_models_template")
parser.add_argument('--bn_user_template', '--bn_user_model', type=str, help="template name of  user bn model",
                    default="persona_model_template.bif")
parser.add_argument('--bn_agent_template', '--bn_agent_model', type=str, help="template name of  agent bn model",
                    default="agent_model_template.bif")
parser.add_argument('--bn_models_output', '--bn_models_output', type=str,
                    help="folder in which the bn model are saved",
                    default="/home/pal/Documents/Framework/GenerativeMutualShapingRL/BN_Models")
parser.add_argument('--therapist_knowledge', '--therapist_knowledge', type=str,
                    help="csv file of the questionnaire filled by the therapist",
                    default="/home/pal/Documents/Framework/GenerativeMutualShapingRL/therapist_questionnaire/therapist_questionnaire.csv")
parser.add_argument('--user_id', '--id', type=int, help="user id", required=True)
parser.add_argument('--with_feedback', '--f', type=eval, choices=[True, False], help="offering sociable", required=True)

args = parser.parse_args()
user_id = args.user_id
with_feedback = args.with_feedback
bn_model_folder = args.bn_template
user_model_template = args.bn_user_template#"persona_model_template.bif"
agent_model_template = args.bn_agent_template#"agent_model_template.bif"#provided by the gui
bn_model_output = args.bn_models_output#"/home/pal/Documents/Framework/GenerativeMutualShapingRL/BN_Models"
therapist_questionnaire_filename = args.therapist_knowledge#"/home/pal/Documents/Framework/GenerativeMutualShapingRL/therapist_questionnaire/therapist_questionnaire.csv"

bn_models = Therapist_BN_Model(bn_model_template_folder=bn_model_folder, user_model_template=user_model_template,
                               agent_model_template=agent_model_template, bn_model_output=bn_model_output, user_id=str(user_id), with_feedback=with_feedback)

attempt_given_user, game_given_user, user_given_assistance, assistance_given_attempt_game = bn_models.read_therapist_data(therapist_questionnaire_filename=therapist_questionnaire_filename, user_id=user_id)

user_action = [i for i in range(User_Action.counter.value)]
attempt = [i for i  in range (Attempt.counter.value)]
game = [i for i in range(Game_State.counter.value)]
agent_assistance = [i for i in range(Agent_Assistance.counter.value)]
max_value_for_user = 15
max_value_for_assistance = 30

attempt_preferences_for_correct_move = attempt_given_user#[2, 3, 4, 4]
assistance_preferences_for_correct_move = user_given_assistance#[2, 3, 3, 4, 5, 5]
game_preferences_for_correct_move = game_given_user#[2, 3, 4]
assistance_preferences_for_correct_move_game_attempt = assistance_given_attempt_game#[[4, 5, 2, 1, 1, 1],
                          # [1, 3, 5, 3, 1, 1],
                          # [1, 2, 2, 5, 4, 1],
                          # [1, 1, 2, 5, 4, 4]]

user_action_vars = ["(correct)", "(wrong)", "(timeout)"]
agent_assistance_vars = ["(lev_0)", "(lev_1)", "(lev_2)", "(lev_3)", "(lev_4)", "(lev_5)"]
attempt_game_vars = [
"(beg, att_1)", "(beg, att_2)","(beg, att_3)","(beg, att_4)",
"(mid, att_1)", "(mid, att_2)","(mid, att_3)","(mid, att_4)",
"(end, att_1)", "(end, att_2)","(end, att_3)","(end, att_4)",
]

#initialise the two models with the templates
bn_models.create_template(filename_out=bn_models.user_model_filename, filename_in=bn_models.user_model_template)
bn_models.create_template(filename_out=bn_models.agent_model_filename, filename_in=bn_models.agent_model_template)


#write all the values on a bif file
user_model = open(bn_models.user_model_filename, "a+")
agent_model = open(bn_models.agent_model_filename, "a+")

user_model.write("probability (game_state | user_action)  { \n")
bn_models.game_preferences = bn_models.get_game_state_given_user_action(game_preferences_for_correct_move)

for elem in range(len(bn_models.game_preferences)):
    bn_models.game_preferences[elem] = list(map(lambda x:x/sum(bn_models.game_preferences[elem]), bn_models.game_preferences[elem]))
    user_model.write(str(user_action_vars[elem]) + "\t" +
                     str(bn_models.game_preferences[elem][0]) + "," +
                     str(bn_models.game_preferences[elem][1]) + "," +
                     str(bn_models.game_preferences[elem][2]) + "; \n")
user_model.write("}\n")

user_model.write("probability (attempt | user_action)  { \n")
bn_models.attempt_preferences = bn_models.get_attempt_given_user_action(attempt_preferences_for_correct_move)
for elem in range(len(bn_models.attempt_preferences)):
    bn_models.attempt_preferences[elem] = list(map(lambda x:x/sum(bn_models.attempt_preferences[elem]), bn_models.attempt_preferences[elem]))
    user_model.write(str(user_action_vars[elem])+ "\t" +
                     str(bn_models.attempt_preferences[elem][0])+","+
                     str(bn_models.attempt_preferences[elem][1])+","+
                     str(bn_models.attempt_preferences[elem][2]) + "," +
                     str(bn_models.attempt_preferences[elem][3])+"; \n")
user_model.write("}\n")

user_model.write("probability (user_action | agent_assistance) { \n")
bn_models.assistance_preferences = bn_models.get_user_action_given_agent_assistance(assistance_preferences_for_correct_move)
for elem in range(Agent_Assistance.counter.value):
    #setting up the effect of each level on the user_action
    user_model.write(str(agent_assistance_vars[elem])+ "\t"+
                     str(bn_models.assistance_preferences[elem][0])+","+
                     str(bn_models.assistance_preferences[elem][1])+","+
                     str(bn_models.assistance_preferences[elem][2])+"; \n")
user_model.write("}")

agent_model.write("probability (agent_assistance | game_state, attempt) { \n")
bn_models.agent_assistance_preferences_on_attempt_game = bn_models.get_agent_assistance_given_attempt_and_game_state(assistance_preferences_for_correct_move_game_attempt)

it = 0
for g in range(Game_State.counter.value):
    for a in range(Attempt.counter.value):
        print("it:", it)
        #get the preference
        agent_model.write(str(attempt_game_vars[it])+"\t"+
                          str(bn_models.agent_assistance_preferences_on_attempt_game[it][0])+", "+
                          str(bn_models.agent_assistance_preferences_on_attempt_game[it][1]) + ", " +
                          str(bn_models.agent_assistance_preferences_on_attempt_game[it][2]) + ", " +
                          str(bn_models.agent_assistance_preferences_on_attempt_game[it][3]) + ", " +
                          str(bn_models.agent_assistance_preferences_on_attempt_game[it][4]) + ", " +
                          str(bn_models.agent_assistance_preferences_on_attempt_game[it][5]) + "; \n"
                          )
        it += 1
agent_model.write("}")


print(bn_models.attempt_preferences)
print(bn_models.game_preferences)
print(bn_models.assistance_preferences)
print(bn_models.agent_assistance_preferences_on_attempt_game)