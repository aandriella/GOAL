import random
#attempt | user_action
from bn_variables import User_Action, Game_State, Attempt, Agent_Assistance


#1 create two files where to include the user model and agent model, respectively.
def create_template(path, filename_out, filename_in):
    dest = open(path+"/"+filename_out, "w")
    orig = open(path+"/"+filename_in, "r")
    contents = orig.readlines()
    if orig.mode == 'r':
        dest.writelines(contents)

user_model_path = "/home/pal/Documents/Framework/bn_generative_model/bn_persona_model"
user_model_filename_out = "persona_test.bif"
user_model_filename_in = "persona_model_test.bif"
agent_model_path = "/home/pal/Documents/Framework/bn_generative_model/bn_agent_model"
agent_model_filename_out = "agent_test.bif"
agent_model_filename_in = "agent_model_test.bif"

create_template(path=user_model_path, filename_out=user_model_filename_out, filename_in=user_model_filename_in)

user_action = [0, 1, 2]
attempt = [0, 1, 2, 3]
game = [0, 1, 2]
agent_assistance = [0, 1,2 ,3 ,4 ,5 ]
max_value_for_user = 15
max_value_for_assistance = 30

user_action_vars = ["(correct)", "(wrong)", "(timeout)"]
agent_assistance_vars = ["(lev_0)", "(lev_1)", "(lev_2)", "(lev_3)", "(lev_4)", "(lev_5)"]
user_action_given_agent_assistance = [[0 for i in range(User_Action.counter.value)]
                                    for j in range(Agent_Assistance.counter.value)]
attempt_given_user_action = [[0 for i in range(Attempt.counter.value)] for j in
                                   range(User_Action.counter.value)]
game_given_user_action = [[0 for i in range(Game_State.counter.value)] for j in
                                      range(User_Action.counter.value)]


attempt_preferences_for_correct_move = [2, 3, 4, 4]
attempt_preferences_for_wrong_move = [(5-attempt_preferences_for_correct_move[a])/2 for a in range(Attempt.counter.value)]
attempt_preferences_for_timeout = [(5-attempt_preferences_for_correct_move[a])/2 for a in range(Attempt.counter.value)]

attempt_preferences_for_correct_move = [attempt_preferences_for_correct_move[i] * User_Action.counter.value for i in
                                     range(len(attempt_preferences_for_correct_move))]
attempt_preferences_for_wrong_move = [attempt_preferences_for_wrong_move[i] * User_Action.counter.value for i in
                                     range(len(attempt_preferences_for_wrong_move))]
attempt_preferences_for_timeout = [attempt_preferences_for_timeout[i] * User_Action.counter.value for i in
                                     range(len(attempt_preferences_for_timeout))]
attempt_preferences = [attempt_preferences_for_correct_move, attempt_preferences_for_wrong_move, attempt_preferences_for_timeout]

user_model = open(user_model_path+"/"+user_model_filename_out, "a+")
user_model.write("probability (game_state | user_action)  { \n")
game_preferences_for_correct_move = [2, 3, 4]
game_preferences_for_wrong_move = [(5-game_preferences_for_correct_move[a])/2 for a in range(Game_State.counter.value)]
game_preferences_for_timeout = [(5-game_preferences_for_correct_move[a])/2 for a in range(Game_State.counter.value)]

game_preferences_for_correct_move = [game_preferences_for_correct_move[i] * User_Action.counter.value for i in
                                     range(len(game_preferences_for_correct_move))]
game_preferences_for_wrong_move = [game_preferences_for_wrong_move[i] * User_Action.counter.value for i in
                                     range(len(game_preferences_for_wrong_move))]
game_preferences_for_timeout = [game_preferences_for_timeout[i] * User_Action.counter.value for i in
                                     range(len(game_preferences_for_timeout))]
game_preferences = [game_preferences_for_correct_move, game_preferences_for_wrong_move, game_preferences_for_timeout]

for elem in range(len(game_given_user_action)):
    game_preferences[elem] = list(map(lambda x:x/sum(game_preferences[elem]), game_preferences[elem]))
    user_model.write(str(user_action_vars[elem]) + "\t" +
                     str(game_preferences[elem][0]) + "," +
                     str(game_preferences[elem][1]) + "," +
                     str(game_preferences[elem][2]) + "; \n")
user_model.write("}\n")


user_model.write("probability (attempt | user_action)  { \n")
for elem in range(len(attempt_given_user_action)):
    attempt_preferences[elem] = list(map(lambda x:x/sum(attempt_preferences[elem]), attempt_preferences[elem]))
    user_model.write(str(user_action_vars[elem])+ "\t" +
                     str(attempt_preferences[elem][0])+","+
                     str(attempt_preferences[elem][1])+","+
                     str(attempt_preferences[elem][2]) + "," +
                     str(attempt_preferences[elem][3])+"; \n")
user_model.write("}\n")



user_model.write("probability (user_action | agent_assistance) { \n")
assistance_score_for_correct_move = [2, 3, 3, 4, 5, 5]
assistance_score_for_wrong_move = [(5 - assistance_score_for_correct_move[i]) / 2 for i in
                                   range(len(assistance_score_for_correct_move))]
assistance_score_for_timeout = [(5 - assistance_score_for_correct_move[i]) / 2 for i in
                                range(len(assistance_score_for_correct_move))]
assistance_score_for_correct_move = [assistance_score_for_correct_move[i] * User_Action.counter.value for i in
                                     range(len(assistance_score_for_correct_move))]
assistance_score_for_wrong_move = [assistance_score_for_wrong_move[i] * User_Action.counter.value for i in
                                     range(len(assistance_score_for_wrong_move))]
assistance_score_for_timeout = [assistance_score_for_timeout[i] * User_Action.counter.value for i in
                                     range(len(assistance_score_for_timeout))]

for elem in range(Agent_Assistance.counter.value):
    #setting up the effect of each level on the user_action
    den = (assistance_score_for_correct_move[elem]+assistance_score_for_wrong_move[elem]+assistance_score_for_timeout[elem])
    user_action_given_agent_assistance[elem] = [assistance_score_for_correct_move[elem]/den, assistance_score_for_wrong_move[elem]/den,
                                                assistance_score_for_timeout[elem]/den]
    user_model.write(str(agent_assistance_vars[elem])+ "\t"+
                     str(user_action_given_agent_assistance[elem][0])+","+
                     str(user_action_given_agent_assistance[elem][1])+","+
                     str(user_action_given_agent_assistance[elem][2])+"; \n")
user_model.write("}")


create_template(path=agent_model_path, filename_out=agent_model_filename_out, filename_in=agent_model_filename_in )

agent_assistance_given_game_attempt = [[0 for ass in range(Agent_Assistance.counter.value)]
                                        for g in range(Game_State.counter.value)
                                         for a in range(Attempt.counter.value)]
agent_model = open(agent_model_path+"/"+agent_model_filename_out, "a+")

attempt_game_vars = [
"(beg, att_1)", "(beg, att_2)","(beg, att_3)","(beg, att_4)",
"(mid, att_1)", "(mid, att_2)","(mid, att_3)","(mid, att_4)",
"(end, att_1)", "(end, att_2)","(end, att_3)","(end, att_4)",
]

agent_model.write("probability (agent_assistance | game_state, attempt) { \n")
assistance_preferences = [[4, 5, 2, 1, 1, 1],
                          [1, 3, 5, 3, 1, 1],
                          [1, 2, 2, 5, 4, 1],
                          [1, 1, 2, 5, 4, 4]]

it = 0
for g in range(Game_State.counter.value):
    for a in range(Attempt.counter.value):
        print("it:", it)
        #get the preference
        assistance_preferences = [[(assistance_preferences[i][j] * Agent_Assistance.counter.value)  for j in range(len(assistance_preferences[i]))] for i in range(len(assistance_preferences))]
        agent_assistance_given_game_attempt[it] = list(map(lambda x:x/sum(assistance_preferences[a]), assistance_preferences[a]))
        agent_model.write(str(attempt_game_vars[it])+"\t"+
                          str(agent_assistance_given_game_attempt[it][0])+", "+
                          str(agent_assistance_given_game_attempt[it][1]) + ", " +
                          str(agent_assistance_given_game_attempt[it][2]) + ", " +
                          str(agent_assistance_given_game_attempt[it][3]) + ", " +
                          str(agent_assistance_given_game_attempt[it][4]) + ", " +
                          str(agent_assistance_given_game_attempt[it][5]) + "; \n"
                          )
        it += 1
agent_model.write("}")


print(attempt_given_user_action)
print(game_given_user_action)
print(user_action_given_agent_assistance)
print(agent_assistance_given_game_attempt)