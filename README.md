### GOAL simulator + Policy generator ### 
#### STEPS:
 -1. Create initial user cognitive model from data (human therapist and patient) <br />
 -2. Create robot policy from data (human therapist) or update it if session > 0  <br />
 -3. Run the simulation using the [BN_GenerativeModel](https://github.com/aandriella/BN_GenerativeModel) package  <br />
 -4. Generate new episodes  <br />
 -5. Learn the robot reward using [Maximum Causal Entropy Inverse Reiforcement Learning](https://github.com/aandriella/MaxEntIRL) algorithm proposed Ziebart's thesis (2010)   <br />
 -6. Compute the policy related to that reward using Value Iteration  <br />
 -7. Run a session between the robot and the patient  <br />
 -Repeat from 2  <br />


#### Package:
- BN_Models folder contains the BNs of the patient and the therapist (human or robot) 
- questionnaire_gui.py is the script for generating the BNs of the patient and the robot from the questionnaire filled in by the therapist about the cognitive capability of the patient in the specific task. In this case the therapist filled in the data using a python GUI with the questionnaire
- questionnaire_googleform.py is the script for generating the BNs of the patient and the robot from the questionnaire filled in by the therapist about the cognitive capability of the patient in the specific task. In this case the therapist filled in the data using a google form and then the excel sheet is processed in order to generate the models.
- plot_results.py: auxiliary class to plot the results
- main.py: it generates the policy for the given patient

### USAGE ###

``` 
python main.py --bn_model_folder ($PATH) --output_policy_filename ($PATH) --output_reward_filename ($PATH) --output_value_filename ($PATH) --therapist_patient_interaction_folder ($PATH) --agent_patient_interaction_folder ($PATH) --epoch 5 --run 50 --user_id 19 --f True --s 0 --objective neutral
```
where:
-- bn_model_folder, folder contains the initial bn models of therapist and patient <br />
-- epoch, the number of epoch of the simulation  <br />
-- run, number of runs per epoch  <br />
-- output_policy_filename, output of the computed policy  <br />
-- output_reward_filename, output of the computed reward  <br />
-- output_value_filename, output of the computed value function  <br />
-- therapist_patient_interaction_folder, data containing the logs from the  sessions between the human therapist and the patient  <br />
-- agent_patient_interaction_folder, data containing the logs from the  sessions between the robot therapist and the patient  <br />
-- user_id, id of the user   <br />
-- with_feedback, if  [SOCIABLE](http://www.iri.upc.edu/files/scidoc/2353-Discovering-SOCIABLE:-Using-a-conceptual-model-to-evaluate-the-legibility-and-effectiveness-of-backchannel-cues-in-an-entertainment-scenario.pdf)  is used   <br />
-- session,  the session id  <br />
-- agent_objective, objective can be either challenge if we want to challenge more the user, help if we want to help more the user or finally it can neutral so we do not reshape the policy. 
