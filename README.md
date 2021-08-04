### GOAL simulator + Policy generator ### 
#### STEPS:
##### - 1: CREATE INITIAL USER COGNITIVE MODEL FROM DATA (human therapist and patient)
##### - 2: CREATE ROBOT INITIAL POLICY FROM DATA (human therapist) OR UPDATE IT IF SESSION > 0
##### - 3: RUN THE SIMULATION using the [BN_GenerativeModel](https://github.com/aandriella/BN_GenerativeModel) package
##### - 4: GENERATE THE NEW EPISODES
##### - 5: LEARN THE ROBOT REWARD USING MAXIMUM CAUSAL ENTROPY INVERSE REINFORCEMENT LEARNING algorithm proposed Ziebart's thesis (2010) [MaxEntropyIRL](https://github.com/aandriella/MaxEntRL)
##### - 6: COMPUTE THE POLICY RELATED TO THAT REWARD USING VALUE ITERATION
##### - 7: RUN A SESSION WITH THE PATIENT
##### - REPEAT FROM 2


#### Package:
- BN_Models folder contains the BNs of the patient and the therapist (human or robot) 
- questionnaire_gui.py is the script for generating the BNs of the patient and the robot from the questionnaire filled in by the therapist about the cognitive capability of the patient in the specific task. In this case the therapist filled in the data using a python GUI with the questionnaire
- questionnaire_googleform.py is the script for generating the BNs of the patient and the robot from the questionnaire filled in by the therapist about the cognitive capability of the patient in the specific task. In this case the therapist filled in the data using a google form and then the excel sheet is processed in order to generate the models.
- plot_results.py: auxiliary class to plot the results
- main.py: it generates the policy for the given patient

### USAGE ###

``` 
python main.py --epoch 5 --run 50 --user_id 19 --f True --s 0 --objective neutral
```
where:
- bn_model_folder, folder contains the initial bn models of therapist and patient
- epoch, the number of epoch of the simulation
- run, number of runs per epoch 
- output_policy_filename, output of the computed policy
- output_reward_filename, output of the computed reward
- output_value_filename, output of the computed value function
- therapist_patient_interaction_folder, data containing the logs from the  sessions between the human therapist and the patient
- agent_patient_interaction_folder, data containing the logs from the  sessions between the robot therapist and the patient
- user_id, id of the user 
- with_feedback, if  [SOCIABLE](http://www.iri.upc.edu/files/scidoc/2353-Discovering-SOCIABLE:-Using-a-conceptual-model-to-evaluate-the-legibility-and-effectiveness-of-backchannel-cues-in-an-entertainment-scenario.pdf)  is used 
- session,  the session id
- agent_objective, objective can be either challenge if we want to challenge more the user, help if we want to help more the user or finally it can neutral so we do not reshape the policy.