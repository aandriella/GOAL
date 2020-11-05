from tkinter import *
from tkinter import ttk

class GUI():
    def __init__(self, root, notebook, frame1, frame2, frame3, frame4, frame5):
        self.root = root
        self.notebook = notebook
        self.frame1 = frame1
        self.frame2 = frame2
        self.frame3 = frame3
        self.frame4 = frame4
        self.frame5 = frame5

        self.id = StringVar()
        self.user_model = StringVar()
        self.agent_model = StringVar()

        self.selected_value_att_1 = IntVar()
        self.selected_value_att_2 = IntVar()
        self.selected_value_att_3 = IntVar()
        self.selected_value_att_4 = IntVar()

        self.selected_value_game_1 = IntVar()
        self.selected_value_game_2 = IntVar()
        self.selected_value_game_3 = IntVar()

        self.selected_value_user_1 = IntVar()
        self.selected_value_user_2 = IntVar()
        self.selected_value_user_3 = IntVar()
        self.selected_value_user_4 = IntVar()
        self.selected_value_user_5 = IntVar()
        self.selected_value_user_6 = IntVar()

        self.selected_value_ass_0_att_1 = IntVar()
        self.selected_value_ass_0_att_2 = IntVar()
        self.selected_value_ass_0_att_3 = IntVar()
        self.selected_value_ass_0_att_4 = IntVar()
        self.selected_value_ass_1_att_1 = IntVar()
        self.selected_value_ass_1_att_2 = IntVar()
        self.selected_value_ass_1_att_3 = IntVar()
        self.selected_value_ass_1_att_4 = IntVar()
        self.selected_value_ass_2_att_1 = IntVar()
        self.selected_value_ass_2_att_2 = IntVar()
        self.selected_value_ass_2_att_3 = IntVar()
        self.selected_value_ass_2_att_4 = IntVar()
        self.selected_value_ass_3_att_1 = IntVar()
        self.selected_value_ass_3_att_2 = IntVar()
        self.selected_value_ass_3_att_3 = IntVar()
        self.selected_value_ass_3_att_4 = IntVar()
        self.selected_value_ass_4_att_1 = IntVar()
        self.selected_value_ass_4_att_2 = IntVar()
        self.selected_value_ass_4_att_3 = IntVar()
        self.selected_value_ass_4_att_4 = IntVar()
        self.selected_value_ass_5_att_1 = IntVar()
        self.selected_value_ass_5_att_2 = IntVar()
        self.selected_value_ass_5_att_3 = IntVar()
        self.selected_value_ass_5_att_4 = IntVar()

        self.notebook.add(self.frame5, text="User Info")
        self.notebook.add(self.frame1, text="Attempt|User_Action")
        self.notebook.add(self.frame2, text="Game_State|User_Action")
        self.notebook.add(self.frame3, text="User_Action|Robot_Assistance")
        self.notebook.add(self.frame4, text="Robot_Assistance|User_Action")

        Label(self.frame5,text="User ID").grid(row=2,column=0)
        Entry(self.frame5, textvariable=self.id).grid(row=2, column=1)
        Label(self.frame5,text="User Model Name").grid(row=3,column=0)
        Entry(self.frame5, textvariable=self.user_model).grid(row=3, column=1)
        Label(self.frame5, text="Agent Model Name").grid(row=4, column=0)
        Entry(self.frame5, textvariable=self.agent_model).grid(row=4, column=1)
        Button(self.frame5, text="Save", command=self.save).grid(row=8, column=0)

        Label(self.frame1, text="What is the chance the patient will move the correct token at his first attempt?").grid(row=2, column=0)
        Radiobutton(self.frame1, text="1", variable=self.selected_value_att_1, value=1).grid(row=2, column=2)
        Radiobutton(self.frame1, text="2", variable=self.selected_value_att_1, value=2).grid(row=2, column=3)
        Radiobutton(self.frame1, text="3", variable=self.selected_value_att_1, value=3).grid(row=2, column=4)
        Radiobutton(self.frame1, text="4", variable=self.selected_value_att_1, value=4).grid(row=2, column=5)
        Radiobutton(self.frame1, text="5", variable=self.selected_value_att_1, value=5).grid(row=2, column=6)

        Label(self.frame1, text="What is the chance the patient will move the correct token at his second attempt?").grid(row=4, column=0)
        Radiobutton(self.frame1, text="1", variable=self.selected_value_att_2, value=1).grid(row=4, column=2)
        Radiobutton(self.frame1, text="2", variable=self.selected_value_att_2, value=2).grid(row=4, column=3)
        Radiobutton(self.frame1, text="3", variable=self.selected_value_att_2, value=3).grid(row=4, column=4)
        Radiobutton(self.frame1, text="4", variable=self.selected_value_att_2, value=4).grid(row=4, column=5)
        Radiobutton(self.frame1, text="5", variable=self.selected_value_att_2, value=5).grid(row=4, column=6)

        Label(self.frame1, text="What is the chance the patient will move the correct token at his third attempt?").grid(row=6, column=0)
        Radiobutton(self.frame1, text="1", variable=self.selected_value_att_3, value=1).grid(row=6, column=2)
        Radiobutton(self.frame1, text="2", variable=self.selected_value_att_3, value=2).grid(row=6, column=3)
        Radiobutton(self.frame1, text="3", variable=self.selected_value_att_3, value=3).grid(row=6, column=4)
        Radiobutton(self.frame1, text="4", variable=self.selected_value_att_3, value=4).grid(row=6, column=5)
        Radiobutton(self.frame1, text="5", variable=self.selected_value_att_3, value=5).grid(row=6, column=6)

        Label(self.frame1, text="What is the chance the patient will move the correct token at his fourth attempt?").grid(row=8)
        Radiobutton(self.frame1, text="1", variable=self.selected_value_att_4, value=1).grid(row=8, column=2)
        Radiobutton(self.frame1, text="2", variable=self.selected_value_att_4, value=2).grid(row=8, column=3)
        Radiobutton(self.frame1, text="3", variable=self.selected_value_att_4, value=3).grid(row=8, column=4)
        Radiobutton(self.frame1, text="4", variable=self.selected_value_att_4, value=4).grid(row=8, column=5)
        Radiobutton(self.frame1, text="5", variable=self.selected_value_att_4, value=5).grid(row=8, column=6)


        Label(self.frame1, text="DONE?").grid(row=12)
        press_button = Button(self.frame1, text="OK", command=self.get_value_attempt).grid(row=13)


        ##################################################################################

        Label(self.frame2, text="What is the chance the patient will move the correct token at the beginning of the game?").grid(row=2, column=0)
        Radiobutton(self.frame2, text="1", variable=self.selected_value_game_1, value=1).grid(row=2, column=2)
        Radiobutton(self.frame2, text="2", variable=self.selected_value_game_1, value=2).grid(row=2, column=3)
        Radiobutton(self.frame2, text="3", variable=self.selected_value_game_1, value=3).grid(row=2, column=4)
        Radiobutton(self.frame2, text="4", variable=self.selected_value_game_1, value=4).grid(row=2, column=5)
        Radiobutton(self.frame2, text="5", variable=self.selected_value_game_1, value=5).grid(row=2, column=6)

        Label(self.frame2, text="What is the chance the patient will move the correct token in the middle of the game?").grid(row=4, column=0)
        Radiobutton(self.frame2, text="1", variable=self.selected_value_game_2, value=1).grid(row=4, column=2)
        Radiobutton(self.frame2, text="2", variable=self.selected_value_game_2, value=2).grid(row=4, column=3)
        Radiobutton(self.frame2, text="3", variable=self.selected_value_game_2, value=3).grid(row=4, column=4)
        Radiobutton(self.frame2, text="4", variable=self.selected_value_game_2, value=4).grid(row=4, column=5)
        Radiobutton(self.frame2, text="5", variable=self.selected_value_game_2, value=5).grid(row=4, column=6)

        Label(self.frame2, text="What is the chance the patient will move the correct token at the end?").grid(row=6, column=0)
        Radiobutton(self.frame2, text="1", variable=self.selected_value_game_3, value=1).grid(row=6, column=2)
        Radiobutton(self.frame2, text="2", variable=self.selected_value_game_3, value=2).grid(row=6, column=3)
        Radiobutton(self.frame2, text="3", variable=self.selected_value_game_3, value=3).grid(row=6, column=4)
        Radiobutton(self.frame2, text="4", variable=self.selected_value_game_3, value=4).grid(row=6, column=5)
        Radiobutton(self.frame2, text="5", variable=self.selected_value_game_3, value=5).grid(row=6, column=6)

        Label(self.frame2, text="DONE?").grid(row=12)
        Button(self.frame2, text="OK", command=self.get_value_game).grid(row=13)
        ############################################################################################

        Label(self.frame3, text="What is the chance the patient will move the correct token at his first attempt?").grid(row=2)
        Radiobutton(self.frame3, text="1", variable=self.selected_value_user_1, value=1).grid(row=2, column=2)
        Radiobutton(self.frame3, text="2", variable=self.selected_value_user_1, value=2).grid(row=2, column=3)
        Radiobutton(self.frame3, text="3", variable=self.selected_value_user_1, value=3).grid(row=2, column=4)
        Radiobutton(self.frame3, text="4", variable=self.selected_value_user_1, value=4).grid(row=2, column=5)
        Radiobutton(self.frame3, text="5", variable=self.selected_value_user_1, value=5).grid(row=2, column=6)

        Label(self.frame3, text="What is the chance the patient will move the correct token at his second attempt?").grid(row=4)
        Radiobutton(self.frame3, text="1", variable=self.selected_value_user_2, value=1).grid(row=4, column=2)
        Radiobutton(self.frame3, text="2", variable=self.selected_value_user_2, value=2).grid(row=4, column=3)
        Radiobutton(self.frame3, text="3", variable=self.selected_value_user_2, value=3).grid(row=4, column=4)
        Radiobutton(self.frame3, text="4", variable=self.selected_value_user_2, value=4).grid(row=4, column=5)
        Radiobutton(self.frame3, text="5", variable=self.selected_value_user_2, value=5).grid(row=4, column=6)

        Label(frame3, text="What is the chance the patient will move the correct token at his third attempt?").grid(row=6)
        Radiobutton(self.frame3, text="1", variable=self.selected_value_user_3, value=1).grid(row=6, column=2)
        Radiobutton(self.frame3, text="2", variable=self.selected_value_user_3, value=2).grid(row=6, column=3)
        Radiobutton(self.frame3, text="3", variable=self.selected_value_user_3, value=3).grid(row=6, column=4)
        Radiobutton(self.frame3, text="4", variable=self.selected_value_user_3, value=4).grid(row=6, column=5)
        Radiobutton(self.frame3, text="5", variable=self.selected_value_user_3, value=5).grid(row=6, column=6)

        Label(self.frame3, text="What is the chance the patient will move the correct token at his fourth attempt?").grid(row=8)
        Radiobutton(self.frame3, text="1", variable=self.selected_value_user_4, value=1).grid(row=8, column=2)
        Radiobutton(self.frame3, text="2", variable=self.selected_value_user_4, value=2).grid(row=8, column=3)
        Radiobutton(self.frame3, text="3", variable=self.selected_value_user_4, value=3).grid(row=8, column=4)
        Radiobutton(self.frame3, text="4", variable=self.selected_value_user_4, value=4).grid(row=8, column=5)
        Radiobutton(self.frame3, text="5", variable=self.selected_value_user_4, value=5).grid(row=8, column=6)

        Label(self.frame3, text="What is the chance the patient will move the correct token at his fourth attempt?").grid(row=10)
        Radiobutton(self.frame3, text="1", variable=self.selected_value_user_5, value=1).grid(row=10, column=2)
        Radiobutton(self.frame3, text="2", variable=self.selected_value_user_5, value=2).grid(row=10, column=3)
        Radiobutton(self.frame3, text="3", variable=self.selected_value_user_5, value=3).grid(row=10, column=4)
        Radiobutton(self.frame3, text="4", variable=self.selected_value_user_5, value=4).grid(row=10, column=5)
        Radiobutton(self.frame3, text="5", variable=self.selected_value_user_5, value=5).grid(row=10, column=6)

        Label(self.frame3, text="What is the chance the patient will move the correct token at his fourth attempt?").grid(row=12)
        Radiobutton(self.frame3, text="1", variable=self.selected_value_user_6, value=1).grid(row=12, column=2)
        Radiobutton(self.frame3, text="2", variable=self.selected_value_user_6, value=2).grid(row=12, column=3)
        Radiobutton(self.frame3, text="3", variable=self.selected_value_user_6, value=3).grid(row=12, column=4)
        Radiobutton(self.frame3, text="4", variable=self.selected_value_user_6, value=4).grid(row=12, column=5)
        Radiobutton(self.frame3, text="5", variable=self.selected_value_user_6, value=5).grid(row=12, column=6)

        Label(self.frame3, text="DONE?").grid(row=17)
        Button(self.frame3, text="OK", command=self.get_value_user).grid(row=18)


        #########################################################################################################
        Label(self.frame4,  text="Would you offer LEV 0 at the user's first attempt?").grid(row=2)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_0_att_1, value=1).grid(row=2, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_0_att_1, value=2).grid(row=2, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_0_att_1, value=3).grid(row=2, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_0_att_1, value=4).grid(row=2, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_0_att_1, value=5).grid(row=2, column=6)
        Label(self.frame4, text="Would you offer LEV 1 at the user's first attempt?").grid(row=4)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_1_att_1, value=1).grid(row=4, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_1_att_1, value=2).grid(row=4, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_1_att_1, value=3).grid(row=4, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_1_att_1, value=4).grid(row=4, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_1_att_1, value=5).grid(row=4, column=6)
        Label(frame4, text="Would you offer LEV 2 at the user's first attempt?").grid(row=6)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_2_att_1, value=1).grid(row=6, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_2_att_1, value=2).grid(row=6, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_2_att_1, value=3).grid(row=6, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_2_att_1, value=4).grid(row=6, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_2_att_1, value=5).grid(row=6, column=6)
        Label(self.frame4, text="Would you offer LEV 3 at the user's first attempt?").grid(row=8)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_3_att_1, value=1).grid(row=8, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_3_att_1, value=2).grid(row=8, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_3_att_1, value=3).grid(row=8, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_3_att_1, value=4).grid(row=8, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_3_att_1, value=5).grid(row=8, column=6)
        Label(self.frame4, text="Would you offer LEV 4 at the user's first attempt?").grid(row=10)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_4_att_1, value=1).grid(row=10, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_4_att_1, value=2).grid(row=10, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_4_att_1, value=3).grid(row=10, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_4_att_1, value=4).grid(row=10, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_4_att_1, value=5).grid(row=10, column=6)
        Label(self.frame4, text="Would you offer LEV 5 at the user's first attempt?").grid(row=12)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_5_att_1, value=1).grid(row=12, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_5_att_1, value=2).grid(row=12, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_5_att_1, value=3).grid(row=12, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_5_att_1, value=4).grid(row=12, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_5_att_1, value=5).grid(row=12, column=6)
        Label(self.frame4, text="-----------------------------------------------------").grid(row=14)

        row_att1= 15
        Label(self.frame4, text="Would you offer LEV 0 at the user's second attempt?").grid(row=2+row_att1)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_0_att_2, value=1).grid(row=2+row_att1, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_0_att_2, value=2).grid(row=2+row_att1, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_0_att_2, value=3).grid(row=2+row_att1, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_0_att_2, value=4).grid(row=2+row_att1, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_0_att_2, value=5).grid(row=2+row_att1, column=6)
        Label(self.frame4, text="Would you offer LEV 1 at the user's second attempt?").grid(row=4+row_att1)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_1_att_2, value=1).grid(row=4+row_att1, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_1_att_2, value=2).grid(row=4+row_att1, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_1_att_2, value=3).grid(row=4+row_att1, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_1_att_2, value=4).grid(row=4+row_att1, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_1_att_2, value=5).grid(row=4+row_att1, column=6)
        Label(frame4, text="Would you offer LEV 2 at the user's second attempt?").grid(row=6+row_att1)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_2_att_2, value=1).grid(row=6+row_att1, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_2_att_2, value=2).grid(row=6+row_att1, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_2_att_2, value=3).grid(row=6+row_att1, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_2_att_2, value=4).grid(row=6+row_att1, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_2_att_2, value=5).grid(row=6+row_att1, column=6)
        Label(self.frame4, text="Would you offer LEV 3 at the user's second attempt?").grid(row=8+row_att1)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_3_att_2, value=1).grid(row=8+row_att1, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_3_att_2, value=2).grid(row=8+row_att1, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_3_att_2, value=3).grid(row=8+row_att1, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_3_att_2, value=4).grid(row=8+row_att1, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_3_att_2, value=5).grid(row=8+row_att1, column=6)
        Label(self.frame4, text="Would you offer LEV 4 at the user's second attempt?").grid(row=10+row_att1)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_4_att_2, value=1).grid(row=10+row_att1, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_4_att_2, value=2).grid(row=10+row_att1, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_4_att_2, value=3).grid(row=10+row_att1, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_4_att_2, value=4).grid(row=10+row_att1, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_4_att_2, value=5).grid(row=10+row_att1, column=6)
        Label(self.frame4, text="Would you offer LEV 5 at the user's second attempt?").grid(row=12+row_att1)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_5_att_2, value=1).grid(row=12+row_att1, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_5_att_2, value=2).grid(row=12+row_att1, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_5_att_2, value=3).grid(row=12+row_att1, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_5_att_2, value=4).grid(row=12+row_att1, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_5_att_2, value=5).grid(row=12+row_att1, column=6)
        Label(self.frame4, text="-----------------------------------------------------").grid(row=14+row_att1)

        row_att2 = 15+row_att1

        Label(self.frame4, text="Would you offer LEV 0 at the user's third attempt?").grid(row=2 + row_att2)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_0_att_3, value=1).grid(row=2 + row_att2, column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_0_att_3, value=2).grid(row=2 + row_att2, column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_0_att_3, value=3).grid(row=2 + row_att2, column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_0_att_3, value=4).grid(row=2 + row_att2, column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_0_att_3, value=5).grid(row=2 + row_att2, column=6)
        Label(self.frame4, text="Would you offer LEV 1 at the user's third attempt?").grid(row=4 + row_att2)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_1_att_3, value=1).grid(row=4 + row_att2,column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_1_att_3, value=2).grid(row=4 + row_att2,column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_1_att_3, value=3).grid(row=4 + row_att2,column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_1_att_3, value=4).grid(row=4 + row_att2,column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_1_att_3, value=5).grid(row=4 + row_att2,column=6)
        Label(frame4, text="Would you offer LEV 2 at the user's third attempt?").grid(row=6 + row_att2)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_2_att_3, value=1).grid(row=6 + row_att2,column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_2_att_3, value=2).grid(row=6 + row_att2,column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_2_att_3, value=3).grid(row=6 + row_att2,column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_2_att_3, value=4).grid(row=6 + row_att2,column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_2_att_3, value=5).grid(row=6 + row_att2,column=6)
        Label(self.frame4, text="Would you offer LEV 3 at the user's third attempt?").grid(row=8 + row_att2)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_3_att_3, value=1).grid(row=8 + row_att2,column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_3_att_3, value=2).grid(row=8 + row_att2,column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_3_att_3, value=3).grid(row=8 + row_att2,column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_3_att_3, value=4).grid(row=8 + row_att2,column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_3_att_3, value=5).grid(row=8 + row_att2,column=6)
        Label(self.frame4, text="Would you offer LEV 4 at the user's third attempt?").grid(row=10 + row_att2)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_4_att_3, value=1).grid(row=10 + row_att2,column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_4_att_3, value=2).grid(row=10 + row_att2,column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_4_att_3, value=3).grid(row=10 + row_att2,column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_4_att_3, value=4).grid(row=10 + row_att2,column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_4_att_3, value=5).grid(row=10 + row_att2,column=6)
        Label(self.frame4, text="Would you offer LEV 5 at the user's third attempt?").grid(row=12 + row_att2)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_5_att_3, value=1).grid(row=12 + row_att2,column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_5_att_3, value=2).grid(row=12 + row_att2,column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_5_att_3, value=3).grid(row=12 + row_att2,column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_5_att_3, value=4).grid(row=12 + row_att2,column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_5_att_3, value=5).grid(row=12 + row_att2,column=6)
        Label(self.frame4, text="-----------------------------------------------------").grid(row=14+row_att2)

        row_att3 = 15+row_att2
        Label(self.frame4, text="Would you offer LEV 0 at the user's fourth attempt?").grid(row=2 + row_att3)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_0_att_4, value=1).grid(row=2 + row_att3,column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_0_att_4, value=2).grid(row=2 + row_att3,column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_0_att_4, value=3).grid(row=2 + row_att3,column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_0_att_4, value=4).grid(row=2 + row_att3,column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_0_att_4, value=5).grid(row=2 + row_att3,column=6)
        Label(self.frame4, text="Would you offer LEV 1 at the user's fourth attempt?").grid(row=4 + row_att3)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_1_att_4, value=1).grid(row=4 + row_att3,column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_1_att_4, value=2).grid(row=4 + row_att3,column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_1_att_4, value=3).grid(row=4 + row_att3,column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_1_att_4, value=4).grid(row=4 + row_att3,column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_1_att_4, value=5).grid(row=4 + row_att3,column=6)
        Label(frame4, text="Would you offer LEV 2 at the user's fourth attempt?").grid(row=6 + row_att3)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_2_att_4, value=1).grid(row=6 + row_att3,column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_2_att_4, value=2).grid(row=6 + row_att3,column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_2_att_4, value=3).grid(row=6 + row_att3,column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_2_att_4, value=4).grid(row=6 + row_att3,column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_2_att_4, value=5).grid(row=6 + row_att3,column=6)
        Label(self.frame4, text="Would you offer LEV 3 at the user's fourth attempt?").grid(row=8 + row_att3)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_3_att_4, value=1).grid(row=8 + row_att3,column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_3_att_4, value=2).grid(row=8 + row_att3,column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_3_att_4, value=3).grid(row=8 + row_att3,column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_3_att_4, value=4).grid(row=8 + row_att3,column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_3_att_4, value=5).grid(row=8 + row_att3,column=6)
        Label(self.frame4, text="Would you offer LEV 4 at the user's fourth attempt?").grid(row=10 + row_att3)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_4_att_4, value=1).grid(row=10 + row_att3,column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_4_att_4, value=2).grid(row=10 + row_att3,column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_4_att_4, value=3).grid(row=10 + row_att3,column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_4_att_4, value=4).grid(row=10 + row_att3,column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_4_att_4, value=5).grid(row=10 + row_att3,column=6)
        Label(self.frame4, text="Would you offer LEV 5 at the user's fourth attempt?").grid(row=12 + row_att3)
        Radiobutton(self.frame4, text="1", variable=self.selected_value_ass_5_att_4, value=1).grid(row=12 + row_att3,column=2)
        Radiobutton(self.frame4, text="2", variable=self.selected_value_ass_5_att_4, value=2).grid(row=12 + row_att3,column=3)
        Radiobutton(self.frame4, text="3", variable=self.selected_value_ass_5_att_4, value=3).grid(row=12 + row_att3,column=4)
        Radiobutton(self.frame4, text="4", variable=self.selected_value_ass_5_att_4, value=4).grid(row=12 + row_att3,column=5)
        Radiobutton(self.frame4, text="5", variable=self.selected_value_ass_5_att_4, value=5).grid(row=12 + row_att3,column=6)
        Label(self.frame4, text="-----------------------------------------------------").grid(row=14+row_att2)

        Label(self.frame4, text="DONE?").grid(row=17+row_att3)
        Button(self.frame4, text="OK", command=self.get_value_ass_att).grid(row=18+row_att3)

        self.notebook.pack()
        Label(self.root, text="Total:").pack()
        #Label(self.root, textvariable=total).pack()

    def save(self):
        print(":",self.id.get())
        print(":", self.user_model.get())
        print(":", self.agent_model.get())

    def get_value_attempt(self):
        print(":", self.selected_value_att_1.get())
        print(":", self.selected_value_att_2.get())
        print(":", self.selected_value_att_3.get())
        print(":", self.selected_value_att_4.get())


    def get_value_game(self):
        print(":", self.selected_value_game_1.get())
        print(":", self.selected_value_game_2.get())
        print(":", self.selected_value_game_3.get())

    def get_value_user(self):
        print(":",self.selected_value_user_1.get())
        print(":",self.selected_value_user_2.get())
        print(":",self.selected_value_user_3.get())
        print(":",self.selected_value_user_4.get())
        print(":",self.selected_value_user_5.get())
        print(":",self.selected_value_user_6.get())

    def get_value_ass_att(self):
        print(":", self.selected_value_ass_0_att_1.get())
        print(":", self.selected_value_ass_0_att_2.get())
        print(":", self.selected_value_ass_0_att_3.get())
        print(":", self.selected_value_ass_0_att_4.get())

        print(":", self.selected_value_ass_1_att_1.get())
        print(":", self.selected_value_ass_1_att_2.get())
        print(":", self.selected_value_ass_1_att_3.get())
        print(":", self.selected_value_ass_1_att_4.get())

        print(":", self.selected_value_ass_2_att_1.get())
        print(":", self.selected_value_ass_2_att_2.get())
        print(":", self.selected_value_ass_2_att_3.get())
        print(":", self.selected_value_ass_2_att_4.get())

        print(":", self.selected_value_ass_3_att_1.get())
        print(":", self.selected_value_ass_3_att_2.get())
        print(":", self.selected_value_ass_3_att_3.get())
        print(":", self.selected_value_ass_3_att_4.get())

        print(":", self.selected_value_ass_4_att_1.get())
        print(":", self.selected_value_ass_4_att_2.get())
        print(":", self.selected_value_ass_4_att_3.get())
        print(":", self.selected_value_ass_4_att_4.get())

        print(":", self.selected_value_ass_5_att_1.get())
        print(":", self.selected_value_ass_5_att_2.get())
        print(":", self.selected_value_ass_5_att_3.get())
        print(":", self.selected_value_ass_5_att_4.get())

    def get_value_ass_1_att_1(self):
        pass

    def get_value_ass_2_att_1(self):
        pass

    def get_value_ass_3_att_1(self):
        pass

    def get_value_ass_4_att_1(self):
        pass

    def get_value_ass_5_att_1(self):
        pass
    ################################
    def get_value_ass_0_att_2(self):
        pass

    def get_value_ass_1_att_2(self):
        pass

    def get_value_ass_2_att_2(self):
        pass

    def get_value_ass_3_att_2(self):
        pass

    def get_value_ass_4_att_2(self):
        pass

    def get_value_ass_5_att_2(self):
        pass
    ################################
    def get_value_ass_0_att_3(self):
        pass

    def get_value_ass_1_att_3(self):
        pass

    def get_value_ass_2_att_3(self):
        pass

    def get_value_ass_3_att_3(self):
        pass

    def get_value_ass_4_att_3(self):
        pass

    def get_value_ass_5_att_3(self):
        pass
    #################################
    def get_value_ass_0_att_4(self):
        pass

    def get_value_ass_1_att_4(self):
        pass

    def get_value_ass_2_att_4(self):
        pass

    def get_value_ass_3_att_4(self):
        pass

    def get_value_ass_4_att_4(self):
        pass

    def get_value_ass_5_att_4(self):
        pass
    #################################


if "__main__" ==  __name__:
    root = Tk()
    notebook = ttk.Notebook(root)
    frame1 = ttk.Frame(notebook)
    frame2 = ttk.Frame(notebook)
    frame3 = ttk.Frame(notebook)
    frame4 = ttk.Frame(notebook)
    frame5 = ttk.Frame(notebook)
    gui = GUI(root, notebook, frame1, frame2, frame3, frame4, frame5)
    gui.root.mainloop()
