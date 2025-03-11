import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import math
from scipy import stats

THETAS = [-2.94928129, 0.49282205, 0.04449541, 0.26916387, 0.71421687, 0.39453386,
 -1.1813215, 0.4354353, 0.390783]
# THETAS[6] *= -1.25
# THETAS[7] *= 1.25
# THETAS[8] *= 1.25

debug = False

# function to load data
def load_data(player_training, team_training, finalist_training):

    # read files  
    sheets_dict = pd.read_excel(player_training, sheet_name=None)  
    teams_dict = pd.read_excel(team_training, sheet_name=None)
    finals_dict = pd.read_excel(finalist_training, sheet_name=None)

    # change per game stats to z score
    for sheets in sheets_dict:
        cur_sheet = sheets_dict[sheets]
        
        # change every to average
        cur_sheet['GOALS'] = cur_sheet['GOALS'] / cur_sheet['GAMES PLAYED']
        cur_sheet['ASSISTS'] = cur_sheet['ASSISTS'] / cur_sheet['GAMES PLAYED']

        # rename columns accordingly
        cur_sheet.rename(columns={
        "GOALS": "GPG",
        "ASSISTS": "APG",
        "GAMES PLAYED" : "GP",
        "TEWAARATON" : "FINALIST",
        "TEAM WINS" : "Win Percentage",
        "SCHOOL" : "TEAM",
        }, inplace=True)
        
        # calculate variance and means
        gp_mean = np.mean(cur_sheet["GP"]) 
        gp_std = np.sqrt(np.var(cur_sheet["GP"]))
        goals_mean = np.mean(cur_sheet["GPG"])
        goals_std = np.sqrt(np.var(cur_sheet["GPG"]))
        assists_mean = np.mean(cur_sheet["APG"])
        assists_std = np.sqrt(np.var(cur_sheet["APG"]))
        ppg_mean = np.mean(cur_sheet["PPG"])
        ppg_std = np.sqrt(np.var(cur_sheet["PPG"]))
        points_mean = np.mean(cur_sheet["PPG"])
        points_std = np.sqrt(np.var(cur_sheet["PPG"]))
        points_mean = np.mean(cur_sheet["POINTS"])
        points_std = np.sqrt(np.var(cur_sheet["POINTS"]))
        
        # add z score information
        cur_sheet["GP"] = cur_sheet["GP"].astype(float)
        cur_sheet["GPG"] = cur_sheet["GPG"].astype(float)
        cur_sheet["APG"] = cur_sheet["APG"].astype(float)
        cur_sheet["PPG"] = cur_sheet["PPG"].astype(float)
        cur_sheet["POINTS"] = cur_sheet["POINTS"].astype(float)
        cur_sheet["POINTS"] = cur_sheet["POINTS"].astype(float)

        cur_sheet["GP"] = (cur_sheet["GP"] - gp_mean) / gp_std
        cur_sheet["GPG"] = (cur_sheet["GPG"] - goals_mean) / goals_std
        cur_sheet["APG"] = (cur_sheet["APG"] - assists_mean) / assists_std
        cur_sheet["PPG"] = (cur_sheet["PPG"] - ppg_mean) / ppg_std
        cur_sheet["POINTS"] = (cur_sheet["POINTS"] - points_mean) / points_std

        cur_sheet.insert(10, "Championship?", 0)
        cur_sheet.insert(11, "Finals", 0)

    for year in teams_dict:
        # add team information     
        team_sheet = teams_dict[year]
        sheets_dict[year] = sheets_dict[year].merge(team_sheet[["TEAM", "Win Percentage"]], on="TEAM", how="left")
        sheets_dict[year] = sheets_dict[year].merge(team_sheet[["TEAM", "Championship?"]], on="TEAM", how="left")
        sheets_dict[year] = sheets_dict[year].merge(team_sheet[["TEAM", "Finals"]], on="TEAM", how="left")

        # add finalist information
        finals_sheet = finals_dict[year]
        sheets_dict[year] = sheets_dict[year].merge(finals_sheet[["NAME", "Finalist"]], on="NAME", how="left")

        # drop redundant/unnecessary columns
        sheets_dict[year].drop(columns=["FINALIST"], inplace=True)
        sheets_dict[year].drop(columns=["Championship?_x"], inplace=True)
        sheets_dict[year].drop(columns=["Finals_x"], inplace=True)
        sheets_dict[year].drop(columns=["Win Percentage_x"], inplace=True)
        sheets_dict[year].drop(columns=["POSITION"], inplace=True)
        sheets_dict[year].drop(columns=["GRADE"], inplace=True)

        # change weird names
        sheets_dict[year].rename(columns={
        "Finals_y" : "Finals",
        "Championship?_y" : "Championship?",
        "Win Percentage_y" : "Win Percentage",
        "Finalist" : "Label"
        }, inplace=True)

    final_test_sheet = pd.DataFrame(columns=['NAME', 'TEAM', 'GP', 'GPG', 'APG', 'POINTS',
    'PPG', 'Win Percentage', 'Championship?', 'Finals', 'Label'])

    # Merge all datasets into one
    for sheets in sheets_dict:
        if not sheets_dict[sheets].empty:  # Prevent concatenating empty datasets
            final_test_sheet = pd.concat([final_test_sheet, sheets_dict[sheets]], ignore_index=True)

    # Fill NaN values safely
    final_test_sheet = final_test_sheet.fillna({"Win Percentage": 0, "Label": 0, "Finals": 0})

    # Ensure correct data types
    final_test_sheet = final_test_sheet.infer_objects(copy=False)

    # final_test_sheet.drop(columns=["NAME", "TEAM"], inplace=True)
    final_test_sheet.drop(columns=["TEAM"], inplace=True)
    final_test_sheet.fillna(0, inplace=True)  # Replace NaN with 0
    final_test_sheet.replace([np.inf, -np.inf], 0, inplace=True)  # Replace infinite values
    final_test_sheet.insert(1, "Bias", 1)
        
    # return DataFrame
    return final_test_sheet

def predict_data(final_sheet):
    final_sheet.drop(columns=["Label"], inplace=True)
    test_sheet = final_sheet.drop(columns=["NAME"])
    finalists = pd.DataFrame(columns=["NAME", "Prob"])
    players = []
    for i in range(len(test_sheet)):
        row = test_sheet.iloc[i]
        lin = np.dot(row, THETAS)
        prob = 1/(1 + np.exp(-lin))
        finalists.loc[len(finalists)] = [final_sheet.iloc[i]["NAME"], prob]
        players.append((final_sheet.iloc[i]["NAME"], prob))
    
    # downloads dataset
    finalists.to_csv("test_csv", index=False)
    
    # returns the list of Name, Prob
    return players

def find_player(players, player_name):
    for player in players:
        if player[0] == player_name:
            return player
    return 0

def similar_players(player_list, player_prob, player_name):
    similar = []
    for player in player_list:
        if player[1] >= player_prob:
            if player[0] != player_name:
                prob = round(player[1] * 100,  2)
                similar.append((player[0], prob))
    return similar

# user interface
def main():
    print("Hello there! This is the Tewaaraton Finalist Prediction Model.\n") 
    print("Motivated by the lack of data analytics in lacrosse, I made this program to give a probability of a College Lacrosse Player being named a Tewaaraton Finalist, the highest achievement in lacrosse aside from winning the award itself!")
    print("All you have to do is give the program some files and a players name and SHABOOYA! The players chance of winning the award appears before your eyes.")
    print("But no need for more words, lets get to the probabilities!\n")

    print("Please enter some file path in the prompts below\n")
    player_data = "Test/Test_Players.xlsx"
    team_data = "Test/Test_Wins.xlsx"
    finalist_data = "Test/Test_Finalists.xlsx"
    print("We now have ALL the probabilities for every player in that dataset --> thanks a ton!\n")

    if not debug:
        player_data = input("Enter player data file path here: ")
        team_data = input("Enter team win data file path here: ")
        finalist_data = input("Enter finalist data file path here: ")
    
    print("Thank you!")

    while True:
        
        # run load data function with those inputs
        final_sheet = load_data(player_data, team_data, finalist_data)
        # returns list of players and probabilities from dataset
        probs = predict_data(final_sheet)

        player_name = input("\nPlease type a player's name that you would like to analyze (leave empty to quit): ")

        if player_name == "":
            break

        player_info = find_player(probs, player_name)
        if player_info == 0:
            print("That player either has no chance of winning or is not real. Please try again.\n")
        else:
            player_prob = player_info[1] * 100
            print(f"\n{player_name} has a {player_prob:.2f}% chance of becoming a Tewaaraton Finalist!")
            similars = similar_players(probs, player_prob/100, player_name)
            if len(similars) == 0:
                print(f"{player_name} has the highest chance to become a Tewaaraton Finalist!")
            else: 
                print(f"Here are some other players with a similar or higher chance of becoming a finalist.\n {similars}\n")
    
if __name__ == "__main__":
    main() 



# player_data = "Test/Test_Players.xlsx"
# team_data = "Test/Test_Wins.xlsx"
# finalist_data = "Test/Test_Finalists.xlsx"