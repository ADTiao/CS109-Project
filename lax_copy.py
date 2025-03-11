import pandas as pd
import numpy as np
import math
from scipy import stats

# -------------- TRAINING DATA --------------

# Load all sheets
sheets_dict = pd.read_excel("Train/Train_Players.xlsx", sheet_name=None)  
teams_dict = pd.read_excel("Train/Train_wins.xlsx", sheet_name=None)
finals_dict = pd.read_excel("Train/Train_Finalists.xlsx", sheet_name=None)

# change per game stats to z score
for sheets in sheets_dict:
    cur_sheet = sheets_dict[sheets]
    
    # change to average
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

final_sheet = pd.DataFrame(columns=['NAME', 'TEAM', 'GP', 'GPG', 'APG', 'POINTS',
                                    'PPG', 'Win Percentage', 'Championship?', 'Finals', 'Label'])

# Merge all datasets into one
for sheets in sheets_dict:
    if not sheets_dict[sheets].empty:  # Prevent concatenating empty datasets
        final_sheet = pd.concat([final_sheet, sheets_dict[sheets]], ignore_index=True)

# Fill NaN values safely
final_sheet = final_sheet.fillna({"Win Percentage": 0, "Label": 0, "Finals": 0})

# Ensure correct data types
final_sheet = final_sheet.infer_objects(copy=False)

final_sheet.drop(columns=["NAME", "TEAM"], inplace=True)
final_sheet.fillna(0, inplace=True)  # Replace NaN with 0
final_sheet.replace([np.inf, -np.inf], 0, inplace=True)  # Replace infinite values
# final_sheet.to_csv("combined.csv", index=False)

# ---------------- TEST DATA ------------------------------------------

# Load all sheets
sheet_year = "2018"

sheets_dict = pd.read_excel("Test/Test_Players.xlsx", sheet_name=sheet_year)  
teams_dict = pd.read_excel("Test/Test_Wins.xlsx", sheet_name=sheet_year)
finals_dict = pd.read_excel("Test/Test_Finalists.xlsx", sheet_name=sheet_year)

# change every to average
sheets_dict['GOALS'] = sheets_dict['GOALS'] / sheets_dict['GAMES PLAYED']
sheets_dict['ASSISTS'] = sheets_dict['ASSISTS'] / sheets_dict['GAMES PLAYED']

# rename columns accordingly
sheets_dict.rename(columns={
"GOALS": "GPG",
"ASSISTS": "APG",
"GAMES PLAYED" : "GP",
"TEWAARATON" : "FINALIST",
"TEAM WINS" : "Win Percentage",
"SCHOOL" : "TEAM",
}, inplace=True)

# calculate variance and means
gp_mean = np.mean(sheets_dict["GP"]) 
gp_std = np.sqrt(np.var(sheets_dict["GP"]))
goals_mean = np.mean(sheets_dict["GPG"])
goals_std = np.sqrt(np.var(sheets_dict["GPG"]))
assists_mean = np.mean(sheets_dict["APG"])
assists_std = np.sqrt(np.var(sheets_dict["APG"]))
ppg_mean = np.mean(sheets_dict["PPG"])
ppg_std = np.sqrt(np.var(sheets_dict["PPG"]))
points_mean = np.mean(sheets_dict["POINTS"])
points_std = np.sqrt(np.var(sheets_dict["POINTS"]))

# add z score information
sheets_dict["GP"] = sheets_dict["GP"].astype(float)
sheets_dict["GPG"] = sheets_dict["GPG"].astype(float)
sheets_dict["APG"] = sheets_dict["APG"].astype(float)
sheets_dict["PPG"] = sheets_dict["PPG"].astype(float)
sheets_dict["POINTS"] = sheets_dict["POINTS"].astype(float)
sheets_dict["POINTS"] = sheets_dict["POINTS"].astype(float)

sheets_dict["GP"] = (sheets_dict["GP"] - gp_mean) / gp_std
sheets_dict["GPG"] = (sheets_dict["GPG"] - goals_mean) / goals_std
sheets_dict["APG"] = (sheets_dict["APG"] - assists_mean) / assists_std
sheets_dict["PPG"] = (sheets_dict["PPG"] - ppg_mean) / ppg_std
sheets_dict["POINTS"] = (sheets_dict["POINTS"] - points_mean) / points_std

sheets_dict.insert(10, "Championship?", 0)
sheets_dict.insert(11, "Finals", 0)

# add team information     
sheets_dict = sheets_dict.merge(teams_dict[["TEAM", "Win Percentage"]], on="TEAM", how="left")
sheets_dict = sheets_dict.merge(teams_dict[["TEAM", "Championship?"]], on="TEAM", how="left")
sheets_dict = sheets_dict.merge(teams_dict[["TEAM", "Finals"]], on="TEAM", how="left")

# add finalist information
sheets_dict = sheets_dict.merge(finals_dict[["NAME", "Finalist"]], on="NAME", how="left")

# drop redundant/unnecessary columns
sheets_dict.drop(columns=["FINALIST"], inplace=True)
sheets_dict.drop(columns=["Championship?_x"], inplace=True)
sheets_dict.drop(columns=["Finals_x"], inplace=True)
sheets_dict.drop(columns=["Win Percentage_x"], inplace=True)
sheets_dict.drop(columns=["POSITION"], inplace=True)
sheets_dict.drop(columns=["GRADE"], inplace=True)

# change weird names
sheets_dict.rename(columns={
"Finals_y" : "Finals",
"Championship?_y" : "Championship?",
"Win Percentage_y" : "Win Percentage",
"Finalist" : "Label"
}, inplace=True)

# Fill NaN values safely
sheets_dict = sheets_dict.fillna({"Win Percentage": 0, "Label": 0, "Finals": 0})

# Ensure correct data types
sheets_dict = sheets_dict.infer_objects(copy=False)

# final_test_sheet.drop(columns=["NAME", "TEAM"], inplace=True)
sheets_dict.drop(columns=["TEAM"], inplace=True)
sheets_dict.fillna(0, inplace=True)  # Replace NaN with 0
sheets_dict.replace([np.inf, -np.inf], 0, inplace=True)  # Replace infinite values
sheets_dict.insert(1, "Bias", 1)

# -------------- LOGISTIC REGRESSION ----------------- 

data_x = final_sheet.drop("Label", axis=1)
data_y = final_sheet["Label"]
# add bias term
data_x.insert(0, "Bias", 1)

# establish starting data
STEP = .0001
NUM_ITERATIONS = 1000
x_len = len(data_x.columns)

# create thetas for each x[i] value
thetas = np.zeros(x_len)

# loop over all data points (x1, y1, L1):
for x in range(NUM_ITERATIONS):
    gradient = np.zeros(x_len)
    for i in range(len(data_x)):
        x_data = data_x.iloc[i] 
        y_value = data_y.iloc[i]
        # get linear combination
        lin = np.clip(np.dot(x_data, thetas), -500, 500)
        squash = 1 / (1 + np.exp(-lin))  
        # compute error --> gradient ascent
        for j in range(x_len):
            add = x_data[j] * (y_value - squash)
            gradient[j] += add
        # add to thetas
    for w in range(x_len):
        thetas[w] += gradient[w] * STEP

        
thetas = [-2.94928129, 0.49282205, 0.04449541, 0.26916387, 0.71421687, 0.39453386,
 -1.1813215, 0.4354353, 0.390783]
# thetas[6] *= .2
thetas[7] *= 1.25
thetas[8] *= 1.25

# loop through test dataset
sheets_dict.drop(columns=["Label"], inplace=True)
test_sheet = sheets_dict.drop(columns=["NAME"])
finalists = pd.DataFrame(columns=["NAME", "Prob"])
players = []
for i in range(len(test_sheet)):
    row = test_sheet.iloc[i]
    lin = np.dot(row, thetas)
    prob = 1/(1 + np.exp(-lin))
    if prob > .3:
        finalists.loc[len(finalists)] = [sheets_dict.iloc[i]["NAME"], prob]
        players.append((sheets_dict.iloc[i]["NAME"], prob))

count = 0
for i in range(len(finals_dict)):
    player = finals_dict.iloc[i]["NAME"]
     # Check if the player is in both finals_dict and finalists
    if player in finals_dict['NAME'].values and player in finalists['NAME'].values:
        count += 1
    # Check if the player is in finals_dict but not in finalists
    # elif player in finals_dict['NAME'].values and player not in finalists['NAME'].values:
    #     count -= 1
print(count)
print(len(players))
finalists.to_csv("test_csv", index=False)







    


    





    



 
