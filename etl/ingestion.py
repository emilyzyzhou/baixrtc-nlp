import pandas as pd
import os

'''Extracts each sheet from source into df, returned as list of dfs'''
def load_data(folder = "data"):
    dfs = []
    #sheets w long names will only match substring of length ~30
    sheets_to_exclude = ["large company list (accumulated", "keywords", "ideal tableau format"]

    #loops through xlsx files in data folder
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        #handle xlsx files
        if file.lower().endswith(".xlsx"):
            #get sheets w data
            data = pd.ExcelFile(file_path)
            #print(data.sheet_names)
            sheets_to_read = [sheet for sheet in data.sheet_names if sheet not in sheets_to_exclude]
            #create df for sheet + track which sheet it came from
            for sheet_name in sheets_to_read:
                df = pd.read_excel(data, sheet_name=sheet_name) #allow nan .fillna("") #prevent na + type float64
                df["sheet_name"] = sheet_name
                dfs.append(df)
            data.close()
        
        #handle csv files
        elif file.lower().endswith(".csv"):
            df = pd.read_csv(file_path).fillna("")
            df["sheet_name"] = os.path.splitext(file)[0] #use file name as sheet name
            dfs.append(df)

    return dfs


if __name__ == "__main__":
    load_data()