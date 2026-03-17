import pandas as pd

#loading the csv file
def load_data(path='datasets/nbastatsv3_2025.csv'):
    df=pd.read_csv(path)
    return df

#converting clock in csv file to seconds
def parse_clock(df):
    #format of the clock is PT11M26.00S
    df['minutes']=df['clock'].str.extract(r'PT(\d+)M').astype(float)
    df['seconds']=df['clock'].str.extract(r'M(\d+\.\d+)S').astype(float)
    #did  26.00 seconds instead of 26 seconds for greater precision later on when calculating clutch shots for end minute
    df['time_remaining']=(df['minutes']*60)+df['seconds']
    return df

#adding shot type because freethrows are not considered fieldgoals and are imp for clutch shots as well
def add_shot_type(df):
    df['is_FG']=df['isFieldGoal']==1 #get 2pt/3pt shot
    df['is_FT']=df['actionType']=='Free Throw' #get 1pt shot
    df['points']=0
    df.loc[df['is_FG'],'points']=df.loc[df['is_FG'],'shotValue'] #assign points for field goals
    df.loc[df['is_FT'],'points']=1 #assign points for free throws
    return df

#calculating clutch shot aka in the last 5 minutes of the game and the score difference is <= 5 points
def computer_clutch(df):
    return df
    