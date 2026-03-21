import pandas as pd
import numpy as np

#loading the csv file
def load_data(path='datasets/nbastatsv3_2025.csv'):
    df=pd.read_csv(path)
    return df

#converting clock in csv file to seconds
#using pd.DataFrame helps debug better
def parse_clock(df: pd.DataFrame)->pd.DataFrame:
    """
    we add a new column 'time_remaining' to the dataframe which is the time remaining in seconds for each shot
    also dropping rows with missing time_remaining values to clean the data
    """
    #format of the clock is PT11M26.00S
    df=df.copy() #to avoid modifying the original dataframe
    df['minutes']=df['clock'].str.extract(r'PT(\d+)M').astype(float)
    df['seconds']=df['clock'].str.extract(r'M(\d+\.\d+)S').astype(float)
    #did  26.00 seconds instead of 26 seconds for greater precision later on when calculating clutch shots for end minute
    df['time_remaining']=(df['minutes']*60)+df['seconds']

    num_missing_time_remaining=df['time_remaining'].isna().sum()
    if num_missing_time_remaining>0:
        print(f"Warning: {num_missing_time_remaining} rows have missing time_remaining values. These rows will be dropped.")
        df=df.dropna(subset=['time_remaining']) #cleaning the data by dropping rows with missing time_remaining values
    return df

#adding shot type because freethrows are not considered fieldgoals and are imp for clutch shots as well
def add_shot_type(df:pd.DataFrame)-> pd.DataFrame:
    """
    this function adds a new column 'points' to the dataframe which indicates the points scored for each shot
    this is because free throws are not considered field goals and are important for clutch shots as well.
    """
    df=df.copy()
    df['is_FG']=df['isFieldGoal']==1 #get 2pt/3pt shot
    df['is_FT']=df['actionType']=='Free Throw' #get 1pt shot
    df['points']=0
    df.loc[df['is_FG'],'points']=df.loc[df['is_FG'],'shotValue'] #assign points for field goals
    df.loc[df['is_FT'],'points']=1 #assign points for free throws
    return df

#calculating clutch shot aka in the last 5 minutes of the game and the score difference is <= 5 points
def compute_clutch(df: pd.DataFrame, time_threshold: float=300, margin_threshold: float=5)->pd.DataFrame:
    """
    time_threshold: time in seconds, default is 300 seconds (5 minutes) for the last 5 minutes of the game in 4th quarter or OT
    margin_threshold: point difference, default is 5 points for the score difference between the two teams
    
    the thresholds can be adjusted based on the definition of a clutch shot, but the default values are commonly used in basketball analytics.

    this function adds a new column 'clutch' and 'score_difference' to the dataframe
    'clutch' is a boolean indicating whether the shot is a clutch shot or not
    'score_difference' is the absolute difference in score between the two teams at the time of the shot.
    """
    df=df.copy()

    df['score_difference']=(df['scoreHome']-df['scoreAway']).abs() #score difference between the two teams
    clutch_time=df['period'].isin([4,5]) & (df['time_remaining']<=time_threshold) #last 5 minutes of the game in 4th quarter or OT
    clutch_score=df['score_difference']<=margin_threshold #score difference is less than or equal to 5 points
    df['clutch']=clutch_time & clutch_score #shot is a clutch shot if within clutch time and clutch score difference
    return df


    