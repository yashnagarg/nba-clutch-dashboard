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

#vectorized way to compute fire aka hot streaks of players for faster computation than loops
def compute_fire(df:pd.DataFrame, hot_threshold: int=3)->pd.DataFrame:
    """
    hot_threshold: number of consecutive shots made to be considered a hot streak
                default is 3 for a hot streak of 3 or more consecutive shots made

    does not just measure streak length, also measures how frequently players enter a ‘hot state’, making the model more representative of real in-game momentum

    calculates FG streaks made consecutively by each player per game
    the streak resets on any miss 

    new columns added
    'made': boolean indicating whether the shot was made or not
    'streak_group': a unique identifier for each streak of made shots by a player in a game between two misses
    'heat_streak_length': the length of the current streak of made shots for each shot
    'is_hot': boolean indicating whether the current shot is part of a hot streak (streak length >= hot_threshold)

    """
    fg_df=df[df['is_FG']].copy()
    fg_df=fg_df.sort_values(
        by=['teamId','playerId','period','time_remaining'],
        ascending=[True,True,True,False])
    fg_df['made']=fg_df['shotResult'].str.lower()=='made'

    #vectorized way to compute streaks
    #increments everytime there is a miss (~x) and cumsum to get unique streak groups for each player in each game
    fg_df['streak_group']=(
        fg_df.groupby(['gameId','playerId'])['made']
        .transform(lambda x: (~x).cumsum()))
    
    #calculates cumulative sum(length) of made shots in each streak group 
    fg_df['heat_streak_length']=(fg_df.
                                 groupby(['gameId','playerId','streak_group'])['made']
                                 .transform('cumsum'))
    
    fg_df['is_hot']=fg_df['heat_streak_length']>=hot_threshold

    return fg_df

def compute_fire_score(df:pd.DataFrame, fg_df:pd.DataFrame)->pd.DataFrame:
    """
    combines clutch performance from compute_clutch and hot streaks from compute_fire to calculate a normalised 'fire_score'(0.0-1.0)

    components of the fire score(all normalised to 0.0-1.0):
    -clutch_points: points scored in clutch situations (last 5 minutes of the game and score difference <= 5 points)
    -clutch_fg_pct: field goal percentage in clutch situations 
    -avg_heat: avg streak length of hot streaks for each player in each game(how hot a player gets)
    -hot_rate: fraction of fg attempts while 'is_hot' (how often/consistently a player gets hot)

    shot difficulty or complexity is not taken into account in this model,
    could be added in the future by incorporating features like shot distance, defender proximity, etc. to further refine the fire score
     + make it more representative of a player's performance under pressure.
    """
    clutch_df=df[df['clutch'] & df['is_FG']]

    #aggregate clutch points and attempts for each player under pressure/clutch situations
    clutch_stats=clutch_df.groupby('playerId').agg(
        clutch_points=('points','sum'),
        clutch_attempts=('is_FG','count'))
    
    #taking into account only successful clutch shots for the FG% 
    made_clutch=clutch_df[clutch_df['shotResult'].str.lower()=='made']
    #gives us the count of the total clutch shots made by each player
    clutch_made=made_clutch.groupby('playerId').size().rename('clutch_made')

    #calculating the clutch FG% for each player
    clutch_stats=clutch_stats.join(clutch_made, on='playerId', how='left').fillna(0)
    clutch_stats['clutch_fg_pct']=(clutch_stats['clutch_made']/clutch_stats['clutch_attempts'].replace(0, np.nan)).round(3) #avoid division by zero

    #aggregate hot streak stats for each player
    heat_stats=fg_df.groupby('playerId').agg(
        avg_heat=('heat_streak_length','mean'),
        max_heat=('heat_streak_length','max'),
        hot_rate=('is_hot','mean')
    )

    #combine clutch and hot streak stats into a single dataframe
    combined=clutch_stats.join(heat_stats,on='playerId',how='inner')

    #normalising the components to a 0.0-1.0 scale
    components=['clutch_points','clutch_fg_pct','avg_heat','hot_rate']

    for column in components:
        column_min=combined[column].min()
        column_max=combined[column].max()
        if column_max>column_min:
            combined[f'{column}_norm']=(combined[column]-column_min)/(column_max-column_min)
        else:
            combined[f'{column}_norm']=0.0  #all values are same thus 0.0 normalisation

    #calculating fire score = average of the normalised components
    combined['fire_score']=combined[[f'{col}_norm' for col in components]].mean(axis=1).round(3)

    #highest fire score on top thus descending order
    return combined.sort_values(by='fire_score', ascending=False)






