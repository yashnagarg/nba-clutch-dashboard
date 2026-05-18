import os
import pandas as pd
import numpy as np
BASE_DIR=os.path.dirname(os.path.abspath(__file__))

#loading the csv file
def load_data(path='backend/datasets/nbastatsv3_2025.csv'):
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
        by=['gameId','personId','period','time_remaining'],
        ascending=[True,True,True,False])
    fg_df['made']=fg_df['shotResult'].str.lower()=='made'

    #vectorized way to compute streaks
    #increments everytime there is a miss (~x) and cumsum to get unique streak groups for each player in each game
    fg_df['streak_group']=(
        fg_df.groupby(['gameId','personId'])['made']
        .transform(lambda x: (~x).cumsum()))
    
    #calculates cumulative sum(length) of made shots in each streak group 
    fg_df['heat_streak_length']=(fg_df.
                                 groupby(['gameId','personId','streak_group'])['made']
                                 .transform('cumsum'))
    
    fg_df['is_hot']=fg_df['heat_streak_length']>=hot_threshold

    return fg_df

def compute_fire_score(df:pd.DataFrame, fg_df:pd.DataFrame)->pd.DataFrame:
    """
    combines clutch performance + hot behaviour to calculate a normalised 'fire_score'(0.0-1.0)
    This answers: who gets hot and stays hot when the game is actually on the line?(during clutch situations)

    components of the fire score(all normalised to 0.0-1.0):
    -clutch_points: points scored in clutch situations (last 5 minutes of the game and score difference <= 5 points)
    -clutch_fg_pct: field goal percentage in clutch situations 
    -clutch_avg_streak: avg streak length of hot streaks for each player in each game(how hot a player gets) during clutch situations
    -clutch_hot_rate: fraction of fg attempts while 'is_hot' (how often/consistently a player gets hot) during CLUTCH situations 

    shot difficulty or complexity is not taken into account in this model,
    could be added in the future by incorporating features like shot distance, defender proximity, etc. to further refine the fire score
     + make it more representative of a player's performance under pressure.

    'is_hot' and 'heat_streak_length' are calculated for all shots in the compute_fire function,
    but for the fire score we will only consider the hot streaks during clutch situations to better capture real performance under pressure.
    - they were calculated for all shots to get an insight to the player's momentum throughout the game which carries into clutch situations
    """
    
    clutch_fg=df[df['clutch'] & df['is_FG']][['actionId','gameId','personId','shotResult','is_FG','points']].merge(fg_df[['actionId','gameId','personId','made','heat_streak_length','is_hot']], on=['actionId','gameId','personId'], how='left')
    clutch_fg['is_hot']=clutch_fg['is_hot'].fillna(False) #shots that are not field goals will have NaN for is_hot, filling them with False
    clutch_fg['heat_streak_length']=clutch_fg['heat_streak_length'].fillna(0).astype(int) #shots that are not field goals will have NaN for heat_streak_length, filling them with 0 and converting to int 
      
    #aggregate clutch points and attempts for each player under pressure/clutch situations
    clutch_stats=clutch_fg.groupby('personId').agg(
        clutch_points=('points','sum'),
        clutch_attempts=('is_FG','count'),
        clutch_made_count=('made','sum'),
        #counted shots were hot during clutch situations
        clutch_and_hot=('is_hot','sum'),
        #% of shots that were hot during clutch situations out of total clutch shots attempted by each player
        clutch_hot_rate=('is_hot','mean'),
        #avg streak length during clutch situations for each player
        clutch_avg_streak=('heat_streak_length','mean'),
        #max streak length during clutch situations for each player
        clutch_max_streak=('heat_streak_length','max')
        )
    
    #taking into account only successful clutch shots for the FG% 
    #made_clutch=clutch_df[clutch_df['shotResult'].str.lower()=='made']
    #gives us the count of the total clutch shots made by each player
    #clutch_made=made_clutch.groupby('personId').size().rename('clutch_made')

    #calculating the clutch FG% for each player
    #clutch_stats=clutch_stats.join(clutch_made, on='personId', how='left').fillna(0)
    

    clutch_stats['clutch_fg_pct']=(clutch_stats['clutch_made_count']/clutch_stats['clutch_attempts'].replace(0, np.nan)).round(3) #avoid division by zero

    
    #aggregate hot streak stats for each player
    overall_heat_stats=fg_df.groupby('personId').agg(
        overall_avg_streak=('heat_streak_length','mean'),
        overall_max_streak=('heat_streak_length','max'),
        overall_hot_rate=('is_hot','mean')
    )

    #combine clutch and hot streak stats into a single dataframe
    combined=clutch_stats.join(overall_heat_stats,on='personId',how='inner')

    #mapping player id to player name, taken the playerNameI column to get the initial for the first name as well for better readability
    personId_to_name=df[['personId','playerNameI']].drop_duplicates().set_index('personId')['playerNameI']

    combined['playerName']=combined.index.map(personId_to_name)
    MIN_CLUTCH_ATTEMPTS = 22  # tuned this according to my distribution
    
    combined = combined[combined['clutch_attempts'] >= MIN_CLUTCH_ATTEMPTS].copy()
    #normalising the components to a 0.0-1.0 scale
    components=['clutch_points','clutch_fg_pct','clutch_hot_rate','clutch_avg_streak']

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

def get_shot_chart_data(df: pd.DataFrame, fg_df: pd.DataFrame,personId:int)->pd.DataFrame:
    """
    returns all FG attempts for a given personId with the necessary columns for shot chart visualisation
    
    -x and y coordinates for shot location
    -hot state info for each shot (is_hot and heat_streak_length) to visualise hot streaks on the shot chart merged with the base shot info
    
    """

    base_columns=[ 'gameId', 'actionId','xLegacy','yLegacy','shotResult', 'shotDistance', 'subType', 'clutch', 'period','time_remaining','score_difference','points']
    #rows of all fg attempts of the given player
    base=df[(df['personId']==personId) & df['is_FG']][base_columns].copy()

    #merging the base info with the hot state info
    hot_columns=fg_df[fg_df['personId']==personId][['gameId','actionId','is_hot','made','heat_streak_length']]

    #actionId and gameId will be used as the unique identifiers for each shot to merge the two dataframes
    merged_shot_chart_data=base.merge(hot_columns,on=['gameId','actionId'],how='left')

    #shots that are not field goals will have NaN for is_hot, filling them with False
    merged_shot_chart_data['is_hot']=merged_shot_chart_data['is_hot'].fillna(False) 

    #shots that are not field goals will have NaN for heat_streak_length, filling them with 0 and converting to int for better readability in the shot chart
    merged_shot_chart_data['heat_streak_length']=merged_shot_chart_data['heat_streak_length'].fillna(0).astype(int)
    #already made sure for 'made' column before so it has only True/False values

    #adding a column to indicate shots that are both clutch and hot for better visualisation on the shot chart
    merged_shot_chart_data['clutch_and_hot']=merged_shot_chart_data['clutch'] & merged_shot_chart_data['is_hot'] 
    return merged_shot_chart_data

def streak_hot_rate_per_game(fg_df:pd.DataFrame,df: pd.DataFrame,personId:int)->pd.DataFrame:
    """
        returns overall vs clutch hot streak performance for each game for a given playerId to visualise the relationship between hot streaks and clutch performance across different games
    """
    player_fg=fg_df[fg_df['personId']==personId].copy()

    clutch_flags=df[(df['personId']==personId) & (df['is_FG'])][['gameId','actionId','clutch']]

    player_fg=player_fg.merge(clutch_flags,on=['gameId','actionId'],how='left')

    
    player_fg['clutch_present']=player_fg['clutch'].notna()
    player_fg['clutch']=player_fg['clutch'].fillna(False)

    overall=player_fg.groupby('gameId').agg(
        max_streak=('heat_streak_length','max'),
        overall_hot_rate=('is_hot','mean'),
        clutch_attempts=('clutch_present','sum')
    )

    clutch_only=player_fg[player_fg['clutch']].groupby('gameId').agg(
        clutch_max_streak=('heat_streak_length','max'),
        clutch_hot_rate=('is_hot','mean'),
        clutch_fgattempts=('clutch','size')
    )

    result=(overall.join(clutch_only,how='left').fillna({'clutch_max_streak':0,'clutch_hot_rate':0.0,'clutch_fgattempts':0}).round({'overall_hot_rate':3,'clutch_hot_rate':3}).reset_index())

    return result

def load_data(path=None):
    if path is None:
        path=os.path.join(BASE_DIR,'datasets','nbastatsv3_2025.csv')

    df=pd.read_csv(path)
    #needed to fix the missing values in scorehome and scoreawaycolumns where shot was not made, ex, rebound, missed,etc. 
    #this correction is required for an accurate compute_clutch func as otherwise we get clutch shots as always made leading to an inaccurate clutch fg performance(%)
    df['scoreHome']=(pd.to_numeric(df['scoreHome'],errors='coerce').groupby(df['gameId']).transform(lambda x: x.ffill().fillna(0)))

    df['scoreAway']=(pd.to_numeric(df['scoreAway'],errors='coerce').groupby(df['gameId']).transform(lambda x: x.ffill().fillna(0)))
    return df

def process_data(path=None):
    """
    pipeline:
    -load csv
    -parse clock to get time remaining in seconds
    -add shot type to differentiate between field goals and free throws for better clutch shot analysis + assign points for each shot
    -flag clutch situations
    -compute the fire streaks and hot state
    """
    df=load_data(path)
    df=parse_clock(df)
    df=add_shot_type(df)
    df=compute_clutch(df)
    fg_df=compute_fire(df)
    return df, fg_df

if __name__=="__main__":
    print("Processing data...")
    df, fg_df = process_data()

    print(f"\nFull data shape: {df.shape}")
    print(f"Field goal data shape: {fg_df.shape}")
    print(f"Total Clutch shots: {df['clutch'].sum()}")
    print(f"Total hot shots: {fg_df['is_hot'].sum()}")

    print("\nCalculating fire scores...")
    rankings=compute_fire_score(df, fg_df)
    # Run this first to see the distribution
    #print(rankings['clutch_attempts'].describe())
    #print(rankings['clutch_attempts'].value_counts().sort_index().head(20))
    print("\nTop 10 fire scores:")
    print(rankings[['playerName','fire_score','clutch_points','clutch_fg_pct','clutch_hot_rate','clutch_and_hot','overall_hot_rate']].head(10).to_string())

    print("\nSample data, high clutch_hot_rate + high clutch_points=real pressure performer")