from fastapi import FastAPI,HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from numpy import argmax
from main import process_data, compute_fire_score, get_shot_chart_data, streak_hot_rate_per_game

app = FastAPI(title="NBA Clutch + Heat Check API", description="API for NBA Clutch and Heat Check data processing", version="1.0.0")
origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET"], allow_headers=["*"])

print("Loading the NBA 24-25 season data...")
df, fg_df=process_data()

#ranking so we can filter out players instead of calculating it every time for frontend
rankings_full=compute_fire_score(df,fg_df,min_attempts=0)

id_to_name=df[['personId','playerNameI']].drop_duplicates().set_index('personId')['playerNameI'].to_dict()
print(f"{len(id_to_name)} players loaded.")

#get all players for frontend dropdown
@app.get("/players")
def get_players():
    return [{"personId":playerid,"playerNameI":name} for playerid,name in sorted(id_to_name.items(),key=lambda x:x[1]) 
            if name and str(name).strip()]

@app.get("/leaderboard")
def get_leaderboard(limit: int=Query(default=10,ge=1,le=20),min_attempts:int=Query(default=15,ge=0,le=60)):
    """returns the leaderboard of top n players based on their fire score, filtered by minimum clutch attempts
    min clutch attempts can be changed using a slider by the user"""
    filtered_rankings=rankings_full[rankings_full['clutch_attempts']>=min_attempts]
    top=filtered_rankings.head(limit).reset_index()
    cols=['personId','playerName','fire_score','clutch_points','clutch_attempts','clutch_fg_pct','clutch_hot_rate','clutch_avg_streak','clutch_max_streak','overall_hot_rate','overall_avg_streak','overall_max_streak']
    return {
        "total_qualifying_players":len(filtered_rankings),
        "min_attempts":min_attempts,
        "top_players":top[cols].to_dict(orient='records')
    }

@app.get("/players/{person_id}")
def get_player_data(person_id: int):
    """returns the player data for the given person_id"""
    if person_id not in rankings_full.index:
        raise HTTPException(status_code=404,detail=f"Player with personId {person_id} not found")
    row=rankings_full.loc[person_id]
    rank=(rankings_full.index== person_id).argmax()+1 

    return {
        'personId':person_id,
        'playerNameI':id_to_name.get(person_id,""),
        'rank':int(rank),
        'fire_score': round(float(row['fire_score']),3),
        'clutch_points':int(row['clutch_points']),
        'clutch_attempts':int(row['clutch_attempts']),
        'clutch_and_hot':int(row['clutch_and_hot']),
        'clutch_fg_pct':round(float(row['clutch_fg_pct']),3),
        'clutch_hot_rate':round(float(row['clutch_hot_rate']),3),
        'clutch_avg_streak':round(float(row['clutch_avg_streak']),3),
        'clutch_max_streak':int(row['clutch_max_streak']),
        'overall_hot_rate':round(float(row['overall_hot_rate']),3),
        'overall_avg_streak':round(float(row['overall_avg_streak']),3),
        'overall_max_streak':int(row['overall_max_streak']),
        'overall_fg_pct':round(float(row['overall_fg_pct']),3)
        
    }

@app.get("/shotchart/{person_id}")
def get_shot_chart(person_id: int):
    """returns the shot chart data for the given person_id"""
    shots=get_shot_chart_data(df,fg_df,person_id)
    if shots.empty:
        raise HTTPException(status_code=404,detail=f"No shot chart data found for {person_id}")

    shots['clutch']=shots['clutch'].astype(bool)
    shots['is_hot']=shots['is_hot'].astype(bool)
    shots['made']=shots['made'].astype(bool)
    shots['clutch_and_hot']=shots['clutch_and_hot'].astype(bool)
    return shots.to_dict(orient='records')

@app.get("/streaks/{person_id}")
def get_streaks(person_id:int):
    """returns the streaks data for the given person_id"""
    streak_data=streak_hot_rate_per_game(fg_df,df,person_id)
    if streak_data.empty:
        raise HTTPException(status_code=404,detail=f"No streak data found for {person_id}")
    return streak_data.to_dict(orient='records')

@app.get("/compare")
def compare_two_players(ids:str):
    """gets data for two players for comparison purposes, ids should be a comma separated string of two personIds"""
    try:
        person_ids=[int(x.strip()) for x in ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400,detail="Invalid. Person IDs should be integers and comma separated.")
    if len(person_ids)!=2:
        raise HTTPException(status_code=400,detail="Invalid. Provide exactly two person IDs for comparison.")

    result=[]
    for pid in person_ids:
        if pid not in rankings_full.index:
            raise HTTPException(status_code=404,detail=f"Player with person ID {pid} not found.")
        row=rankings_full.loc[pid]
        result.append({
            "personId":pid,
            "playerNameI": id_to_name.get(pid,""),
            "fire_score": round(float(row['fire_score']),3),
            'clutch_points':int(row['clutch_points']),
            'clutch_attempts':int(row['clutch_attempts']),
            'clutch_and_hot':int(row['clutch_and_hot']),
            'clutch_fg_pct':round(float(row['clutch_fg_pct']),3),
            'clutch_hot_rate':round(float(row['clutch_hot_rate']),3),
            'clutch_avg_streak':round(float(row['clutch_avg_streak']),3),
            'clutch_max_streak':int(row['clutch_max_streak']),
            'overall_hot_rate':round(float(row['overall_hot_rate']),3),
            'overall_avg_streak':round(float(row['overall_avg_streak']),3),
            'overall_max_streak':int(row['overall_max_streak']),
            'overall_fg_pct':round(float(row['overall_fg_pct']),3)})

    return result
