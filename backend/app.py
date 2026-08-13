from fastapi import FastAPI,HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from main import process_data, compute_fire_score, get_shot_chart_data, streak_hot_rate_per_game

app = FastAPI(title="NBA Clutch + Heat Check API", description="API for NBA Clutch and Heat Check data processing", version="1.0.0")
origins = ["https://localhost:5173"]

app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET"], allow_headers=["*"])

print("Loading the NBA 24-25 season data...")
df, fg_df=process_data()

rankings_full=compute_fire_score(df,fg_df,min_attempts=0)

id_to_name=df[['personId','playerNameI']].drop_duplicates().set_index('personId')['playerNameI'].to_dict()
print(f"{len(id_to_name)} players loaded.")
@app.get("/")
def home():
    return {"message": "NBA Clutch API is running"}