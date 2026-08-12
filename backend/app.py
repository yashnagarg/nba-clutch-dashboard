from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {"message": "NBA Clutch API is running"}