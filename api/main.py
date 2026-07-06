from fastapi import FastAPI

app = FastAPI(title="MLOps PGP API")

@app.get("/")
def home():
    return{"message":"Deployment API is running successfully"}

