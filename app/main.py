from fastapi import FastAPI
from routers import predictions, users

app = FastAPI()

app.include_router(predictions.router)
app.include_router(users.router)
