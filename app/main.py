from fastapi import FastAPI
from routers import predictions, users, mock_predictions

app = FastAPI()

app.include_router(mock_predictions.router)
app.include_router(predictions.router)
app.include_router(users.router)
