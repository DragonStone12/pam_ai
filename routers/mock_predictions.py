from models.mock_responses import mock_prediction
from routers.predictions import lifespan
from schemas.obesity_schema import ObesityPredictionInput, ObesityPredictionOutput

from fastapi import APIRouter, status, FastAPI

router = APIRouter(
    prefix="/predict",
    tags=["predict"]
)


@router.post("/mock/prediction", status_code=status.HTTP_201_CREATED, response_model=ObesityPredictionOutput)
def mock_predict_obesity(
    prediction_input: ObesityPredictionInput,
) -> ObesityPredictionOutput:
    return mock_prediction()


app = FastAPI(lifespan=lifespan)

app.include_router(router)
