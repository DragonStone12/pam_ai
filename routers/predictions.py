from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, status
import contextlib

from models.mock_responses import mock_prediction
from models.obesity_model import obesity_model, memory, ObesityModel
from schemas.obesity_schema import ObesityPredictionInput, ObesityPredictionOutput

router = APIRouter(
    prefix="/predict",
    tags=["predict"]
)

@contextlib.asynccontextmanager
async def lifespan(api: FastAPI):
    obesity_model.load_model()
    yield



@router.post("/prediction", status_code=status.HTTP_201_CREATED, response_model=ObesityPredictionOutput)
def predict_obesity(
    prediction_input: ObesityPredictionInput,
    model: ObesityModel = Depends(lambda: obesity_model)
) -> ObesityPredictionOutput:
    return model.predict_obesity(prediction_input)


@router.delete("/cache", status_code=status.HTTP_204_NO_CONTENT)
def delete_cache():
    memory.clear()


@router.post("/mock/prediction", status_code=status.HTTP_201_CREATED, response_model=ObesityPredictionOutput)
def mock_predict_obesity(
    prediction_input: ObesityPredictionInput,
) -> ObesityPredictionOutput:
    return mock_prediction()


app = FastAPI(lifespan=lifespan)

app.include_router(router)
