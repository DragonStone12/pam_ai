from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, status
import contextlib

from models.mock_responses import mock_predictions, mock_prediction
from models.obesity_model import obesity_model, memory, ObesityModel
from schemas.obesity_schema import ObesityPredictionOutput, CombinedPredictionOutput, ObesityPredictionInput

router = APIRouter(
    prefix="/predict",
    tags=["predict"]
)

@contextlib.asynccontextmanager
async def lifespan(api: FastAPI):
    obesity_model.load_model()
    yield

app = FastAPI(lifespan=lifespan)

@router.post("/prediction", status_code=status.HTTP_201_CREATED, response_model=ObesityPredictionOutput)
def predict_obesity(
    prediction_input: ObesityPredictionInput,
    model: ObesityModel = Depends(lambda: obesity_model)
) -> ObesityPredictionOutput:
    return model.predict_obesity(prediction_input)


@router.post("/predictions", response_model=CombinedPredictionOutput)
def predict_all_models(
    output: CombinedPredictionOutput = Depends(obesity_model.predict_all),
) -> CombinedPredictionOutput:
    return output


@router.delete("/cache", status_code=status.HTTP_204_NO_CONTENT)
def delete_cache():
    memory.clear()


@router.post("/mock/prediction", status_code=status.HTTP_201_CREATED, response_model=ObesityPredictionOutput)
def predict_obesity(
) -> ObesityPredictionOutput:
    return mock_prediction()


@router.post("/mock/predictions", status_code=status.HTTP_201_CREATED, response_model=CombinedPredictionOutput)
def predict_obesity(
) -> CombinedPredictionOutput:
    return mock_predictions()

app.include_router(router)

# {
#     "location": "Urban",
#     "marital_status": "Married",
#     "age_group": "35-39 years",
#     "education": "Finished Senior High School",
#     "sweet_drinks": "1 time per day",
#     "fatty_oily_foods": "1-2 times per week",
#     "grilled_foods": "<3 times per month",
#     "preserved_foods": "Never",
#     "seasoning_powders": ">1 time per day",
#     "soft_carbonated_drinks": "Never",
#     "alcoholic_drinks": "No",
#     "mental_emotional_disorders": "No",
#     "diagnosed_hypertension": "No",
#     "physical_activity": "Not adequate",
#     "smoking": "No",
#     "fruit_vegetables_consumption": "Not adequate"
# }

# {
#     "obesity_status": "non-obese" | "obese",
#     "probability": 0.XXXX  // A float between 0 and 1, rounded to 4 decimal places (e.g., 0.7325)
# }


# prediction = "Obese" if probability >= 0.5 else "Non-obese" uses a 0.5 threshold, which is standard for binary
#     classification but might need adjustment if the dataset is imbalanced (as noted in the paper, 21.77% obese vs. 78.23% non-obese). The paper’s Logistic
# Regression model had an AUC of 0.798, suggesting a potential need for a different threshold to optimize sensitivity/specificity.


# loaded_model: tuple[Pipeline, list[str]] = joblib.load(model_file) assumes obesity_model.joblib contains a
# tuple of (Pipeline, feature_order). If the saved model format differs (e.g., just a Pipeline), this will raise
# a TypeError. Verify the .joblib file structure matches this assumption.
# If the model doesn’t include preprocessing (e.g., OneHotEncoder), predict might fail on categorical data. Ensure
# the Pipeline includes appropriate transformers (e.g., ColumnTransformer with OneHotEncoder for categorical
# features, as described in the paper).
