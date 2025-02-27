from pydantic import BaseModel

class ModelPrediction(BaseModel):
    obesity_status: str
    probability: float


class ObesityPredictionOutput(BaseModel):
    logistic: ModelPrediction
    cart: ModelPrediction
    naive_bayes: ModelPrediction
    prediction: str
    probability: float


class ObesityPredictionInput(BaseModel):
    location: str
    marital_status: str
    age_group: str
    education: str
    sweet_drinks: str
    fatty_oily_foods: str
    grilled_foods: str
    preserved_foods: str
    seasoning_powders: str
    soft_carbonated_drinks: str
    alcoholic_drinks: str
    mental_emotional_disorders: str
    diagnosed_hypertension: str
    physical_activity: str
    smoking: str
    fruit_vegetables_consumption: str

class ObesityResults(BaseModel):
    obesity_status: str
    probability: float
