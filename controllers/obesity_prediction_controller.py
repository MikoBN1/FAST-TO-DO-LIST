from fastapi import APIRouter, HTTPException
from services.obesity_prediction_service import get_knn_prediction
router = APIRouter(prefix="/prediction")

@router.get("/obesity")
def make_prediction():
    return get_knn_prediction()
