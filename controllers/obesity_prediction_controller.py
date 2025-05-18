from fastapi import APIRouter, HTTPException, Response

from schemas.dataset_schema import ObesityInput
from schemas.user_input_schema import UserInput
from services.obesity_prediction_service import get_knn_prediction, get_knn_error_plot, preprocess_user_input, predict_obesity_class, add_new_data_and_retrain, plot_confusion_matrix
router = APIRouter(prefix="/prediction", tags=["prediction"])

@router.get("/obesity")
def make_prediction():
    return get_knn_prediction()

@router.get("/confusion/matrix")
def make_prediction():
    return plot_confusion_matrix()

@router.get("/error-plot")
def get_error_plot():
    image_bytes = get_knn_error_plot()
    return Response(content=image_bytes, media_type="image/png")

@router.post("/obesity/user")
def predict(user_data: UserInput):
    user_df = preprocess_user_input(user_data)
    pred_class = predict_obesity_class(user_df)
    user_dict = user_df.to_dict(orient="records")
    return {"pred_class": pred_class}

@router.post("/data")
def add_data(data: ObesityInput):
    add_new_data_and_retrain(data)
    return {"status": "success"}
