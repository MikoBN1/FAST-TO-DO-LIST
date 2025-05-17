from fastapi import FastAPI
from controllers import obesity_prediction_controller
app = FastAPI()

app.include_router(obesity_prediction_controller.router)