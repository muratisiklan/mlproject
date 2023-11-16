from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get('/')
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predictdata")
def get_predict(request: Request):
    return templates.TemplateResponse('home.html', {"request": request})


@app.post("/predictdata")
async def predict_datapoint(request: Request):

    form_data = dict(await request.form())

    data = CustomData(
        gender=form_data["gender"],
        race_ethnicity=form_data["ethnicity"],
        parental_level_of_education=form_data["parental_level_of_education"],
        lunch=form_data["lunch"],
        test_preparation_course=form_data["test_preparation_course"],
        reading_score=float(form_data["reading_score"]),
        writing_score=float(form_data["writing_score"])
    )

    pred_df = data.get_data_as_data_frame()
    print(pred_df)

    predict_pipline = PredictPipeline()
    results = predict_pipline.predict(pred_df)

    return templates.TemplateResponse('home.html', {"request": request, "results": results[0]})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
