import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import io
from PIL import Image
import numpy as np
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tensorflow.keras.models import load_model
from pydantic import BaseModel


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"


engine = create_engine(SQLALCHEMY_DATABASE_URL)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    username = Column(String, primary_key=True, index=True)
    password = Column(String)


Base.metadata.create_all(bind=engine)


class UserSignup(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str

class ForgotUser(BaseModel):
    username: str

# origins = ["http://lcp.ssa.onrender.com"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


model = load_model('./static/model2_fit.h5')


classes = ['Bengin cases', 'Malignant cases', 'Normal cases']


try:
    joblib_in = open("./static/model_fit.joblib", "rb")
    model_fit = joblib.load(joblib_in)
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model_fit = None


templates = Jinja2Templates(directory="templates")

@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": ""})


@app.get('/login.html')
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "results": ""})

@app.get('/signup.html')
async def login(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request, "results": ""})

@app.get('/home')
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "results": ""})

@app.post("/signup")
def signup(user: UserSignup):
    db = SessionLocal()
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="User already exists")
    db_user = User(username=user.username, password=user.password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    db.close()
    return {"message": "User signed up successfully"}


@app.post("/login")
def login(user: UserLogin):
    db = SessionLocal()
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or db_user.password != user.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    db.close()
    return {"message": "Login successful"}

from fastapi.security import OAuth2PasswordRequestForm

@app.post("/forgot_password")
def forgot_password(user: ForgotUser):
    print("insideeee forgot")
    db = SessionLocal()
    db_user = db.query(User).filter(User.username == user.username).first()
    
    if not db_user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")
    # Delete the user's data from the database
    db.delete(db_user)
    db.commit()
    db.close()
    # Redirect the user to the signup page
    return {"message": "User account deleted successfully"}


@app.post('/predict', response_class=HTMLResponse)
async def predict_lungcancer(request: Request, AGE: int = Form(...), SMOKING: int = Form(...), YELLOW_FINGERS: int = Form(...), ANXIETY: int = Form(...), 
                            PEER_PRESSURE: int = Form(...), CHRONIC_DISEASE: int = Form(...), WHEEZING: int = Form(...), ALCOHOL_CONSUMING: int = Form(...), 
                            COUGHING: int = Form(...), SHORTNESS_OF_BREATH: int = Form(...), SWALLOWING_DIFFICULTY: int = Form(...), CHEST_PAIN: int = Form(...), 
                            GENDER_NEW: int = Form(...)):

    data = {
        'AGE': AGE,
        'SMOKING': SMOKING,
        'YELLOW_FINGERS': YELLOW_FINGERS,
        'ANXIETY': ANXIETY,
        'PEER_PRESSURE': PEER_PRESSURE,
        'CHRONIC_DISEASE': CHRONIC_DISEASE,
        'WHEEZING': WHEEZING,
        'ALCOHOL_CONSUMING': ALCOHOL_CONSUMING,
        'COUGHING': COUGHING,
        'SHORTNESS_OF_BREATH': SHORTNESS_OF_BREATH,
        'SWALLOWING_DIFFICULTY': SWALLOWING_DIFFICULTY,
        'CHEST_PAIN': CHEST_PAIN,
        'GENDER_NEW': GENDER_NEW
    }

    prediction_value = model_fit.predict([list(data.values())])

    prediction = "You have LOW RISK of Lung Cancer" if prediction_value[0] < 0.5 else "You have HIGH RISK of Lung Cancer"

    return RedirectResponse(url=f"/result?prediction={prediction}")

@app.route("/result", methods=["GET", "POST"])
async def show_result(request: Request):
    if request.method == "GET":
        prediction = request.query_params.get("prediction", None)
        print("GET Prediction:", prediction)
        return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction})
    elif request.method == "POST":
        prediction = request.query_params.get("prediction", None)
        print("POST Prediction:", prediction)
        return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction})

@app.get("/mri", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("mri.html", {"request": request})

@app.post("/predict1")
async def predict(request: Request, file: UploadFile = File(None)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0  

        prediction = model.predict(np.expand_dims(img_array, axis=0))
        predicted_class_index = np.argmax(prediction)
        predicted_class = classes[predicted_class_index]
        print("Prediction:", prediction)
        print("Predicted class:", predicted_class)
        return RedirectResponse(url=f"/result1?prediction={predicted_class}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.route("/result1", methods=["GET", "POST"])
async def show_result(request: Request):
    if request.method == "GET":
        prediction = request.query_params.get("prediction", None)
        print("GET Prediction:", prediction)
        return templates.TemplateResponse("predImg.html", {"request": request, "prediction": prediction})
    elif request.method == "POST":
        prediction = request.query_params.get("prediction", None)
        print("POST Prediction:", prediction)
        return templates.TemplateResponse("predImg.html", {"request": request, "prediction": prediction})
    
@app.get("/geo", response_class=HTMLResponse)
async def read_geo(request: Request):
    return templates.TemplateResponse("geo.html", {"request": request})


# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)



# uvicorn app:app --reload
