import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware



model = load_model('rice_model.keras')
class_names = ['bacterial leaf blight',
 'brown spot',
 'healthy',
 'leaf blast',
 'leaf scald',
 'narrow brown spot']

app = FastAPI()

origins = [
    '*'  
   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Prediction function
def predict1(image: Image.Image):
    try:
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        
        image = image.resize((256, 256))
        
        
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, 0)
        
        
        predictions = model.predict(img_array)
        result = class_names[np.argmax(predictions)]
        confidence = round(100 * (np.max(predictions)), 2)
        
        
        if confidence < 76:
            return {"disease": "can't say for sure", "confidence": f"{confidence}%"}
        else:
            return {"disease": result, "confidence": f"{confidence}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        
        image = Image.open(BytesIO(await file.read()))
        
        prediction = predict1(image)
        return JSONResponse(content=prediction)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
