from tensorflow.keras.models import load_model

model = load_model('my_model_fixed.keras', compile=False)
model.summary()