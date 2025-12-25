import tensorflow as tf

model = tf.keras.models.load_model("cardiac_risk_model.keras")
model.save("cardiac_risk_model")
