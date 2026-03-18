import gradio as gr
import requests
import json

def predict_car_make(colour, odometer, doors, price):
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "Colour": colour,
        "Odometer": int(odometer),
        "Doors": int(doors),
        "Price": int(price)
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            return f"🚗 Predicted Manufacturer: {result['predicted_make']}"
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❗ Connection Error: Ensure car_app.py is running on port 8000."

# Define the Gradio interface
interface = gr.Interface(
    fn=predict_car_make,
    inputs=[
        gr.Dropdown(["White", "Blue", "Red", "Black", "Green"], label="Colour"),
        gr.Number(label="Odometer (KM)", value=50000),
        gr.Slider(minimum=2, maximum=5, step=1, label="Doors", value=4),
        gr.Number(label="Price ($)", value=15000)
    ],
    outputs=gr.Textbox(label="Result"),
    title="Car Manufacturer Predictor",
    description="Interface for the Car FastAPI model.",
    theme="soft"
)

if __name__ == "__main__":
    interface.launch(server_port=7860)
