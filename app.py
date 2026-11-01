import gradio as gr
import os
from transformers import pipeline

# Load trained plant disease model
model = pipeline(
    "image-classification",
    model="LourivalGontijo/plantvillage-mobilenetv2",
    device=-1
)

# Disease prediction function
def predict_disease(image):
    if image is None:
        return "Please upload a plant leaf image."

    try:
        results = model(image.convert("RGB"), top_k=3)

        output = "🌿 PLANT DISEASE DETECTION RESULT\n\n"

        best = results[0]
        disease = best["label"].replace("___", " - ").replace("_", " ")
        confidence = round(best["score"] * 100, 2)

        output += f"Predicted Disease: {disease}\n"
        output += f"Confidence: {confidence}%\n\n"
        output += "Other Predictions:\n"

        for item in results:
            name = item["label"].replace("___", " - ").replace("_", " ")
            score = round(item["score"] * 100, 2)
            output += f"{name}: {score}%\n"

        return output

    except Exception as e:
        return f"Prediction Error: {str(e)}"


# Website design
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Plant AI-Care"
) as demo:

    gr.Markdown("""
    # 🌿 Plant AI-Care
    ### Detect plant diseases using Artificial Intelligence.
    Upload a plant leaf image to identify possible diseases.
    """)

    image = gr.Image(
        type="pil",
        label="Upload Plant Leaf Image"
    )

    button = gr.Button("🔍 Detect Disease", variant="primary")

    result = gr.Textbox(
        label="Detection Result",
        lines=10
    )

    button.click(
        fn=predict_disease,
        inputs=image,
        outputs=result
    )

    gr.Markdown("""
    ---
    **Technology:** Python, Gradio, PyTorch, Hugging Face

    **Note:** Predictions are educational estimates.
    Please verify results with an agricultural expert.
    """)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
     
