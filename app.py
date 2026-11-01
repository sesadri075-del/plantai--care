import gradio as gr

# Disease prediction function
def predict_disease(image):

    if image is None:
        return "Please upload a plant leaf image."

    # AI model prediction
    results = model(
        image.convert("RGB"),
        top_k=3
    )

    # Best prediction
    best = results[0]

    disease = best["label"].replace(
        "___", " - "
    ).replace("_", " ")

    confidence = round(
        best["score"] * 100, 2
    )

    # Show top 3 predictions
    output = "🌿 PLANT DISEASE DETECTION RESULT\n\n"

    output += f"Predicted Disease: {disease}\n"

    output += f"Confidence: {confidence}%\n\n"

    output += "Other Predictions:\n"

    for result in results:
        name = result["label"].replace(
            "___", " - "
        ).replace("_", " ")

        score = round(
            result["score"] * 100, 2
        )

        output += f"{name}: {score}%\n"

    return output
# Website Design - Professional Version

custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    background: #f4f9f4;
}

#hero {
    background: linear-gradient(135deg, #14532d, #22c55e);
    color: white;
    padding: 35px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 20px;
}

#hero h1 {
    color: white;
    font-size: 36px;
}

#hero p {
    color: white;
    font-size: 17px;
}

.section-title {
    color: #166534;
    text-align: center;
}

#detect-btn {
    background: #15803d;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}

#result-box textarea {
    font-size: 16px !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="PlantCare AI | Plant Disease Detection"
) as demo:

    gr.Markdown("""
    <div id="hero">
        <h1>🌿 PlantCare AI</h1>
        <p>Plant Disease Detection Using Deep Learning</p>
        <p>Upload a leaf image and let Artificial Intelligence
        predict the possible plant disease.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📸 Upload Plant Image")

            image = gr.Image(
                type="pil",
                label="Choose a plant leaf image"
            )

            button = gr.Button(
                "🔍 Detect Plant Disease",
                variant="primary",
                elem_id="detect-btn"
            )

            gr.Markdown("""
            **For better results:**
            - Upload a clear leaf image.
            - Keep the leaf in good lighting.
            - Avoid blurry photographs.
            """)

        with gr.Column(scale=1):
            gr.Markdown("## 🔬 AI Detection Result")

            result = gr.Textbox(
                label="Prediction and Confidence",
                lines=10,
                interactive=False,
                elem_id="result-box"
            )

            gr.Markdown("""
            ### 🌱 About the Result

            The AI model predicts the most likely disease
            based on the uploaded image.

            Always verify the result before deciding on
            any plant treatment.
            """)

    button.click(
        fn=predict_disease,
        inputs=image,
        outputs=result
    )

    gr.Markdown("""
    ---

    ## 🌍 About This Project

    This project uses Deep Learning and an image
    classification model to identify possible plant diseases.

    **Technologies:** Python · Gradio · PyTorch · Hugging Face

    **Applications:** Smart Agriculture · Plant Health Monitoring

    ---

    <div style="text-align:center">
        <p>🌿 PlantCare AI | Academic Project</p>
        <p>AI predictions are estimates and may not always be accurate.</p>
    </div>
    """)

demo.launch(share=True, debug=True)

