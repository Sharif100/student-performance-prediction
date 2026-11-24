
import os
import joblib
import pandas as pd
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "Final_Random_Forest_Model.pkl"))
data = joblib.load(os.path.join(BASE_DIR, "step2_preprocessing.pkl"))
X = data["X"]
feature_columns = joblib.load(os.path.join(BASE_DIR, "Feature_Columns.pkl"))

groups = {
    "Student Information": [
        "school", "sex", "age", "address", "famsize", "Pstatus"
    ],
    "Family and Education": [
        "Medu", "Fedu", "Mjob", "Fjob", "guardian",
        "schoolsup", "famsup", "paid", "nursery", "higher"
    ],
    "Study and Lifestyle": [
        "reason", "traveltime", "studytime", "failures",
        "activities", "internet", "romantic"
    ],
    "Social and Health Information": [
        "famrel", "freetime", "goout", "Dalc",
        "Walc", "health", "absences"
    ]
}

def predict_student(*values):
    try:
        cleaned = []
        for col, value in zip(X.columns, values):
            if pd.api.types.is_numeric_dtype(X[col]):
                cleaned.append(int(value))
            else:
                cleaned.append(value)

        student = pd.DataFrame([cleaned], columns=X.columns)
        combined = pd.concat([X, student], ignore_index=True)
        encoded = pd.get_dummies(combined, drop_first=True)
        encoded = encoded.iloc[[-1]].reindex(
            columns=feature_columns, fill_value=0
        )

        prediction = model.predict(encoded)[0]
        probabilities = model.predict_proba(encoded)[0]
        confidence = max(probabilities) * 100

        colors = {
            "Poor": "#dc3545",
            "Average": "#e09b18",
            "Good": "#198754"
        }
        color = colors.get(str(prediction), "#333333")

        return f"""
        <div style="text-align:center;padding:25px;
        border-radius:14px;border:2px solid {color};">
            <h2 style="color:{color};">Prediction Result</h2>
            <h1 style="color:{color};font-size:36px;">{prediction}</h1>
            <p style="font-size:18px;">
            Model Confidence: <b>{confidence:.2f}%</b></p>
            <p style="color:gray;font-size:13px;">
            Confidence is predicted probability, not overall model accuracy.
            </p>
        </div>
        """
    except Exception as e:
        return f"<p style='color:red;'>Prediction error: {str(e)}</p>"

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Student Performance Prediction"
) as app:

    gr.Markdown("""
    # 🎓 Student Performance Prediction
    ### Machine Learning Based Academic Performance Analysis

    Enter the student's information below to predict
    the expected academic performance.
    """)

    components = {}

    for group_name, columns in groups.items():
        with gr.Accordion(group_name, open=True):
            with gr.Row():
                for col in columns:
                    with gr.Column(min_width=180):
                        if X[col].dtype == "object":
                            choices = sorted(
                                X[col].dropna().unique().tolist()
                            )
                            component = gr.Dropdown(
                                choices=choices,
                                value=X[col].mode()[0],
                                label=col
                            )
                        else:
                            component = gr.Slider(
                                minimum=int(X[col].min()),
                                maximum=int(X[col].max()),
                                value=int(X[col].median()),
                                step=1,
                                label=col
                            )
                        components[col] = component

    ordered_inputs = [components[col] for col in X.columns]

    with gr.Row():
        predict_button = gr.Button(
            "🔍 Predict Student Performance",
            variant="primary"
        )
        clear_button = gr.ClearButton(
            components=ordered_inputs,
            value="↻ Reset Form"
        )

    gr.Markdown("## Prediction Output")
    result = gr.HTML(
        "<p style='text-align:center;'>Prediction will appear here.</p>"
    )

    predict_button.click(
        fn=predict_student,
        inputs=ordered_inputs,
        outputs=result
    )

    gr.Markdown("""
    ---
    **Research Project:** Student Performance Prediction Using Machine Learning

    **Model:** Random Forest Classifier

    **Research Dataset:** UCI Student Performance Dataset

    *This application is a research prototype. Predictions are
    estimates and should not be treated as definitive academic decisions.*
    """)


    gr.HTML("""
    <div style="
        text-align: center;
        padding: 25px 10px;
        margin-top: 25px;
        border-top: 1px solid #888;
        line-height: 2;
    ">
        <div style="font-size: 17px; font-weight: bold;">
            Developed by Md Sharif Hossain
        </div>
        <div>
            B.Sc (
            <a href="https://pust.ac.bd" target="_blank">
                pust.ac.bd
            </a>)
        </div>
        <div>
            M.Sc (
            <a href="https://uttara.ac.bd" target="_blank">
                uttara.ac.bd
            </a>)
        </div>
        <div>Assistant Teacher, CPSCM</div>
    </divdivdiv>
    """)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
