import streamlit as st
import pandas as pd
from fpdf import FPDF
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from paddleocr import PaddleOCR
from PIL import Image
import pytesseract
import re
import io

# Load Summarization Model (Flan-T5)
summarization_model_name = "google/flan-t5-base"
summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # English OCR

# Streamlit Layout
st.title("AI Medical Record Summarization")
st.sidebar.header("Input Options")
st.sidebar.markdown("""
1. Upload an image or type text.
2. Click *Summarize* to get treatment and dietary recommendations.
""")

input_option = st.sidebar.radio("Choose input type:", ["Text", "Image"])

# Function to preprocess image
def preprocess_image(image):
    try:
        img = image.convert("RGB")  # Convert image to RGB for OCR processing
        return img
    except Exception as e:
        st.error(f"Error in preprocess_image: {e}")
        return image

# Function to Extract Text from Image using OCR (with Tesseract)
def extract_text_from_image(image):
    try:
        img = preprocess_image(image)
        ocr_text = pytesseract.image_to_string(img)
        return ocr_text
    except Exception as e:
        st.error(f"Error in extract_text_from_image: {e}")
        return ""

# Function to clean extracted text
def clean_text(text):
    """Clean the OCR-extracted text by removing unnecessary characters."""
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s\.,;:]', '', text)  # Remove unwanted characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra spaces
    return cleaned_text

# Function to Summarize Medical Record using Flan-T5 (Hugging Face Model)
def summarize_medical_record_with_flan_t5(text):
    try:
        inputs = summarization_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        summary_ids = summarization_model.generate(inputs['input_ids'], num_beams=4, min_length=50, max_length=300, early_stopping=True)
        summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Error in Flan-T5 summarization: {e}")
        return f"Error: {str(e)}"

# Function to Map extracted terms to ICD-10 codes
def map_to_icd_10_dynamically(text):
    icd_10_terms = {
        "hypertension": "I10",
        "myocardial infarction": "I21.9",
        "diabetes": "E11.9",
        "asthma": "J45.909",
        "pneumonia": "J18.9",
        "stroke": "I63.9",
        "infection": "A41.9",
        "cancer": "C80.9",
        "fatigue": "R53.83",
        "fever": "R50.9",
        "anemia": "D64.9",
        "arthritis": "M19.90",
        "chronic kidney disease": "N18.9",
        "depression": "F32.9",
        "migraine": "G43.909",
        "obesity": "E66.9",
        "hyperlipidemia": "E78.5",
        "hypothyroidism": "E03.9",
        "insomnia": "G47.00",
        "sepsis": "A41.9",
        "COPD": "J44.9",
        "allergy": "T78.4",
        "gout": "M10.9"
    }

    for term, code in icd_10_terms.items():
        text = text.replace(term, f"{term} (ICD-10: {code})")
    return text

# Function to Suggest Treatments and Medicines
def suggest_treatments_and_medicines(mapped_text):
    treatment_suggestions = {
        "hypertension": "ACE inhibitors (e.g., Lisinopril), Calcium channel blockers, Diuretics, Lifestyle changes (low-salt diet, exercise)",
        "myocardial infarction": "Aspirin, Statins (e.g., Atorvastatin), Beta-blockers (e.g., Metoprolol), Nitroglycerin, Coronary angioplasty",
        "diabetes": "Insulin therapy, Metformin, SGLT2 inhibitors, Dietary control, Continuous glucose monitoring",
        "asthma": "Inhaled corticosteroids (e.g., Fluticasone), Long-acting bronchodilators (e.g., Salmeterol), Leukotriene modifiers (e.g., Montelukast)",
        "pneumonia": "Antibiotics (e.g., Amoxicillin), Oxygen therapy, Hospitalization if severe, Fluids and rest",
        "stroke": "Thrombolytics (e.g., Alteplase), Antiplatelet therapy (e.g., Aspirin), Physical rehabilitation, Anticoagulants",
        "infection": "Antibiotics (e.g., Ceftriaxone, Ciprofloxacin), IV fluids, Symptomatic treatment, Rest",
        "cancer": "Chemotherapy, Radiation therapy, Immunotherapy, Surgery (depending on type), Palliative care for advanced stages",
        "fatigue": "Lifestyle changes, Addressing underlying causes (e.g., anemia, depression), Nutritional supplements (Iron, B12)",
        "fever": "Antipyretics (e.g., Paracetamol, Ibuprofen), Hydration, Rest, Treat underlying infection",
        "anemia": "Iron supplements (e.g., Ferrous sulfate), Vitamin B12 injections, Folate supplements, Address underlying causes",
        "arthritis": "NSAIDs (e.g., Ibuprofen, Naproxen), Disease-modifying antirheumatic drugs (DMARDs) for RA, Physical therapy, Steroid injections",
        "chronic kidney disease": "Blood pressure control, ACE inhibitors (e.g., Enalapril), Dialysis in advanced stages, Erythropoietin for anemia",
        "depression": "SSRIs (e.g., Sertraline, Fluoxetine), Cognitive Behavioral Therapy (CBT), Lifestyle changes (exercise, diet)",
        "migraine": "Triptans (e.g., Sumatriptan), Preventive therapy (e.g., Beta-blockers, Topiramate), Lifestyle modifications",
        "obesity": "Lifestyle changes (diet and exercise), Weight loss medications (e.g., Orlistat), Bariatric surgery in severe cases",
        "hyperlipidemia": "Statins (e.g., Atorvastatin), Fibrates (e.g., Gemfibrozil), Diet low in saturated fat, Exercise",
        "hypothyroidism": "Levothyroxine, Thyroid hormone replacement therapy, Regular monitoring of thyroid levels",
        "insomnia": "Cognitive Behavioral Therapy for Insomnia (CBT-I), Melatonin supplements, Sleep hygiene techniques",
        "sepsis": "Broad-spectrum antibiotics (e.g., Piperacillin-tazobactam), Fluid resuscitation, Vasopressors, ICU care if severe",
        "COPD": "Inhaled bronchodilators (e.g., Albuterol), Inhaled corticosteroids, Oxygen therapy, Pulmonary rehabilitation",
        "allergy": "Antihistamines (e.g., Cetirizine), Decongestants (e.g., Pseudoephedrine), Allergy shots (immunotherapy), Avoidance of allergens",
        "gout": "NSAIDs (e.g., Indomethacin), Colchicine, Allopurinol for long-term management, Lifestyle modifications (low-purine diet, reduced alcohol)"
    }

    suggestions = []
    for condition in treatment_suggestions.keys():
        if condition in mapped_text.lower():
            suggestions.append(f"**{condition.capitalize()}**: {treatment_suggestions[condition]}")
    return "\n\n".join(suggestions)

# Function to Suggest Dietary Recommendations
def suggest_dietary_recommendations(mapped_text):
    dietary_recommendations = {
        "hypertension": "Low-sodium DASH diet, rich in fruits, vegetables, and whole grains",
        "diabetes": "Low-glycemic index foods, high fiber, controlled carbohydrate intake, avoid sugary drinks",
        "asthma": "Anti-inflammatory foods like turmeric, ginger, omega-3 fatty acids, avoid sulfites and allergens",
        "pneumonia": "High-protein diet, plenty of fluids, vitamin-rich foods (A, C, and zinc)",
        "stroke": "Mediterranean diet, rich in omega-3 fatty acids, low in saturated fats and cholesterol",
        "infection": "Immune-boosting foods like garlic, citrus fruits, leafy greens, and probiotics",
        "cancer": "Nutrient-dense foods, high antioxidants, avoid processed and sugary foods",
        "fatigue": "Balanced diet with iron, B12, magnesium, and hydration",
        "anemia": "Iron-rich foods (spinach, lentils), vitamin C to enhance absorption, red meat (if non-vegetarian)",
        "arthritis": "Omega-3 fatty acids (fish, flaxseed), anti-inflammatory foods, limit sugar and processed foods",
        "chronic kidney disease": "Low-sodium, low-potassium diet, controlled protein intake, avoid processed foods",
        "depression": "Mediterranean diet, omega-3s, whole grains, avoid alcohol and processed foods",
        "migraine": "Magnesium-rich foods, hydration, avoid trigger foods (chocolate, caffeine, aged cheese)",
        "obesity": "Calorie-controlled diet, rich in vegetables, lean proteins, avoid sugary drinks and fast food",
        "hyperlipidemia": "Plant-based diet, fiber-rich foods (oats, beans), healthy fats (avocado, nuts)",
        "hypothyroidism": "Iodine-rich foods (seafood, iodized salt), selenium-rich foods (brazil nuts), avoid goitrogens",
        "insomnia": "Magnesium-rich foods (nuts, seeds), melatonin-rich foods (cherries), avoid caffeine in the evening",
        "sepsis": "Easily digestible foods, high in protein, avoid sugary and processed foods",
        "COPD": "High-protein, small frequent meals, avoid gas-producing foods and high-salt diet",
        "allergy": "Anti-inflammatory foods (ginger, turmeric), avoid allergenic foods",
        "gout": "Low-purine diet, cherries, avoid alcohol, sugary drinks, and organ meats"
    }

    recommendations = []
    for condition in dietary_recommendations.keys():
        if condition in mapped_text.lower():
            recommendations.append(f"**{condition.capitalize()}**: {dietary_recommendations[condition]}")
    return "\n\n".join(recommendations)

# Input Handling
if input_option == "Text":
    input_text = st.text_area("Enter medical record text here:")
elif input_option == "Image":
    uploaded_image = st.file_uploader("Upload an image of the medical record:")
    if uploaded_image:
        image = Image.open(io.BytesIO(uploaded_image.read()))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.info("Processing OCR...")
        input_text = extract_text_from_image(image)
    else:
        input_text = ""

# Generate Summary and Recommendations
if st.button("Summarize"):
    if input_text.strip():
        cleaned_text = clean_text(input_text)
        st.subheader("Cleaned Medical Record Text")
        st.write(cleaned_text)

        summary = summarize_medical_record_with_flan_t5(cleaned_text)
        st.subheader("Summarized Medical Record")
        st.write(summary)

        mapped_text = map_to_icd_10_dynamically(summary)
        st.subheader("Mapped ICD-10 Summary")
        st.write(mapped_text)

        treatments = suggest_treatments_and_medicines(mapped_text)
        if treatments:
            st.subheader("Treatment and Medicine Recommendations")
            st.write(treatments)

        diet = suggest_dietary_recommendations(mapped_text)
        if diet:
            st.subheader("Dietary Recommendations")
            st.write(diet)
    else:
        st.warning("Please enter text or upload an image to proceed.")

# Export PDF Button
if st.button("Export Summary as PDF"):
    if extracted_text:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="Medical Report Summary", ln=True, align="C")

        # Summary Text
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"Summary:\n{summary}")

        # Save PDF
        pdf_file = "/tmp/summary_report.pdf"
        pdf.output(pdf_file)

        # Provide download link
        st.download_button(
            label="Download PDF",
            data=open(pdf_file, "rb").read(),
            file_name="summary_report.pdf",
            mime="application/pdf"
        )