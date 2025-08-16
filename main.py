from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
from datetime import datetime  # Add this import


# flask app
app = Flask(__name__)

# load databasedataset===================================
sym_des = pd.read_csv("/Users/vedanshsharma/Downloads/Cap_project/dataset/symtoms_df.csv")
precautions = pd.read_csv("/Users/vedanshsharma/Downloads/Cap_project/dataset/precautions_df.csv")
workout = pd.read_csv("/Users/vedanshsharma/Downloads/Cap_project/dataset/workout_df.csv")
description = pd.read_csv("/Users/vedanshsharma/Downloads/Cap_project/dataset/description.csv")
medications = pd.read_csv('/Users/vedanshsharma/Downloads/Cap_project/dataset/Medicine.csv')
diets = pd.read_csv("/Users/vedanshsharma/Downloads/Cap_project/dataset/diets.csv")
alternatives = pd.read_csv("/Users/vedanshsharma/Downloads/Cap_project/dataset/alternatives.csv")

# load model===========================================
svc = pickle.load(open('/Users/vedanshsharma/Downloads/Cap_project/svc.pkl','rb'))



#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis][['Medication1', 'Dosage_Mild1','Dosage_Severe1', 'Medication2','Dosage_Mild2','Dosage_Severe2', 'Medication3','Dosage_Mild3','Dosage_Severe3', 'Medication4','Dosage_Mild4','Dosage_Severe4']]
    med = [med for med in med.values]
    
    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


# creating routes========================================


@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        # mysysms = request.form.get('mysysms')
        # print(mysysms)
        print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:

            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('index.html')

# about view function and path
@app.route('/about')
def about():
    return render_template("about.html")

# contact view function and path
@app.route('/contact')
def contact():
    return render_template("contact.html")

# blog view function and path
@app.route('/blog')
def blog():
    return render_template("blog.html")

# Alternative medicine recommendations
#

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').lower()
        
        # Initialize response
        response = None
        show_alternatives = 'alternative' in user_message
        
        # Check for disease mentions in the description dataset
        for disease in description['Disease'].values:
            if disease.lower() in user_message:
                alt_data = alternatives[alternatives['Disease'] == disease]
                
                if 'medicine' in user_message or 'medication' in user_message or 'treatment' in user_message:
                    if not alt_data.empty:
                        response = f"For {disease}, here are the treatment recommendations:\n\n"
                        response += f"Primary Medication:\n• {alt_data.iloc[0]['Primary_Medication']}\n"
                        if show_alternatives:
                            response += f"\nAlternative Medications:\n• {alt_data.iloc[0]['Alternative_Medication']}\n"
                
                elif 'diet' in user_message or 'food' in user_message or 'eat' in user_message:
                    if not alt_data.empty:
                        response = f"For {disease}, here are the dietary recommendations:\n\n"
                        response += f"Diet:\n• {alt_data.iloc[0]['Primary_Diet']}\n"
                        if show_alternatives:
                            response += f"\nFoods you can have :\n• {alt_data.iloc[0]['Alternative_Diet']}\n"
                
                elif 'exercise' in user_message or 'workout' in user_message or 'activity' in user_message:
                    if not alt_data.empty:
                        response = f"For {disease}, here are the recommended physical activities:\n\n"
                        response += f"Primary Exercise:\n• {alt_data.iloc[0]['Primary_Workout']}\n"
                        if show_alternatives:
                            response += f"\nAlternative Exercise:\n• {alt_data.iloc[0]['Alternative_Workout']}\n"
                
                elif show_alternatives:
                    if not alt_data.empty:
                        response = f"For {disease}, here is the complete treatment plan including alternatives:\n\n"
                        response += "=== Medications ===\n"
                        #response += f"Primary: {alt_data.iloc[0]['Primary_Medication']}\n"
                        response += f": {alt_data.iloc[0]['Alternative_Medication']}\n\n"
                        response += "=== Diet ===\n"
                        #response += f"Primary: {alt_data.iloc[0]['Primary_Diet']}\n"
                        response += f": {alt_data.iloc[0]['Alternative_Diet']}\n\n"
                        response += "=== Exercise ===\n"
                        #response += f"Primary: {alt_data.iloc[0]['Primary_Workout']}\n"
                        response += f": {alt_data.iloc[0]['Alternative_Workout']}\n"
                
                elif 'precaution' in user_message or 'prevent' in user_message:
                    prec = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
                    if len(prec) > 0:
                        response = f"For {disease}, here are all recommended precautions:\n\n"
                        for p in prec[0]:
                            if p and isinstance(p, str) and p.strip():
                                response += f"• {p}\n"

                # If no specific category was asked for, provide comprehensive information
                if not response:
                    if not alt_data.empty:
                        desc = description[description['Disease'] == disease]['Description'].values
                        response = f"=== {disease} ===\n\n"
                        if len(desc) > 0:
                            response += f"Description:\n{desc[0]}\n\n"
                        
                        response += "Primary Treatment Plan\n"
                        response += f"Medication:\n• {alt_data.iloc[0]['Alternative_Medication']}\n\n"
                        response += f"Diet:\n• {alt_data.iloc[0]['Alternative_Diet']}\n\n"
                        response += f"Exercise:\n• {alt_data.iloc[0]['Alternative_Workout']}\n\n"
                        
                        # Add precautions
                        prec = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
                        if len(prec) > 0:
                            response += "Precautions:\n"
                            for p in prec[0]:
                                if p and isinstance(p, str) and p.strip():
                                    response += f"• {p}\n"
                            response += "\n"
                        
                        response += "You can ask specifically about:\n"
                        response += "• Alternative medications\n"
                        response += "• Alternative diets\n"
                        response += "• Alternative exercises\n"
                        response += "• Or type 'alternatives' to see all options"
                break
        
        # Help message for general queries
        if not response and ('help' in user_message or 'guide' in user_message):
            response = """I can help you with comprehensive information about:
• Disease descriptions and symptoms
• Primary and alternative medications
• Primary and alternative diets
• Primary and alternative exercises
• Precautions and preventive measures

Try asking questions like:
• "Tell me about diabetes"
• "What are the alternatives for GERD?"
• "What medications are available for arthritis?"
• "What diet should I follow for hypertension?"
• "What exercises are recommended for asthma?"
• "What precautions should I take for bronchitis?"
"""
        
        # If still no response, provide the default message
        if not response:
            response = "Please ask about specific diseases and their treatments, diets, exercises, or precautions. You can also ask about alternatives or type 'help' for guidance."
        
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({
            'response': 'Sorry, there was an error processing your request. Please try again.'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
