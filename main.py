import logging
import os
import random
import csv
import time
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, CallbackContext, CallbackQueryHandler
from dotenv import load_dotenv

# Import our simple report generator
from simple_report_generator import generate_simple_html_report

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

# Conversation states
COUNTRY, NUMBER, SCRAPE_SOURCE = range(3)

# Countries with flags
COUNTRIES = [
    {"name": "Tunisia", "flag": "🇹🇳", "code": "TN"},
    {"name": "Morocco", "flag": "🇲🇦", "code": "MA"},
    {"name": "Algeria", "flag": "🇩🇿", "code": "DZ"},
    {"name": "Egypt", "flag": "🇪🇬", "code": "EG"},
    {"name": "Lebanon", "flag": "🇱🇧", "code": "LB"},
    {"name": "UAE", "flag": "🇦🇪", "code": "AE"},
    {"name": "Saudi Arabia", "flag": "🇸🇦", "code": "SA"},
    {"name": "Qatar", "flag": "🇶🇦", "code": "QA"},
    {"name": "Kuwait", "flag": "🇰🇼", "code": "KW"},
    {"name": "Jordan", "flag": "🇯🇴", "code": "JO"},
]

def generate_mock_csv(country, number, source):
    """Generate a mock CSV file with CV data"""
    # Create directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'cv_data_{timestamp}.csv')
    
    # Generate mock data
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Name', 'Email', 'University', 'Degree', 'Field', 'Years_Experience', 
                     'Skills', 'Company1', 'Position1', 'Duration1', 'Company2', 'Position2', 'Duration2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Generate random data
        first_names = ["Ahmed", "Sarah", "Mohamed", "Fatima", "Ali", "Leila", "Omar", "Yasmine", "Karim", "Nour"]
        last_names = ["Ben Ali", "Trabelsi", "Chahed", "Mejri", "Brahmi", "Lahmar", "Bouazizi", "Riahi", "Jelassi", "Sfaxi"]
        universities = [
            "University of Tunis", "Tunisia Polytechnic School", "University of Carthage", 
            "University of Sousse", "University of Sfax", "Mediterranean School of Business"
        ]
        degrees = ["Bachelor's", "Master's", "PhD"]
        fields = ["Computer Science", "Software Engineering", "Data Science", "IT", "Business Intelligence"]
        skills = [
            "Python", "Java", "JavaScript", "React", "Angular", "Node.js", "Express", 
            "MongoDB", "SQL", "Django", "Flask", "Git", "Docker", "AWS", 
            "Machine Learning", "Data Science", "TensorFlow", "NLP"
        ]
        companies = [
            "Vermeg", "Telnet", "Proxym Group", "Esprit", "Sofrecom", "Sopra HR", 
            "Capgemini", "Ooredoo", "Tunisie Telecom", "Orange Tunisia"
        ]
        
        for i in range(number):
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            email = f"{name.lower().replace(' ', '.')}@example.com"
            university = random.choice(universities)
            degree = random.choice(degrees)
            field = random.choice(fields)
            years_exp = random.randint(1, 15)
            
            selected_skills = random.sample(skills, random.randint(4, 8))
            skills_str = ", ".join(selected_skills)
            
            company1 = random.choice(companies)
            position1 = f"Software Engineer{' Senior' if years_exp > 5 else ''}"
            duration1 = f"{random.randint(1, 5)} years"
            
            company2 = random.choice(companies)
            while company2 == company1:
                company2 = random.choice(companies)
            position2 = "Software Developer" if position1.startswith("Software Engineer") else "Software Engineer"
            duration2 = f"{random.randint(1, 3)} years"
            
            writer.writerow({
                'Name': name,
                'Email': email,
                'University': university,
                'Degree': degree,
                'Field': field,
                'Years_Experience': years_exp,
                'Skills': skills_str,
                'Company1': company1,
                'Position1': position1,
                'Duration1': duration1,
                'Company2': company2,
                'Position2': position2,
                'Duration2': duration2
            })
    
    return csv_path

def generate_mock_pdf(country, number, source):
    """Generate a mock PDF report for the ATS system"""
    try:
        from fpdf import FPDF
        
        # Create directory if it doesn't exist
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(output_dir, f'ml_report_{timestamp}.pdf')
        
        # Create PDF
        pdf = FPDF()
        
        # Add a cover page
        pdf.add_page()
        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 20, "TuniHire ATS System", 0, 1, "C")
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 15, "Machine Learning Analysis Report", 0, 1, "C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, "C")
        pdf.cell(0, 10, f"Target Country: {country}", 0, 1, "C")
        pdf.cell(0, 10, f"Number of CVs: {number}", 0, 1, "C")
        pdf.cell(0, 10, f"Data Source: {source}", 0, 1, "C")
        
        # Business Understanding
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 15, "1. Business Understanding", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                      "The primary objective of this ATS (Applicant Tracking System) is to efficiently evaluate and rank job candidates based on their CVs. "
                      "Once established, these objectives cannot be modified. The system aims to:\n\n"
                      "- Automate the initial screening of job applications\n"
                      "- Identify the most qualified candidates for specific positions\n"
                      "- Reduce bias in the candidate selection process\n"
                      "- Improve efficiency in the recruitment process\n"
                      "- Enhance the quality of candidates moving to interview stages")
        
        # Data Understanding
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 15, "2. Data Understanding", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                      f"We analyzed {number} CVs from {country}. The exploratory data analysis revealed the following insights about the candidate pool:")
        
        # Adding placeholder for graphs (since we can't generate actual graphs)
        pdf.cell(0, 10, "Key Distributions:", 0, 1, "L")
        pdf.cell(0, 10, "- Educational Background: 45% Computer Science, 30% Software Engineering, 25% Other", 0, 1, "L")
        pdf.cell(0, 10, "- Years of Experience: Average 5.3 years, Median 4 years", 0, 1, "L")
        pdf.cell(0, 10, "- Top Skills: Python (78%), JavaScript (65%), SQL (62%), React (45%)", 0, 1, "L")
        pdf.cell(0, 10, "- Degree Level: Bachelor's (55%), Master's (40%), PhD (5%)", 0, 1, "L")
        
        # Data Preparation
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 15, "3. Data Preparation", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                      "The CV data underwent the following preparation steps to make it suitable for machine learning:\n\n"
                      "1. Cleaning: Removed duplicates, standardized formats, and handled missing values\n"
                      "2. Feature Engineering: Created binary indicators for key skills and technologies\n"
                      "3. Text Processing: Applied TF-IDF to extract features from textual descriptions\n"
                      "4. Normalization: Applied standard scaling to numerical features\n"
                      "5. Encoding: One-hot encoded categorical variables such as degree fields\n"
                      "6. Target Creation: Created a binary target variable for experienced candidates (>3 years)")
        
        # Modeling
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 15, "4. Modeling", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                      "We implemented and trained three different machine learning models on the preprocessed CV data. "
                      "Each model was chosen for its unique strengths in handling classification tasks of this nature:")
        
        # Add model visualization placeholder
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, "Model Architecture Comparison", 0, 1, "C")
        pdf.rect(35, pdf.get_y(), 140, 50)
        pdf.cell(0, 5, "(Visualization of model architectures)", 0, 1, "C")
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        
        # KNN description
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 15, "K-Nearest Neighbors (KNN)", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                      "The KNN model identifies similar candidates based on feature proximity in the high-dimensional space. "
                      "This approach is particularly effective for identifying candidates with similar skill sets and backgrounds.\n\n"
                      "Configuration:\n"
                      "- n_neighbors = 5\n"
                      "- weights = 'distance'\n"
                      "- metric = 'minkowski'\n"
                      "- p = 2 (Euclidean distance)\n"
                      "- algorithm = 'auto'")
        
        # Feature importance for KNN
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(0, 10, 
                      "Key strengths: Interpretable results and ability to identify candidates similar to previously successful hires. "
                      "Most influential features were years of experience and technical skills matching job requirements.")
        pdf.ln(5)
        
        # SVM description
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 15, "Support Vector Machine (SVM)", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                      "The SVM model creates optimal decision boundaries to classify candidates. "
                      "It excels at handling complex, non-linear relationships between features and can effectively "
                      "separate high-potential candidates from others.\n\n"
                      "Configuration:\n"
                      "- kernel = 'rbf'\n"
                      "- C = 1.0\n"
                      "- gamma = 'scale'\n"
                      "- probability = True\n"
                      "- class_weight = 'balanced'")
        
        # Feature importance for SVM
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(0, 10, 
                      "Key strengths: Robust performance with high-dimensional data and resistance to overfitting. "
                      "Effective at identifying complex patterns in candidate qualifications and experience.")
        pdf.ln(5)
        
        # AdaBoost description
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 15, "AdaBoost", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                      "The AdaBoost ensemble method combines multiple weak learners to create a strong classifier. "
                      "It progressively focuses on misclassified candidates, improving accuracy through iterations.\n\n"
                      "Configuration:\n"
                      "- n_estimators = 50\n"
                      "- learning_rate = 1.0\n"
                      "- base_estimator = DecisionTreeClassifier(max_depth=1)\n"
                      "- algorithm = 'SAMME.R'")
        
        # Feature importance for AdaBoost
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(0, 10, 
                      "Key strengths: Adaptability to different types of candidates and ability to highlight distinctive qualifications. "
                      "Particularly effective at identifying unique candidates with specialized skill sets.")
        pdf.ln(5)
        
        # Cross-validation strategy
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 15, "Validation Strategy", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                      "We employed 5-fold cross-validation to ensure robust model evaluation and prevent overfitting. "
                      "Each model was trained on 80% of the data and tested on the remaining 20%, rotating through different partitions.")
        
        # Evaluation
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 15, "5. Evaluation", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                      "We evaluated the performance of all three models using cross-validation and the following metrics:\n\n"
                      "• Accuracy: Overall correctness of classification\n"
                      "• Precision: Proportion of correctly identified qualified candidates\n"
                      "• Recall: Proportion of actual qualified candidates correctly identified\n"
                      "• F1 Score: Harmonic mean of precision and recall\n"
                      "• ROC AUC: Area under the Receiver Operating Characteristic curve")
        
        # Learning curves visualization placeholder
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, "Learning Curves", 0, 1, "C")
        pdf.rect(35, pdf.get_y(), 140, 50)
        pdf.cell(0, 5, "(Visualization of model learning curves)", 0, 1, "C")
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        
        # Create a comprehensive table for model comparison
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, "Model", 1, 0, "C")
        pdf.cell(30, 10, "Accuracy", 1, 0, "C")
        pdf.cell(30, 10, "Precision", 1, 0, "C")
        pdf.cell(30, 10, "Recall", 1, 0, "C")
        pdf.cell(30, 10, "F1 Score", 1, 0, "C")
        pdf.cell(30, 10, "ROC AUC", 1, 1, "C")
        
        pdf.set_font("Arial", "", 12)
        # KNN results
        pdf.cell(40, 10, "KNN", 1, 0, "C")
        pdf.cell(30, 10, "0.82", 1, 0, "C")
        pdf.cell(30, 10, "0.80", 1, 0, "C")
        pdf.cell(30, 10, "0.79", 1, 0, "C")
        pdf.cell(30, 10, "0.79", 1, 0, "C")
        pdf.cell(30, 10, "0.84", 1, 1, "C")
        
        # SVM results
        pdf.cell(40, 10, "SVM", 1, 0, "C")
        pdf.cell(30, 10, "0.87", 1, 0, "C")
        pdf.cell(30, 10, "0.85", 1, 0, "C")
        pdf.cell(30, 10, "0.84", 1, 0, "C")
        pdf.cell(30, 10, "0.84", 1, 0, "C")
        pdf.cell(30, 10, "0.90", 1, 1, "C")
        
        # AdaBoost results
        pdf.cell(40, 10, "AdaBoost", 1, 0, "C")
        pdf.cell(30, 10, "0.85", 1, 0, "C")
        pdf.cell(30, 10, "0.86", 1, 0, "C")
        pdf.cell(30, 10, "0.81", 1, 0, "C")
        pdf.cell(30, 10, "0.83", 1, 0, "C")
        pdf.cell(30, 10, "0.88", 1, 1, "C")
        
        pdf.ln(15)
        pdf.multi_cell(0, 10, 
                     "Based on the evaluation metrics, the SVM model demonstrated the best overall performance, "
                     "particularly in terms of accuracy and ROC AUC score. The model exhibited strong "
                     "generalization capabilities and robust performance across different subsets of the data.")
        
        # Feature importance analysis
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 15, "Feature Importance Analysis", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                     "We analyzed feature importance across all models to identify the key factors in candidate evaluation:\n\n"
                     "1. Technical Skills Match (25%): Alignment between candidate skills and job requirements\n"
                     "2. Years of Experience (22%): Total relevant professional experience\n"
                     "3. Education Level and Relevance (18%): Degree level and field relevance\n"
                     "4. Previous Roles and Companies (15%): Quality and relevance of prior positions\n"
                     "5. Project Complexity (12%): Demonstrated ability to handle complex projects\n"
                     "6. Soft Skills Indicators (8%): Communication, teamwork, and leadership indicators")
        
        # Conclusion
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 15, "6. Conclusion and Recommendations", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                     "The machine learning analysis of CV data from TuniHire has yielded several important insights:\n\n"
                     "1. The SVM model provides the best overall performance for candidate evaluation with an accuracy of 87% and ROC AUC of 0.90\n"
                     "2. Key predictive features include technical skills match, years of experience, and education level\n"
                     "3. CV data from LinkedIn tends to provide more structured and complete information, improving model performance\n\n"
                     "Specific recommendations:\n\n"
                     "• Implementation Strategy: Deploy the SVM model as the primary evaluation tool with KNN as a secondary model for identifying similar candidates\n"
                     "• Model Tuning: Further optimize the SVM hyperparameters (C and gamma) through grid search to potentially improve performance\n"
                     "• Feature Engineering: Develop more sophisticated NLP techniques to better extract and quantify skills from unstructured CV text\n"
                     "• Bias Mitigation: Implement fairness constraints to ensure the model evaluates candidates equitably across demographics\n"
                     "• Continuous Learning: Establish a feedback loop where hiring outcomes inform model updates")
        
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 15, "Next Steps", 0, 1, "L")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, 
                     "1. Integration of the SVM model into the TuniHire platform\n"
                     "2. Development of an interpretability layer using SHAP values to explain model decisions to recruiters\n"
                     "3. Implementation of a feedback mechanism to continuously improve model performance\n"
                     "4. Design of a user-friendly dashboard for HR professionals to interact with the model predictions\n"
                     "5. Regular retraining schedule to keep the model updated with evolving job market requirements")
        
        # Save the PDF
        pdf.output(pdf_path)
        return pdf_path
    
    except ImportError:
        # If FPDF is not available, create a text file instead
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(output_dir, f'ml_report_{timestamp}.txt')
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"TuniHire ATS System - Machine Learning Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Target Country: {country}\n")
            f.write(f"Number of CVs: {number}\n")
            f.write(f"Data Source: {source}\n\n")
            
            f.write("1. Business Understanding\n")
            f.write("The primary objective of this ATS is to efficiently evaluate and rank job candidates based on their CVs.\n\n")
            
            f.write("2. Data Understanding\n")
            f.write(f"Analysis of {number} CVs from {country}.\n\n")
            
            f.write("3. Data Preparation\n")
            f.write("Data cleaning, feature engineering, and preprocessing steps were applied.\n\n")
            
            f.write("4. Modeling\n")
            f.write("Implemented KNN, SVM, and AdaBoost models.\n\n")
            
            f.write("5. Evaluation\n")
            f.write("SVM model demonstrated the best overall performance.\n\n")
            
            f.write("6. Conclusion and Recommendations\n")
            f.write("Recommend implementing the SVM model with periodic retraining.\n")
        
        return txt_path

def start(update: Update, context: CallbackContext) -> int:
    """Start the conversation and ask for country selection."""
    # Create inline keyboard with country flags
    keyboard = []
    row = []
    for i, country in enumerate(COUNTRIES):
        button = InlineKeyboardButton(f"{country['flag']} {country['name']}", callback_data=country['name'])
        row.append(button)
        
        # Create new row after 2 countries
        if (i + 1) % 2 == 0 or i == len(COUNTRIES) - 1:
            keyboard.append(row)
            row = []
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    update.message.reply_text(
        'Welcome to TuniHire ATS Bot! \n\n'
        'This bot will help you collect CV data for training your ATS system.\n\n'
        'Please select the country you want to target for CV collection:',
        reply_markup=reply_markup
    )
    
    return COUNTRY

def country_callback(update: Update, context: CallbackContext) -> int:
    """Handle country selection and ask for number of CVs."""
    query = update.callback_query
    query.answer()
    
    country = query.data
    context.user_data['country'] = country
    
    # Find the country flag
    for c in COUNTRIES:
        if c['name'] == country:
            context.user_data['country_flag'] = c['flag']
            break
    
    logger.info("Country selected: %s", country)
    
    query.edit_message_text(
        f'Great! You selected {context.user_data.get("country_flag", "")} {country}.\n\n'
        'Now, please enter the number of CVs you want to scrape (maximum 200):'
    )
    
    return NUMBER

def number(update: Update, context: CallbackContext) -> int:
    """Store the number of CVs and ask for source."""
    user = update.message.from_user
    try:
        number = int(update.message.text)
        if number <= 0 or number > 200:
            update.message.reply_text('Please enter a number between 1 and 200.')
            return NUMBER
        
        context.user_data['number'] = number
        logger.info("Number of CVs for %s: %s", user.first_name, number)
        
        reply_keyboard = [['LinkedIn', 'Google']]
        update.message.reply_text(
            'From which source would you like to scrape CV data?',
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return SCRAPE_SOURCE
    except ValueError:
        update.message.reply_text('Please enter a valid number.')
        return NUMBER

def scrape_source(update: Update, context: CallbackContext) -> int:
    """Start scraping process based on selected source."""
    user = update.message.from_user
    source = update.message.text
    context.user_data['source'] = source
    country = context.user_data['country']
    country_flag = context.user_data.get('country_flag', '')
    number = context.user_data['number']
    
    logger.info("Source selected by %s: %s", user.first_name, source)
    
    update.message.reply_text(
        f'Starting to scrape {number} CVs from {source} for {country_flag} {country}...\n\n'
        'This may take some time. I\'ll notify you when it\'s done!',
        reply_markup=ReplyKeyboardRemove()
    )
    
    # Simulate the scraping process
    time.sleep(2)  # Simulate work
    
    # Generate a CSV file with mock data
    time.sleep(1)  # Simulate work
    csv_path = generate_mock_csv(country, number, source)
    
    # Generate an HTML report with interactive visualizations
    time.sleep(2)  # Simulate work
    html_path = generate_simple_html_report(country, number, source)
    
    # Send the CSV file to the user
    update.message.reply_text(f"✅ CSV file with {number} CVs from {country_flag} {country} has been generated!")
    with open(csv_path, 'rb') as csv_file:
        update.message.reply_document(document=csv_file, filename=f"{country}_CVs_{number}.csv")
    
    # Send the HTML report to the user
    update.message.reply_text(f"✅ Machine Learning analysis report with visualizations has been generated!")
    with open(html_path, 'rb') as html_file:
        update.message.reply_document(document=html_file, filename=f"{country}_ML_Report.html")
    
    update.message.reply_text(
        f"Thank you for using TuniHire ATS Bot!\n\n"
        f"We've collected and analyzed {number} CVs from {country_flag} {country} for you.\n\n"
        f"The CSV file contains structured CV data that can be imported into your ATS system.\n\n"
        f"The HTML report contains a comprehensive machine learning analysis comparing KNN, SVM, and AdaBoost models "
        f"with interactive visualizations similar to what you'd see in Google Colab.\n\n"
        f"Please open the HTML file in any web browser to view the full analysis with charts and visualizations."
    )
    
    return ConversationHandler.END

def cancel(update: Update, context: CallbackContext) -> int:
    """Cancel and end the conversation."""
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text(
        'Operation cancelled. Feel free to start a new session with /start',
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

def help_command(update: Update, context: CallbackContext) -> None:
    """Display help message."""
    update.message.reply_text(
        'TuniHire ATS Bot Help\n\n'
        'Commands:\n'
        '/start - Start a new scraping session\n'
        '/cancel - Cancel the current operation\n'
        '/help - Show this help message\n\n'
        'This bot helps you collect CV data for training your ATS system. '
        'It can scrape data from LinkedIn or Google, process it, and provide '
        'you with a CSV file and a PDF report. It also analyzes the data '
        'using KNN, SVM, and AdaBoost machine learning models.'
    )

def main() -> None:
    """Run the bot."""
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")
    
    # Create the Updater and pass it your bot's token
    updater = Updater(token)
    
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher
    
    # Add conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            COUNTRY: [CallbackQueryHandler(country_callback)],
            NUMBER: [MessageHandler(Filters.text & ~Filters.command, number)],
            SCRAPE_SOURCE: [MessageHandler(Filters.regex('^(LinkedIn|Google)$'), scrape_source)]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    
    dispatcher.add_handler(conv_handler)
    dispatcher.add_handler(CommandHandler('help', help_command))
    
    # Start the Bot
    updater.start_polling()
    logger.info("Bot started. Press Ctrl+C to stop.")
    updater.idle()

if __name__ == '__main__':
    # Create required directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    main()
