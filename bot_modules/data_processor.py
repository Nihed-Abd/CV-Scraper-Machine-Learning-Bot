import os
import pandas as pd
import logging
from fpdf import FPDF
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(scraped_data):
    """
    Process the raw scraped data to extract relevant information
    
    Args:
        scraped_data: List of dictionaries containing CV data
        
    Returns:
        processed_data: Processed data ready for CSV and ML models
    """
    logger.info(f"Processing {len(scraped_data)} CV records")
    
    processed_data = []
    
    for cv in scraped_data:
        try:
            # Extract basic information
            record = {
                'name': cv.get('name', ''),
                'email': cv.get('email', ''),
                'country': cv.get('country', ''),
                'university': cv.get('education', {}).get('university', ''),
                'degree': cv.get('education', {}).get('degree', ''),
                'field': cv.get('education', {}).get('field', ''),
                'graduation_year': cv.get('education', {}).get('graduation_year', ''),
                'skills': ', '.join(cv.get('skills', [])),
                'years_of_experience': sum(int(exp.get('duration', '0').split()[0]) for exp in cv.get('experience', []))
            }
            
            # Extract experience information
            for i, exp in enumerate(cv.get('experience', [])):
                record[f'company_{i+1}'] = exp.get('company', '')
                record[f'position_{i+1}'] = exp.get('position', '')
                record[f'duration_{i+1}'] = exp.get('duration', '')
                record[f'description_{i+1}'] = exp.get('description', '')
            
            # Extract project information
            for i, proj in enumerate(cv.get('projects', [])):
                record[f'project_{i+1}_name'] = proj.get('name', '')
                record[f'project_{i+1}_description'] = proj.get('description', '')
                record[f'project_{i+1}_technologies'] = ', '.join(proj.get('technologies', []))
            
            processed_data.append(record)
            
        except Exception as e:
            logger.error(f"Error processing CV: {str(e)}")
            continue
    
    logger.info(f"Successfully processed {len(processed_data)} CV records")
    return processed_data

def generate_csv(processed_data):
    """
    Generate a CSV file from processed data
    
    Args:
        processed_data: List of dictionaries containing processed CV data
        
    Returns:
        csv_path: Path to the generated CSV file
    """
    if not processed_data:
        logger.error("No data available to generate CSV")
        return None
    
    logger.info("Generating CSV file")
    
    # Create directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame
    df = pd.DataFrame(processed_data)
    
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'cv_data_{timestamp}.csv')
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    
    logger.info(f"CSV file generated: {csv_path}")
    return csv_path

def generate_pdf_report(processed_data):
    """
    Generate a PDF report from processed data
    
    Args:
        processed_data: List of dictionaries containing processed CV data
        
    Returns:
        pdf_path: Path to the generated PDF file
    """
    if not processed_data:
        logger.error("No data available to generate PDF report")
        return None
    
    logger.info("Generating PDF report")
    
    # Create directory if it doesn't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(output_dir, f'cv_report_{timestamp}.pdf')
    
    # Create PDF instance
    pdf = FPDF()
    pdf.add_page()
    
    # Set up fonts
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "TuniHire ATS System - CV Data Report", 0, 1, "C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
    pdf.cell(0, 10, f"Total CVs processed: {len(processed_data)}", 0, 1, "C")
    pdf.ln(10)
    
    # Summary section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Data Summary", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    
    # Calculate education statistics
    education_fields = {}
    education_degrees = {}
    for cv in processed_data:
        field = cv.get('field', 'Unknown')
        degree = cv.get('degree', 'Unknown')
        
        education_fields[field] = education_fields.get(field, 0) + 1
        education_degrees[degree] = education_degrees.get(degree, 0) + 1
    
    # Calculate skills statistics
    all_skills = []
    for cv in processed_data:
        skills = cv.get('skills', '').split(', ')
        all_skills.extend([s for s in skills if s])
    
    skill_counts = {}
    for skill in all_skills:
        skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    # Sort skills by frequency
    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Add education statistics to PDF
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Education Fields Distribution:", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    
    for field, count in education_fields.items():
        percentage = (count / len(processed_data)) * 100
        pdf.cell(0, 8, f"{field}: {count} ({percentage:.1f}%)", 0, 1, "L")
    
    # Add degree statistics to PDF
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Degree Distribution:", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    
    for degree, count in education_degrees.items():
        percentage = (count / len(processed_data)) * 100
        pdf.cell(0, 8, f"{degree}: {count} ({percentage:.1f}%)", 0, 1, "L")
    
    # Add top skills to PDF
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Top 10 Skills:", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    
    for skill, count in top_skills:
        percentage = (count / len(processed_data)) * 100
        pdf.cell(0, 8, f"{skill}: {count} ({percentage:.1f}%)", 0, 1, "L")
    
    # Individual CV data
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Individual CV Data", 0, 1, "L")
    
    for i, cv in enumerate(processed_data):
        if i > 0:
            pdf.add_page()
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"CV #{i+1}", 0, 1, "L")
        
        pdf.set_font("Arial", "B", 10)
        pdf.cell(60, 8, "Name:", 0, 0, "L")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, cv.get('name', 'N/A'), 0, 1, "L")
        
        pdf.set_font("Arial", "B", 10)
        pdf.cell(60, 8, "Email:", 0, 0, "L")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, cv.get('email', 'N/A'), 0, 1, "L")
        
        pdf.set_font("Arial", "B", 10)
        pdf.cell(60, 8, "Country:", 0, 0, "L")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, cv.get('country', 'N/A'), 0, 1, "L")
        
        pdf.set_font("Arial", "B", 10)
        pdf.cell(60, 8, "University:", 0, 0, "L")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, cv.get('university', 'N/A'), 0, 1, "L")
        
        pdf.set_font("Arial", "B", 10)
        pdf.cell(60, 8, "Degree & Field:", 0, 0, "L")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, f"{cv.get('degree', 'N/A')} in {cv.get('field', 'N/A')}", 0, 1, "L")
        
        pdf.set_font("Arial", "B", 10)
        pdf.cell(60, 8, "Skills:", 0, 0, "L")
        pdf.set_font("Arial", "", 10)
        
        # Handle long skill lists by wrapping text
        skills_text = cv.get('skills', 'N/A')
        if len(skills_text) > 100:
            skills_parts = [skills_text[i:i+100] for i in range(0, len(skills_text), 100)]
            pdf.cell(0, 8, skills_parts[0], 0, 1, "L")
            for part in skills_parts[1:]:
                pdf.cell(60, 8, "", 0, 0, "L")  # Indent
                pdf.cell(0, 8, part, 0, 1, "L")
        else:
            pdf.cell(0, 8, skills_text, 0, 1, "L")
        
        # Experience section
        pdf.ln(5)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 8, "Experience:", 0, 1, "L")
        pdf.set_font("Arial", "", 10)
        
        for j in range(1, 4):  # Handle up to 3 experiences
            company_key = f'company_{j}'
            position_key = f'position_{j}'
            duration_key = f'duration_{j}'
            description_key = f'description_{j}'
            
            if company_key in cv and cv[company_key]:
                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 8, f"{cv.get(company_key, '')} - {cv.get(position_key, '')}", 0, 1, "L")
                pdf.set_font("Arial", "I", 10)
                pdf.cell(0, 8, f"Duration: {cv.get(duration_key, '')}", 0, 1, "L")
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 8, cv.get(description_key, ''))
                pdf.ln(2)
    
    # Save the PDF
    pdf.output(pdf_path)
    
    logger.info(f"PDF report generated: {pdf_path}")
    return pdf_path
