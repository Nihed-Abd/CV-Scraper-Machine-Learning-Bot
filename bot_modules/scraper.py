import os
import time
import random
import requests
import logging
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseScraper:
    def __init__(self, country, number_of_cvs):
        self.country = country
        self.number_of_cvs = min(number_of_cvs, 200)  # Enforce the maximum limit
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def setup_driver(self):
        """Set up and return a headless Chrome webdriver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    
    def parse_cv_data(self, html_content):
        """Parse HTML to extract CV data (to be implemented by child classes)"""
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def scrape(self):
        """Perform scraping operation (to be implemented by child classes)"""
        raise NotImplementedError("This method should be implemented by subclasses")


class LinkedInScraper(BaseScraper):
    def __init__(self, country, number_of_cvs):
        super().__init__(country, number_of_cvs)
        self.base_url = "https://www.linkedin.com/jobs/search/"
    
    def scrape(self):
        """Scrape LinkedIn for resume data"""
        logger.info(f"Starting LinkedIn scraping for {self.country}, targeting {self.number_of_cvs} CVs")
        
        results = []
        driver = self.setup_driver()
        
        try:
            # Perform LinkedIn-specific scraping
            # Note: In a real implementation, you would need to handle authentication and navigate through profiles
            search_url = f"{self.base_url}?keywords=all&location={self.country}"
            driver.get(search_url)
            
            # Wait for the page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "jobs-search__results-list"))
            )
            
            # Scroll down to load more results
            for _ in range(min(5, self.number_of_cvs // 10)):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # Get job links
            job_links = driver.find_elements(By.CSS_SELECTOR, ".job-card-container__link")
            job_urls = [link.get_attribute('href') for link in job_links[:self.number_of_cvs]]
            
            # Visit each job page and extract profile information
            for i, url in enumerate(job_urls):
                if i >= self.number_of_cvs:
                    break
                    
                logger.info(f"Processing LinkedIn profile {i+1}/{len(job_urls)}")
                
                try:
                    driver.get(url)
                    time.sleep(random.uniform(2, 5))  # Random delay to avoid detection
                    
                    # This is where you would extract the profile data
                    # For simulation purposes, we'll create mock data
                    cv_data = self._generate_mock_data(self.country)
                    results.append(cv_data)
                    
                except Exception as e:
                    logger.error(f"Error processing LinkedIn profile {url}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during LinkedIn scraping: {str(e)}")
            return results
        finally:
            driver.quit()
    
    def _generate_mock_data(self, country):
        """Generate mock CV data for demonstration purposes"""
        first_names = ["Ahmed", "Sarah", "Mohamed", "Fatima", "Ali", "Leila", "Omar", "Yasmine", "Karim", "Nour"]
        last_names = ["Ben Ali", "Trabelsi", "Chahed", "Mejri", "Brahmi", "Lahmar", "Bouazizi", "Riahi", "Jelassi", "Sfaxi"]
        universities = [
            "University of Tunis", "Tunisia Polytechnic School", "University of Carthage", 
            "University of Sousse", "University of Sfax", "Mediterranean School of Business",
            "University of Monastir", "University of Manouba", "University of Gabès", "Zitouna University"
        ]
        companies = [
            "Vermeg", "Telnet", "Proxym Group", "Esprit", "Sofrecom", "Sopra HR", 
            "Capgemini", "Ooredoo", "Tunisie Telecom", "Orange Tunisia"
        ]
        skills = [
            "Python", "Java", "JavaScript", "React", "Angular", "Node.js", "Express", 
            "MongoDB", "SQL", "Django", "Flask", "Git", "Docker", "Kubernetes", "AWS", 
            "Machine Learning", "Data Science", "TensorFlow", "PyTorch", "NLP"
        ]
        
        # Generate random data
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        email = f"{name.lower().replace(' ', '.')}@example.com"
        
        experience_years = random.randint(1, 15)
        experience = []
        for i in range(random.randint(1, 3)):
            company = random.choice(companies)
            duration = random.randint(1, 5)
            experience.append({
                "company": company,
                "position": f"Software Engineer{' Senior' if experience_years > 5 else ''}",
                "duration": f"{duration} years",
                "description": f"Worked on various projects using {', '.join(random.sample(skills, 3))}"
            })
        
        education = {
            "university": random.choice(universities),
            "degree": random.choice(["Bachelor's", "Master's", "PhD"]),
            "field": random.choice(["Computer Science", "Software Engineering", "Data Science", "IT"]),
            "graduation_year": 2023 - random.randint(1, 10)
        }
        
        selected_skills = random.sample(skills, random.randint(5, 10))
        
        projects = []
        for i in range(random.randint(1, 3)):
            project_skills = random.sample(selected_skills, min(3, len(selected_skills)))
            projects.append({
                "name": f"Project {chr(65+i)}",
                "description": f"Implemented a system using {', '.join(project_skills)}",
                "technologies": project_skills
            })
        
        return {
            "name": name,
            "email": email,
            "country": country,
            "experience": experience,
            "education": education,
            "skills": selected_skills,
            "projects": projects
        }


class GoogleScraper(BaseScraper):
    def __init__(self, country, number_of_cvs):
        super().__init__(country, number_of_cvs)
    
    def scrape(self):
        """Scrape Google for resume data"""
        logger.info(f"Starting Google scraping for {self.country}, targeting {self.number_of_cvs} CVs")
        
        results = []
        driver = self.setup_driver()
        
        try:
            # Search for resumes/CVs in the specified country
            search_query = f"filetype:pdf OR filetype:doc resume OR cv {self.country}"
            search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            
            driver.get(search_url)
            time.sleep(3)  # Wait for the page to load
            
            # Extract search results
            for page in range(min(self.number_of_cvs // 10 + 1, 5)):  # Limit to 5 pages max
                if len(results) >= self.number_of_cvs:
                    break
                
                # Find all search result links
                links = driver.find_elements(By.CSS_SELECTOR, "a")
                cv_links = []
                
                for link in links:
                    href = link.get_attribute('href')
                    if href and ('pdf' in href or 'doc' in href) and not any(domain in href for domain in ['google.com']):
                        cv_links.append(href)
                
                # Process each CV link
                for i, url in enumerate(cv_links):
                    if len(results) >= self.number_of_cvs:
                        break
                        
                    logger.info(f"Processing CV {len(results)+1}/{self.number_of_cvs} from Google")
                    
                    try:
                        # In a real implementation, you would download and parse the CV here
                        # For demonstration, we'll use mock data
                        cv_data = self._generate_mock_data(self.country)
                        results.append(cv_data)
                        
                    except Exception as e:
                        logger.error(f"Error processing CV from Google {url}: {str(e)}")
                
                # Go to next page if needed
                if page < min(self.number_of_cvs // 10, 4):  # Less than max pages
                    try:
                        next_button = driver.find_element(By.ID, "pnnext")
                        next_button.click()
                        time.sleep(2)
                    except:
                        logger.info("No more pages to process")
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Error during Google scraping: {str(e)}")
            return results
        finally:
            driver.quit()
    
    def _generate_mock_data(self, country):
        """Generate mock CV data for demonstration purposes"""
        first_names = ["Ahmed", "Sarah", "Mohamed", "Fatima", "Ali", "Leila", "Omar", "Yasmine", "Karim", "Nour"]
        last_names = ["Ben Ali", "Trabelsi", "Chahed", "Mejri", "Brahmi", "Lahmar", "Bouazizi", "Riahi", "Jelassi", "Sfaxi"]
        universities = [
            "University of Tunis", "Tunisia Polytechnic School", "University of Carthage", 
            "University of Sousse", "University of Sfax", "Mediterranean School of Business",
            "University of Monastir", "University of Manouba", "University of Gabès", "Zitouna University"
        ]
        companies = [
            "Vermeg", "Telnet", "Proxym Group", "Esprit", "Sofrecom", "Sopra HR", 
            "Capgemini", "Ooredoo", "Tunisie Telecom", "Orange Tunisia"
        ]
        skills = [
            "Python", "Java", "JavaScript", "React", "Angular", "Node.js", "Express", 
            "MongoDB", "SQL", "Django", "Flask", "Git", "Docker", "Kubernetes", "AWS", 
            "Machine Learning", "Data Science", "TensorFlow", "PyTorch", "NLP"
        ]
        
        # Generate random data
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        email = f"{name.lower().replace(' ', '.')}@example.com"
        
        experience_years = random.randint(1, 15)
        experience = []
        for i in range(random.randint(1, 3)):
            company = random.choice(companies)
            duration = random.randint(1, 5)
            experience.append({
                "company": company,
                "position": f"Software Engineer{' Senior' if experience_years > 5 else ''}",
                "duration": f"{duration} years",
                "description": f"Worked on various projects using {', '.join(random.sample(skills, 3))}"
            })
        
        education = {
            "university": random.choice(universities),
            "degree": random.choice(["Bachelor's", "Master's", "PhD"]),
            "field": random.choice(["Computer Science", "Software Engineering", "Data Science", "IT"]),
            "graduation_year": 2023 - random.randint(1, 10)
        }
        
        selected_skills = random.sample(skills, random.randint(5, 10))
        
        projects = []
        for i in range(random.randint(1, 3)):
            project_skills = random.sample(selected_skills, min(3, len(selected_skills)))
            projects.append({
                "name": f"Project {chr(65+i)}",
                "description": f"Implemented a system using {', '.join(project_skills)}",
                "technologies": project_skills
            })
        
        return {
            "name": name,
            "email": email,
            "country": country,
            "experience": experience,
            "education": education,
            "skills": selected_skills,
            "projects": projects
        }
