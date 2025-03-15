# TuniHire ATS Bot

A Telegram bot designed to help collect CV data for training ATS (Applicant Tracking System) systems. This bot can scrape resume data from LinkedIn or Google, process the data, generate CSV files, and train ML models to support the TuniHire platform.

## Features

- Web scraping of CV data from LinkedIn and Google
- Data preprocessing and cleaning
- CSV data export for ATS training 
- PDF report generation with data visualizations
- Machine learning model training and evaluation using:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machines (SVM)
  - AdaBoost (ADD)
- User-friendly Telegram bot interface

## Requirements

- Python 3.7+
- Chrome/Chromium browser (for web scraping)
- Telegram account

## Setup Instructions

1. **Create a Bot on Telegram**:
   - Open Telegram and search for "BotFather."
   - Start a chat with BotFather and send the command `/newbot`.
   - Follow the instructions to set up your bot's name and username.
   - Once your bot is created, BotFather will provide you with a token.

2. **Store the Token Securely**:
   - Create a file named `.env` in the root directory of your project.
   - Add your token to this file in the format: `TELEGRAM_TOKEN=your_telegram_bot_token_here`

3. **Update `.gitignore`**:
   - Add `.env` to your `.gitignore` file to prevent it from being tracked by Git.

4. **Install Dependencies**:
   - Ensure all dependencies are installed using `pip install -r requirements.txt`.

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/TuniHireBot.git
cd TuniHireBot
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Install Chrome/Chromium browser (required for Selenium)

## Demo Video

![Demo Video](Demo/demo.mp4)
## Usage

1. Start the bot by running:
```
python main.py
```

2. In Telegram, find your bot (@TuniHireBot) and start a conversation.

3. Use the following commands:
   - `/start` - Begin a new scraping session
   - `/cancel` - Cancel the current operation
   - `/help` - Show help message

4. Follow the bot's instructions to:
   - Specify target country
   - Set number of CVs to scrape (max 200)
   - Select data source (LinkedIn or Google)

5. After processing, the bot will provide:
   - CSV file with structured CV data
   - PDF report with data analysis
   - ML model evaluation results

## Project Structure

```
TuniHireBot/
├── main.py                # Main bot file
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── bot_modules/           # Bot modules
│   ├── scraper.py         # Web scraping functionality
│   ├── data_processor.py  # Data processing and output generation
│   └── ml_models.py       # ML model implementation and evaluation
├── data/                  # Storage for temporary data
└── output/                # Generated CSV and PDF files
```

## Security Note

This bot requires a Telegram API token to function. Keep your token secure and never share it publicly.

## License

MIT License

## Disclaimer

This tool is intended for educational and research purposes only. Users are responsible for compliance with all applicable laws and website terms of service when using this tool.
