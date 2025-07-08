Cyberbullying Detection System (Django-Based Web Application)
=============================================================

Overview
--------
This project is a Django-based web application designed to identify cyberbullying content in user-submitted text. 
It provides a user-friendly web interface where users can enter messages, and the system checks them for offensive or abusive language.

The detection is based on analyzing words from a dataset of bullying terms (dataset.txt, features.txt, etc.).
This is a simple project intended for learning purposes.

Key Features
------------
- Detects cyberbullying words from text input.
- Web-based interface using Django and HTML templates.
- Organized static files (images, CSS).
- Dataset files are included in the project (no need to download separately).

Technologies Used
-----------------
- Python
- Django Framework
- HTML / CSS / JavaScript
- Basic text file handling for word checking

Project Structure (Important Files)
------------------------------------
Cyber/                  --> Django project core (settings, urls)
CyberBullying/          --> Main app (views, models, templates)
static/                 --> Static images and CSS
templates/              --> HTML pages
media/                  --> Media images (optional)
dataset.txt             --> Dataset file with bullying-related words
features.txt            --> Additional feature data
requirements.txt        --> Libraries to install
manage.py               --> Django project runner

Installation & Setup
--------------------

Step 1: Clone the Repository
----------------------------
git clone https://github.com/sharan-karnakanti/Identification-of-cyberbullies.git
cd Identification-of-cyberbullies

Step 2: Setup a Virtual Environment (Recommended)
-------------------------------------------------
It's recommended to use a virtual environment to avoid conflicts.

On Windows:
python -m venv venv
venv\Scripts\activate

On Linux/Mac:
python3 -m venv venv
source venv/bin/activate

Step 3: Install Dependencies
----------------------------
Install Django and other required packages:
pip install -r requirements.txt

Step 4: Run Database Migrations
-------------------------------
Since this is a simple project, database setup may not be critical, but run migrations to avoid issues:
python manage.py migrate

Step 5: Run the Django Server
-----------------------------
python manage.py runserver

Visit http://127.0.0.1:8000/ in your browser.

Running the Project
-------------------
Once the server is running:
- Open the home page in your browser.
- Enter some text in the input box.
- Submit to check whether the text contains cyberbullying terms.
- The result will be displayed on the page.

Example Output
--------------
Input: "You are an idiot"
Output: Cyberbullying detected 🚨

Input: "Have a nice day"
Output: No cyberbullying detected ✅

Future Improvements
-------------------
- Add a machine learning model for better accuracy.
- Build an API for other applications to use.
- Add user authentication for better control.
- Enhance the dataset with more diverse examples.

Author
------
Sharan Kumar

LinkedIn: https://www.linkedin.com/in/k-sharan-kumar/
GitHub: https://github.com/sharan-karnakanti
Email: sharankumar1132002@gmail.com

Disclaimer
----------
This project is for educational and learning purposes only.
Do not deploy it to production without further improvements in detection accuracy and security.
