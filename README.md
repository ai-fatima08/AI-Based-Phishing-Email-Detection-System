# AI-Based-Phishing-Email-Detection-System

A machine learning-based web application/system that detects phishing emails in real-time using Random Forest classifier with SHAP explanations.

## Features

## User Features
- User Registration & Login (JWT Authentication)
- Real-time email phishing detection
- Scan history with detailed results
- Feedback submission for model improvement
- User dashboard with statistics

## Admin Features
- System-wide analytics dashboard
- View all registered users
- View all scans and feedback
- Delete users (soft delete)
- Monitor system health

## ML Model Features
- Random Forest classifier (200 trees, depth 20)
- TF-IDF vectorization (3000 features)
- SMOTE for class imbalance handling
- SHAP explanations for predictions
- 95%+ accuracy on test data

## Tech Stack

## Backend

| Technology  | Purpose            |
|------------ |------------------- |
| Python 3.9+ | Core language      |
| FastAPI     | REST API framework |
| MySQL       | Database           |
| JWT         | Authentication     |
| bcrypt      | Password hashing   |

## Machine Learning

| Library | Purpose                    |
|---------|----------------------------|
| scikit-learn | Random Forest, TF-IDF |
| NLTK         | Text preprocessing    |
| imb-learn    | SMOTE                 |
| SHAP         | Model explanations    |
| joblib       | Model serialization   |

## Frontend

| Technology | Purpose                   |
|------------|---------------------------|
| HTML5/CSS3 | Structure & styling       |
| JavaScript | API calls & interactivity |

## Setup Instructions
1. Install Python 3.9+
2. Install XAMPP and start MySQL
3. Install requirements: `pip install -r requirements.txt`
4. Create database: Run `database/schema.sql`
5. Train model: `python backend/train_model.py`
6. Start server: `python backend/main.py`
7. Open frontend: `frontend/index.html`

## API Endpoints

## Authentication

|Method  |	Endpoint          | Description      |
| POST   | /api/auth/register | User registration|
| POST   | /api/auth/login	  | User login       |
| POST   | /api/admin/login	  | Admin login      | 

## User

|Method	| Endpoint	             | Description     |
| GET	| /api/users/me	         |Get profile      |
| PUT	| /api/users/me	         |Update username  |
| GET	| /api/user/{id}/scans	 |Get scan history |
| POST	| /api/analyze	         |Analyze email    |
| POST	| /api/feedback	         |Submit feedback  |

## Admin

|Method	 |Endpoint      	     | Description       |
| GET	 | /api/admin/stats	     | System statistics |
| GET	 | /api/admin/users	     | All users         |
| GET	 | /api/admin/scans	     | All scans         |
| DELETE | /api/admin/users/{id} | Delete user       |
| GET	 | /api/admin/feedback	 | All feedback      |

## ML

| Method   |  Endpoint	     | Description       |
| POST	   | /api/explain	 | SHAP explanation  |
| GET	   | /api/model/info | Model info        |
| GET	   | /api/health	 | Health check      |
