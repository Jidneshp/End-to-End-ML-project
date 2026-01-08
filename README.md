# End-to-End ML Project

## Overview

This is a comprehensive end-to-end machine learning project that predicts student exam performance based on various demographic and educational factors. The project demonstrates a complete ML workflow from data preprocessing, model training, and evaluation to deployment via a Flask web application.

## Features

- **Data Preprocessing**: Automated data cleaning and feature scaling
- **Machine Learning Models**: Implementation of multiple algorithms including:
  - CatBoost
  - XGBoost
  - Scikit-learn models
- **Web Application**: Flask-based REST API for real-time predictions
- **Production-Ready Code**: Modular architecture with proper configuration management
- **Model Persistence**: Serialized models using dill for easy deployment

## Project Structure

```
End-to-End-ML-project/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   └── utils.py
├── app.py                 # Flask application
├── setup.py              # Package configuration
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jidneshp/End-to-End-ML-project.git
   cd End-to-End-ML-project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

```bash
python app.py
```

The application will start on `http://0.0.0.0:5000`. Access the web interface to make predictions by providing:
- Gender
- Race/Ethnicity
- Parent's Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

### Making Predictions

The predict pipeline automatically handles:
- Feature transformation
- Standard scaling
- Model inference

## Technologies Used

- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **CatBoost**: Gradient boosting framework
- **XGBoost**: Extreme gradient boosting
- **Flask**: Web framework for deployment
- **Matplotlib & Seaborn**: Data visualization

## Model Performance

The project implements multiple models and selects the best performer based on evaluation metrics. Models are compared using appropriate metrics for regression tasks (MAE, MSE, R² score).

## Configuration

### Project Metadata
- **Version**: 0.0.1
- **Author**: Jidnesh
- **Email**: jidneshpatil866@gmail.com

### Dependencies
All required packages are listed in `requirements.txt` and include:
- Core ML libraries: pandas, numpy, scikit-learn
- Advanced boosting: xgboost, catboost
- Visualization: matplotlib, seaborn
- Web framework: flask
- Utilities: dill (for model serialization)

## Future Enhancements

- [ ] Add model explainability features (SHAP values)
- [ ] Implement cross-validation framework
- [ ] Add unit tests and integration tests
- [ ] Deploy to cloud platforms (AWS, GCP, Azure)
- [ ] Add model monitoring and performance tracking
- [ ] Implement CI/CD pipeline

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available under the MIT License.

## Author

**Jidnesh Patil**
- GitHub: [@Jidneshp](https://github.com/Jidneshp)
- Email: jidneshpatil866@gmail.com

---
