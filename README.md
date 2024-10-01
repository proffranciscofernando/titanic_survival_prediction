# Titanic Survival Prediction

## Overview

This project showcases a machine learning pipeline built with Scikit-Learn to predict passenger survival on the Titanic. It encompasses data preprocessing, model training, hyperparameter optimisation using Grid Search with cross-validation, and an interactive interface for making predictions based on user input.

Note: The project includes two versions of the main notebook:

- main.ipynb: Written in British English.
- main_pt_BR.ipynb: Written in Brazilian Portuguese.

## Features

- **Data Preprocessing:** Handles missing values, encodes categorical variables, and scales numerical features.
- **Multiple Classification Models:** Implements Logistic Regression, Random Forest, SVM, KNN, and Decision Trees.
- **Hyperparameter Optimisation:** Utilises Grid Search with cross-validation to find the best model parameters.
- **Model Evaluation:** Assesses models using Accuracy, Precision, Recall, and F1-Score.
- **Model Persistence:** Saves the best-performing model for future use.
- **Interactive Predictions:** Provides an interface for users to input new data and receive survival predictions.
- **Clean Output:** Suppresses non-critical warnings for a streamlined notebook experience.

## Dataset

- **Source:** [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic/data)
- **Features Used:**
  - `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
  - `Sex`: Gender of the passenger
  - `Age`: Age of the passenger
  - `Fare`: Fare paid by the passenger (**unit:** British Pounds £)
  - `Survived`: Survival status (0 = No, 1 = Yes)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ffrpereira/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

*Alternatively, install dependencies individually:*
```bash
pip install numpy pandas scikit-learn joblib
```

## Usage

1. **Open the Notebook:**
   - Navigate to `main.ipynb` or `main_pt_BR.ipynb`  and open it in Jupyter Notebook or [Google Colab](https://colab.research.google.com/).

2. **Run the Cells:**
   - Execute each cell sequentially to load data, preprocess, train models, optimise hyperparameters, and evaluate performance.

3. **Interactive Predictions:**
   - In the **8.1 Making Predictions with New User Input** section, input passenger details when prompted to receive survival predictions.

## Project Structure

```
titanic-survival-prediction/
│
├── main.ipynb            # Main Jupyter Notebook (British English)
├── main_pt_BR.ipynb      # Main Jupyter Notebook (Brazilian Portuguese)
├── titanic.csv           # Titanic Dataset
├── README.md             # Project Documentation
└── requirements.txt      # Python Dependencies
```

## Contributing

Contributions are welcome! Please fork the repository, create a new branch for your feature or bugfix, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
