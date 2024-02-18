# Penguin Classifier

This application uses machine learning to predict penguin species based on input parameters. The model is trained on Palmer's Penguins dataset using a Random Forest classifier.

## Features

- **Predictive Model**: Utilizes Random Forest model trained on island, bill measurements, flipper length, body mass, and sex to predict penguin species.
- **Interactive Interface**: Users input penguin characteristics via a simple form.
- **Feature Importance Visualization**: Displays a bar plot showing importance of features in predicting penguin species.

## Usage

1. **Installation**: Install dependencies with `pip install -r requirements.txt`.
2. **Run Application**: Execute with `streamlit run app.py`.
3. **Input Penguin Characteristics**: Use form interface to input penguin characteristics.
4. **View Prediction**: Predicted species displayed based on input parameters.
5. **Feature Importance**: Bar plot shows feature importance.

## Files

- **app.py**: Streamlit application code.
- **random_forest_penguin.pickle**: Pickled Random Forest classifier.
- **output_penguin.pickle**: Pickled mapping for penguin species.
- **feature_importance.png**: Feature importance visualization.

## Libraries Used

- **Streamlit**: For building interactive web app.
- **Seaborn & Matplotlib**: For data visualization.
- **Pandas**: For data manipulation.
- **Scikit-learn**: For machine learning models.

## Note

- Upload custom penguin dataset via file uploader, ensuring required features.
