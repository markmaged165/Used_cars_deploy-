import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

def train_model():
    try:
        # Load the dataset
        data = pd.read_csv(r'D:\fast_api\car-sales.csv')
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        
        # Features and target
        # Based on the CSV structure (assuming columns: Make, Colour, Odometer (KM), Doors, Price)
        # We'll use Price as an object-to-int if it has currency symbols
        if data['Price'].dtype == 'object':
            data['Price'] = data['Price'].str.replace(r'[\$,]', '', regex=True).astype(float).astype(int)

        X = data[['Colour', 'Odometer (KM)', 'Doors', 'Price']]
        y = data['Make']

        # Preprocessing
        categorical_features = ['Colour']
        numeric_features = ['Odometer (KM)', 'Doors', 'Price']

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(), numeric_features)
            ])

        # SVM model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
        ])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit model
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, 'model.joblib')
        print('Model trained and saved successfully as model.joblib')
        
        return model
    except Exception as e:
        print(f"Error training model: {e}")



        



