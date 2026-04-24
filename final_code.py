import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, f1_score
 
#Preprocessing
def pre_process_regression(filename):
    """Preprocess data for regression task.
    
    Args:
        filename (str): Path to CSV file containing raw data
        
    Returns:
        pd.DataFrame: Processed DataFrame with engineered features and cleaned columns
    
    """
    # Load data with 'id' as index
    df = pd.read_csv(filename, index_col='id')

    # Handle datetime conversion 
    df['date_sold'] = pd.to_datetime(df['date_sold'])
    df['year_sold'] = df['date_sold'].dt.year

    # Drop irrelevant columns
    drop_cols = [
        'suburb_lat','date_sold','ethnic_breakdown','suburb_elevation',
        'cash_rate','suburb_lng','suburb_sqkm','suburb_population','median_house_rent_per_week',
        'median_apartment_rent_per_week','suburbpopulation','public_housing_pct','postcode',
        'nearest_train_station','highlights_attractions', 'ideal_for', 'traffic', 'public_transport', 'affordability_rental',
        'affordability_buying', 'nature', 'noise', 'things_to_see_do', 'family_friendliness', 'pet_friendliness', 'safety', 'overall_rating'
    ]
    df.drop(drop_cols, axis=1, inplace=True)

    # String columns
    for col in ['suburb', 'type', 'region']:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Feature Engineering
    # Commute time : New Feature
    df['time_to_cbd_public_transport_town_hall_st'] = df['time_to_cbd_public_transport_town_hall_st'].fillna(df['time_to_cbd_public_transport_town_hall_st'].mean())
    df['time_to_cbd_driving_town_hall_st'] = df['time_to_cbd_driving_town_hall_st'].fillna(df['time_to_cbd_driving_town_hall_st'].mean())
    df['commute_time'] = (df['time_to_cbd_public_transport_town_hall_st'] + df['time_to_cbd_driving_town_hall_st']) / 2

    # Suburb median price
    def choose_median_price(row):
        property_type = str(row['type']).lower()
        if property_type == 'house':
            return row['suburb_median_house_price']
        elif 'apartment' in property_type:
            return row['suburb_median_apartment_price']
        else:
            return (row['suburb_median_apartment_price'] + row['suburb_median_house_price']) / 2
    # To handle missing values
    df['suburb_median_price'] = df.apply(choose_median_price, axis=1)

    # Add new features
    df['num_of_rooms'] = df['num_bed'] + df['num_bath']
    df['is_sold_in_2021'] = (df['year_sold'] == 2021).astype(int)
    df['years_diff'] = 2022 - df['year_sold']
    df['inverse_cbd_distance'] = 1 / (df['km_from_cbd'] + 1)
    df['is_cbd'] = (df['km_from_cbd'] < 10).astype(int)


    # Region mapping based on training set grouped by region and mean price 
    region_map = {
        'South West': 0, 'Western Suburbs': 1, 'Hills Shire': 2, 'Inner South': 3, 'Sutherland Shire': 4,
        'Southern Suburbs': 5, 'Northern Suburbs': 6, 'Sydney City': 7, 'Inner West': 8, 'Inner East': 9,
        'Northern Beaches': 10, 'North Shore': 11, 'Upper North Shore': 12, 'Lower North Shore': 13, 'Eastern Suburbs': 14
    }
    if 'region' in df.columns:
        df['region_group'] = df['region'].map(region_map)
        df['low_suburb_price'] = df['region_group'].isin([0,1]).astype(int)
        df['high_suburb_price'] = df['region_group'].isin([13,14]).astype(int)

    # Drop redundant
    df.drop(['year_sold','km_from_cbd','suburb_median_apartment_price','suburb_median_house_price','suburb', 'type', 'region',
    'time_to_cbd_public_transport_town_hall_st','time_to_cbd_driving_town_hall_st'], axis=1, inplace=True)

    return df


def pre_process_classification(filename):
    """Preprocess data for classification task.
    
    Args:
        filename (str): Path to CSV file containing raw data
        
    Returns:
        pd.DataFrame: Processed DataFrame with features optimized for classification
    
    """
    # Load data
    df = pd.read_csv(filename, index_col='id')
    # Drop irrelevant columns
    drop_cols = [
        'suburb_lat','date_sold','ethnic_breakdown','suburb_elevation','suburb','region','avg_years_held',
        'cash_rate','suburb_lng','suburb_sqkm','suburb_population','suburbpopulation','public_housing_pct','postcode',
        'nearest_train_station','highlights_attractions', 'ideal_for', 'traffic', 'public_transport', 'affordability_rental',
        'suburb_median_income', 'property_inflation_index', 'affordability_buying', 'nature', 'noise', 'things_to_see_do',
        'family_friendliness', 'pet_friendliness', 'safety', 'overall_rating'
    ]
    df.drop(drop_cols, axis=1, inplace=True, errors='ignore')

    
    # Add new features
    df['num_of_rooms'] = df['num_bed'] + df['num_bath']
    df['inverse_cbd_distance'] = 1 / (df['km_from_cbd'] + 1)

    # Drop redundant
    df.drop(['time_to_cbd_public_transport_town_hall_st','time_to_cbd_driving_town_hall_st','km_from_cbd'], axis=1, inplace=True)

    return df

# Regression Model
# =========================
def regression_hgb(train_df, test_df, features, target):
    """Train HistGradientBoostingRegressor and predict prices.
    
    Args:
        train_df (pd.DataFrame): Preprocessed training data
        test_df (pd.DataFrame): Preprocessed test data
        features (list): List of feature column names
        target (str): Name of target column (price)
        
    Returns:
        np.ndarray: Array of predicted prices for test set
    
    Model Configuration:
        - Uses MAE loss function
        - Includes regularization through max_depth
        - Balanced learning rate for convergence
    """
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    # Train the model
    model = HistGradientBoostingRegressor(
        learning_rate=0.15,
        max_iter=500,
        max_depth=7,
        loss='absolute_error',
        random_state=42
    )
    model.fit(X_train, y_train)
    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(test_df[target], y_test_pred)
    
    print("\nRegression Metrics:")
    print(f" Training MAE: ${train_mae:,.2f}")
    print(f" Test MAE:     ${test_mae:,.2f}")
    
    return y_test_pred

# =========================
# Classification Model
# =========================
def classify_property_type_rf(train_df, test_df, features, target='type'):

    """Train RandomForestClassifier to predict property types.
    
    Args:
        train_df (pd.DataFrame): Preprocessed training data
        test_df (pd.DataFrame): Preprocessed test data
        features (list): List of feature column names
        target (str): Name of target column (default: 'type')
        
    Returns:
        np.ndarray: Decoded property type predictions for test set
    
    Model Features:
        - Uses class weighting for imbalance correction
        - Includes feature importance analysis
        - Encodes target labels automatically
    """

    le = LabelEncoder()
    train_df['type_encoded'] = le.fit_transform(train_df[target])
    test_df['type_encoded'] = le.transform(test_df[target]) 
    
    X_train = train_df[features]
    y_train = train_df['type_encoded']
    X_test = test_df[features]
    
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Generate predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    train_f1 = f1_score(y_train, train_preds, average='weighted', zero_division=1)
    test_f1 = f1_score(test_df['type_encoded'], test_preds, average='weighted', zero_division=1)
    
    print("\n Classification Metrics:")
    print(f" Training F1: {train_f1:.3f}")
    print(f" Test F1:     {test_f1:.3f}")
    
    return le.inverse_transform(test_preds)

# =========================
# Main function
# =========================
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 z{id}.py train.csv test.csv")
        sys.exit(1)

    # Get script and output file names
    script_name = os.path.basename(sys.argv[0])
    id_part = script_name.split('.')[0]  # e.g., 'z1234567'
    regression_out = f"{id_part}.regression.csv"
    classification_out = f"{id_part}.classification.csv"

    train_csv = sys.argv[1]
    test_csv = sys.argv[2]

    # Regression
    train_df_reg = pre_process_regression(train_csv)
    test_df_reg = pre_process_regression(test_csv)
    
    features_reg = [col for col in train_df_reg.columns if col != 'price']
    # Predict prices
    y_test_pred = regression_hgb(train_df_reg, test_df_reg, features_reg, 'price')
    # Write regression output
    regression_df = pd.DataFrame({
        'id': test_df_reg.index,
        'price': y_test_pred
    })
    regression_df.to_csv('z5521431.regression.csv', index=False)

    # Classification
    train_df_cls = pre_process_classification(train_csv)
    test_df_cls = pre_process_classification(test_csv)
    
    
    features_cls = [col for col in train_df_cls.columns if col != 'type']
    # Predict types
    type_preds = classify_property_type_rf(train_df_cls, test_df_cls, features_cls, 'type')
    # Write classification output
    classification_df = pd.DataFrame({
        'id': test_df_cls.index,
        'type': type_preds
    })
    classification_df.to_csv('z5521431.classification.csv', index=False)

if __name__ == "__main__":
    main()
