import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ruta del dataset
data_path = './data/Berlin_Marathon_Adjusted_AverageTime.csv'
models_path = './models'

# Cargar el dataset
data = pd.read_csv(data_path)

# Preprocesamiento de los datos
data = data.drop(columns=['YEAR'])
runtype_mapping = {'Outdoor': 0, 'Track': 1, 'Trail': 2, 'Treadmill': 3}
data['RunType'] = data['RunType'].map(runtype_mapping)
data['CrossTraining'] = data['CrossTraining'].str.extract(r'(\d+)').astype(float).fillna(0)
data = pd.get_dummies(data, columns=['GENDER'], drop_first=True)
X = data.drop(columns=['Average_MarathonTime', 'Category'])
y = data['Average_MarathonTime']
X = X.dropna()
y = y[X.index]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo de regresión Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Crear directorio de modelos
os.makedirs(models_path, exist_ok=True)

# Guardar el modelo y el escalador
joblib.dump(ridge_model, f'{models_path}/ridge_model.joblib')
joblib.dump(scaler, f'{models_path}/scaler.joblib')

# Evaluar el modelo
y_pred = ridge_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
