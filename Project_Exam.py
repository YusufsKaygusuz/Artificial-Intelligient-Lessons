# Regression (Tahminleme) için Kodlar % 96 Doğruluk Değeri
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# Veri Setinin Yüklenmesi
dataset = pd.read_csv('emission_dataset.csv')
print(dataset.info())
print(dataset.isnull().sum())

"""
#Plot total emissions by country
plt.figure(figsize=(14, 7))
total_emissions_by_country = dataset.groupby('country_or_area')['value'].sum().reset_index()
sns.barplot(x='country_or_area', y='value', data=total_emissions_by_country)
plt.title('Ülkelere Göre Toplam Emisyon Miktarı')
plt.xlabel('Ülke')
plt.ylabel('Toplam Emisyon Miktarı')
plt.xticks(rotation=90)
plt.show()

# Plot total emissions by category
plt.figure(figsize=(10, 6))
total_emissions_by_category = dataset.groupby('category')['value'].sum().reset_index()
sns.barplot(x='category', y='value', data=total_emissions_by_category)
plt.title('Kategorilere Göre Toplam Emisyon Miktarı')
plt.xlabel('Kategori')
plt.ylabel('Toplam Emisyon Miktarı')
plt.xticks(rotation=45)
plt.show()

# Plot emissions over years by category
plt.figure(figsize=(14, 7))
sns.lineplot(data=dataset, x='year', y='value', hue='category')
plt.title('Emisyon Kategorilerinin Yıllara Göre Dağılımı')
plt.xlabel('Yıl')
plt.ylabel('Emisyon Değeri')
plt.legend(title='Kategori', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
"""

# Verilerin Hazırlanması
X = dataset.drop('value', axis=1)
y = dataset['value']

# Kategorik özellikleri aynı formatta kodla
X_encoded = pd.get_dummies(X, columns=['country_or_area', 'category'])

# Ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# XGBoost Regressor uygulama
model = XGBRegressor()

# Cross Validation (K değeri 10) uygulama
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_mse = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
cv_mae = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')

print(f'10-Fold Cross Validation MSE Scores: {cv_mse}')
print(f'Mean MSE Score: {np.mean(cv_mse):.2f}')
print(f'Standard Deviation of MSE Scores: {np.std(cv_mse):.2f}')
print(f'10-Fold Cross Validation MAE Scores: {cv_mae}')
print(f'Mean MAE Score: {np.mean(cv_mae):.2f}')
print(f'Standard Deviation of MAE Scores: {np.std(cv_mae):.2f}')

# Verileri %50 eğitim ve %50 test verileri olacak şekild ayırma
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.5, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model eğitim adımı
model.fit(X_train_scaled, y_train)

# Tahmin Adımı
y_pred = model.predict(X_test_scaled)

# Tahmin etme metrikleri
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Test Mean Squared Error: {mse:.2f}')
print(f'Test Mean Absolute Error: {mae:.2f}')

# Karmaşıklık Matrisinin Oluşturulması
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Tahmin Edilen vs Gerçek Değerler')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.show()

# Accuracy değerinin oluşturulması
tolerance = 0.1 * np.mean(y_test)  # 10% of mean value of y_test
accuracy = np.mean(np.abs(y_pred - y_test) <= tolerance)
print(f'Test Accuracy within tolerance: {accuracy:.2f}')

# Hata analizi
error = y_pred - y_test
error_ranges = [-np.inf, -tolerance, tolerance, np.inf]
error_labels = ['Large Underestimate', 'Within Tolerance', 'Large Overestimate']
error_categories = pd.cut(error, bins=error_ranges, labels=error_labels)

# Confusion matrix'in ekrana çizdirilmesi
error_summary = pd.value_counts(error_categories).sort_index()
print('\nError Summary:')
print(error_summary)

# Hata Özeti Çıktısı
plt.figure(figsize=(10, 6))
sns.barplot(x=error_summary.index, y=error_summary.values)
plt.title('Error Analysis')
plt.xlabel('Error Category')
plt.ylabel('Frequency')
plt.show()
# Regression (Tahminleme) için Kodlar % 96 Doğruluk Değeri Sonu



# Classification (Sınıflandırma) için Doğru Kodlar % 90 Doğruluk Değeri
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

dataset = pd.read_csv('emission_dataset.csv')
print(dataset.info())
print(dataset.isnull().sum())

X = dataset.drop('value', axis=1)
y = dataset['value']

bins = [0, 100, 500, np.inf] 
labels = ['low', 'medium', 'high']
y_binned = pd.cut(y, bins=bins, labels=labels)

X_encoded = pd.get_dummies(X, columns=['country_or_area', 'category'])

le = LabelEncoder()
y_encoded = le.fit_transform(y_binned)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

model = XGBClassifier()

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=kf, scoring='accuracy')

print(f'10-Fold Cross Validation Accuracy Scores: {cv_scores}')
print(f'Mean Accuracy Score: {np.mean(cv_scores):.2f}')
print(f'Standard Deviation of Accuracy Scores: {np.std(cv_scores):.2f}')

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.5, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=labels)

print(f'Test Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)

plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Classification (Sınıflandırma) için Doğru Kodlar % 90 Doğruluk Değeri Sonu

