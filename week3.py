# Week 3 - Naive Bayes Sınıflandırma Alg. %82,96 Doğruluk Değeri

import numpy as np
import pandas as pd # Bir dosyaya erişip içeriğini okumak-yazmak veya baştan sona tasarlamak

# Veri setini yükleme aşaması
train = pd.read_csv("mitbih_train.csv")
# Eğitim verilerini özellikler ve etiketler olarak ayırma
X_train = np.array(train)[:, :187] # Özellikler
y_train = np.array(train)[:, 187]  # Etiketler

# Test Veri setini Yükleme
test = pd.read_csv("mitbih_test.csv")
# Eğitim verilerini özellikler ve etiketler olarak ayırma
X_test = np.array(test)[:, :187] # Özellikler
y_test = np.array(test)[:, 187]  # Etiketler

# Naive Bayes Sınıflandırıcısını kullanarak modelin Eğitimi
from sklearn.naive_bayes import CategoricalNB
gnp = CategoricalNB()
gnp.fit(X_train, y_train)

# Test verileri üzerinde tahmin yapma
y_pred = gnp.predict(X_test)

# Karmaşıklık matrisini oluşturma
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
import matplotlib.pyplot as plt

# Karmaşıklık matrisini görselleştirme
index = ['No' , 'S', 'V', 'F', 'Q'] # Sınıf Etiketleri
columns = ['No' , 'S', 'V', 'F', 'Q'] # Tahmin edilen Sınıflar
cm_df = pd.DataFrame(cm, columns, index)

plt.figure(figsize=(10,6))
# Karmaşıklık matrisini ısı haritası olarak gösterme
sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")
plt.show()

# Modelin doğruluğunu hesaplama ve yazdırma
from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

# Week 3 - Naive Bayes Sınıflandırma Alg. Kod Sonu


# Week 3 - Decision Tree Sınıflandırma Alg. %95,26 Doğruluk Değeri (Kod Başlangıcı)
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükleme aşaması
train = pd.read_csv("mitbih_train.csv")
# Eğitim verilerini özellikler ve etiketler olarak ayırma
X_train = np.array(train)[:, :187] # Özellikler
y_train = np.array(train)[:, 187]  # Etiketler

# Test Veri setini Yükleme
test = pd.read_csv("mitbih_test.csv")
# Eğitim verilerini özellikler ve etiketler olarak ayırma
X_test = np.array(test)[:, :187] # Özellikler
y_test = np.array(test)[:, 187]  # Etiketler

# Decision Tree Sınıflandırıcısını kullanarak modelin Eğitimi
dtc = DecisionTreeClassifier()

# Cross-validation ile modelin performansını değerlendirme
cv_scores = cross_val_score(dtc, X_train, y_train, cv=10)

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Modeli tüm eğitim verisiyle tekrar eğitme
dtc.fit(X_train, y_train)

# Test verileri üzerinde tahmin yapma
y_pred = dtc.predict(X_test)

# Karmaşıklık matrisini oluşturma
cm = confusion_matrix(y_test, y_pred)

# Karmaşıklık matrisini görselleştirme
index = ['No' , 'S', 'V', 'F', 'Q'] # Sınıf Etiketleri
columns = ['No' , 'S', 'V', 'F', 'Q'] # Tahmin edilen Sınıflar
cm_df = pd.DataFrame(cm, columns, index)

plt.figure(figsize=(10,6))
# Karmaşıklık matrisini ısı haritası olarak gösterme
sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")
plt.show()

# Modelin doğruluğunu hesaplama ve yazdırma
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Week 3 - Decision Tree Sınıflandırma Alg. Kod Sonu



# Week 3 - Decision Tree Sınıflandırma Alg. (K-Cross = 10) %95,48 Doğruluk Değeri (Kod Başlangıcı)

import numpy as np
import pandas as pd

# Veri setini yükleme aşaması
train = pd.read_csv("mitbih_train.csv")
# Eğitim verilerini özellikler ve etiketler olarak ayırma
X_train = np.array(train)[:, :187] # Özellikler
y_train = np.array(train)[:, 187]  # Etiketler

# Test Veri setini Yükleme
test = pd.read_csv("mitbih_test.csv")
# Eğitim verilerini özellikler ve etiketler olarak ayırma
X_test = np.array(test)[:, :187] # Özellikler
y_test = np.array(test)[:, 187]  # Etiketler

# Decision Tree Sınıflandırıcısını kullanarak modelin Eğitimi
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

from sklearn.model_selection import cross_val_score
# Cross-validation ile modelin performansını değerlendirme
cv_scores = cross_val_score(dtc, X_train, y_train, cv=10)

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Modeli tüm eğitim verisiyle tekrar eğitme
dtc.fit(X_train, y_train)

# Test verileri üzerinde tahmin yapma
y_pred = dtc.predict(X_test)

# Karmaşıklık matrisini oluşturma
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

# Karmaşıklık matrisini görselleştirme
index = ['No' , 'S', 'V', 'F', 'Q'] # Sınıf Etiketleri
columns = ['No' , 'S', 'V', 'F', 'Q'] # Tahmin edilen Sınıflar
cm_df = pd.DataFrame(cm, columns, index)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
# Karmaşıklık matrisini ısı haritası olarak gösterme
sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")
plt.show()

# Modelin doğruluğunu hesaplama ve yazdırma
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Week 3 - Decision Tree Sınıflandırma Alg. (K-Cross = 10) Kod Sonu
