# 🚀🤖 Yapay Zeka Dersi 🦾🚀

<p align="center">
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/cd98b111-b66c-4ddb-b0c4-f62ce0ab8b46" alt="ReLU" width="150"/>
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/7bfa61ee-d340-41b9-8855-dec4c561744f" alt="ReLU" width="200"/> 
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/a4e54abd-9ff4-4d8f-b784-bd0653e9b8f3" alt="ReLU" width="150"/>
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/a90a23b8-0c21-40ee-9617-b17d2858b100" alt="ReLU" width="150"/>
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/705deb43-4977-46c8-8d32-b0c34b4b7b66" alt="ReLU" width="150"/>

</p>


## 📚 İçindekiler
| Hafta | Haftalık İçerik                             |
|-------|--------------------------------------------|
| 📆 Week 1 | [**Iris Veri Seti ile Sınıflandırma**](#week-1-iris-veri-seti-ile-sınıflandırma) |

## Week 1: Iris Veri Seti ile Sınıflandırma

Bu proje, Python dilinde scikit-learn kütüphanesini kullanarak Iris veri setini kullanarak K En Yakın Komşu (K Neighbors) ve Karar Ağacı (Decision Tree) sınıflandırma algoritmalarını nasıl uygulayacağınızı adım adım göstermektedir.

<h3>Iris Veri Seti</h3>

İris veri seti, bitki bilimi alanında yaygın olarak kullanılan bir veri setidir. Üç farklı türde (setosa, versicolor, virginica) 150 adet iris çiçeği örneğini içerir. Her bir örnek için dört özellik (uzunluk ve genişlik gibi) mevcuttur.

<h3>K En Yakın Komşu (K Neighbors) Algoritması</h3>

K En Yakın Komşu algoritması, bir veri noktasını sınıflandırmak için komşularının etiketlerini kullanır. Bu proje, K En Yakın Komşu algoritması kullanarak Iris veri setini sınıflandırmayı göstermektedir.

<h3>Kurulum</h3>

Bu projeyi çalıştırmak için Python ve scikit-learn kütüphanesinin yüklü olması gerekir. İlgili kütüphaneleri yüklemek için terminale şu komutu yazabilirsiniz:

```python
pip install scikit-learn seaborn pandas matplotlib
```

<h2> Kod Analizi </h2>
<h3>Veri Seti Yükleme</h3>
İlk adımda, sklearn.datasets modülünden load_iris() fonksiyonunu kullanarak Iris veri setini yüklüyoruz.

```python
from sklearn.datasets import load_iris
iris = load_iris()
```

<h3>Veri Seti Hakkında Bilgiler</h3>
Daha sonra, yüklenen veri setinin özellik adlarını, hedef sınıf adlarını, hedef sınıf dizisini ve veri noktalarını yazdırıyoruz.

```python
print (iris.feature_names)
print (iris.target_names)
print (iris.target)
print (iris.data)
```

<h3>Veri Setini Eğitim ve Test Setlerine Bölme</h3>
Veri setini eğitim ve test setlerine bölmek için train_test_split() fonksiyonunu kullanıyoruz.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
```

<h3>K En Yakın Komşu Modeli Oluşturma ve Eğitme</h3>
K En Yakın Komşu sınıflandırma modelini oluşturmak için KNeighborsClassifier() sınıfını kullanıyoruz ve ardından eğitim verilerini bu modele uyum sağlıyoruz.

```python
from sklearn.neighbors import KNeighborsClassifier
model =  KNeighborsClassifier()
model.fit(X_train,Y_train)
```

<h3>Model Performansını Değerlendirme</h3>
Eğitilen modelin performansını değerlendirmek için test seti üzerinde tahminler yaparak bir hata matrisi oluşturuyoruz ve bu matrisi yazdırıyoruz.

```python
Y_tahmin = model.predict(X_test)
from sklearn.metrics import confusion_matrix
hata_matrisi = confusion_matrix(Y_test, Y_tahmin)
print(hata_matrisi)
```

<h3>Hata Matrisini Görselleştirme</h3>
Son olarak, oluşturduğumuz hata matrisini bir ısı haritası olarak görselleştiriyoruz.

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
index = ['setosa','versicolor','virginica'] 
columns = ['setosa','versicolor','virginica'] 
hata_goster = pd.DataFrame(hata_matrisi,columns,index)                      
plt.figure(figsize=(10,6))  
sns.heatmap(hata_goster, annot=True)
```
