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
| 📆 Week 2 | [**Bulaşık Yıkama Süresi Kontrol Sistemi**](#week-2-bulaşık-yıkama-süresi-kontrol-sistemi) |
| 📆 Week 3 | [**Naive Bayes ile Kalp Ritim Tespiti**](#week-3-naive-bayes-ile-kalp-ritim-tespiti) |

## Week 1: Iris Veri Seti ile Sınıflandırma

Bu proje, Python dilinde scikit-learn kütüphanesini kullanarak Iris veri setini kullanarak K En Yakın Komşu (K Neighbors) ve Karar Ağacı (Decision Tree) sınıflandırma algoritmalarını nasıl uygulayacağınızı adım adım göstermektedir.

<h3>☘️ Iris Veri Seti</h3>

İris veri seti, bitki bilimi alanında yaygın olarak kullanılan bir veri setidir. Üç farklı türde (setosa, versicolor, virginica) 150 adet iris çiçeği örneğini içerir. Her bir örnek için dört özellik (uzunluk ve genişlik gibi) mevcuttur.

<h3>🦾 K En Yakın Komşu (K Neighbors) Algoritması</h3>

K En Yakın Komşu algoritması, bir veri noktasını sınıflandırmak için komşularının etiketlerini kullanır. Bu proje, K En Yakın Komşu algoritması kullanarak Iris veri setini sınıflandırmayı göstermektedir.

<h3>🛠️ Kurulum</h3>

Bu projeyi çalıştırmak için Python ve scikit-learn kütüphanesinin yüklü olması gerekir. İlgili kütüphaneleri yüklemek için terminale şu komutu yazabilirsiniz:

```python
pip install scikit-learn seaborn pandas matplotlib
```

<h2>🔎 Kod Analizi </h2>
<h3>1. Veri Seti Yükleme</h3>
İlk adımda, sklearn.datasets modülünden load_iris() fonksiyonunu kullanarak Iris veri setini yüklüyoruz.

```python
from sklearn.datasets import load_iris
iris = load_iris()
```

<h3>2. Veri Seti Hakkında Bilgiler</h3>
Daha sonra, yüklenen veri setinin özellik adlarını, hedef sınıf adlarını, hedef sınıf dizisini ve veri noktalarını yazdırıyoruz.

```python
print (iris.feature_names)
print (iris.target_names)
print (iris.target)
print (iris.data)
```

<h3>3. Veri Setini Eğitim ve Test Setlerine Bölme</h3>
Veri setini eğitim ve test setlerine bölmek için train_test_split() fonksiyonunu kullanıyoruz.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
```

<h3>4. K En Yakın Komşu Modeli Oluşturma ve Eğitme</h3>
K En Yakın Komşu sınıflandırma modelini oluşturmak için KNeighborsClassifier() sınıfını kullanıyoruz ve ardından eğitim verilerini bu modele uyum sağlıyoruz.

```python
from sklearn.neighbors import KNeighborsClassifier
model =  KNeighborsClassifier()
model.fit(X_train,Y_train)
```

<h3>5. Model Performansını Değerlendirme</h3>
Eğitilen modelin performansını değerlendirmek için test seti üzerinde tahminler yaparak bir hata matrisi oluşturuyoruz ve bu matrisi yazdırıyoruz.

```python
Y_tahmin = model.predict(X_test)
from sklearn.metrics import confusion_matrix
hata_matrisi = confusion_matrix(Y_test, Y_tahmin)
print(hata_matrisi)
```

<h3>6. Hata Matrisini Görselleştirme</h3>
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


## Week 2: Bulaşık Yıkama Süresi Kontrol Sistemi
Bu hafta, bulanıklık mantığı (fuzzy logic) kullanarak bulaşık miktarı ve kirlilik seviyesi gibi girdi değişkenlerine dayanarak bulaşık yıkama süresini belirleyen bir kontrol sistemi oluşturulur.

<h2>🔎 Kod Analizi</h2>
<h3>1. İlgili kütüphanelerin yüklenmesi</h3>

```python
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
```

<h3>2. Giriş ve çıkış değişkenlerinin tanımlanması</h3>

```python
bulaşık_miktarı = ctrl.Antecedent(np.arange(0, 100, 1), 'bulaşık miktarı')
kirlilik = ctrl.Antecedent(np.arange(0, 100, 1), 'kirlilik seviyesi')
yıkama_süresi = ctrl.Consequent(np.arange(0, 180, 1), 'yıkama süresi')
```

<h3>3. Üyelik fonksiyonlarının tanımlanması</h3>

```python
bulaşık_miktarı['az'] = fuzz.trimf(bulaşık_miktarı.universe, [0, 0, 30])
bulaşık_miktarı['normal'] = fuzz.trimf(bulaşık_miktarı.universe, [10, 30, 60])
bulaşık_miktarı['çok'] = fuzz.trimf(bulaşık_miktarı.universe, [50, 60, 100])

kirlilik['az'] = fuzz.trimf(kirlilik.universe, [0, 0, 30])
kirlilik['normal'] = fuzz.trimf(kirlilik.universe, [10, 30, 60])
kirlilik['çok'] = fuzz.trimf(kirlilik.universe, [50, 60, 100])

yıkama_süresi['kısa'] = fuzz.trimf(yıkama_süresi.universe, [0, 0, 50])
yıkama_süresi['normal'] = fuzz.trimf(yıkama_süresi.universe, [40, 50, 100])
yıkama_süresi['uzun'] = fuzz.trimf(yıkama_süresi.universe, [60, 80, 180])
```

<h3>4. Kuralların Tanımlanması</h3>

```python
kural1 = ctrl.Rule(bulaşık_miktarı['az'] & kirlilik['az'], yıkama_süresi['kısa'])
kural2 = ctrl.Rule(bulaşık_miktarı['normal'] & kirlilik['az'], yıkama_süresi['normal'])
```

<h3>5. Kontrol sistemi ile simülasyon oluşturulması</h3>
  
```python
kontrol_sistemi = ctrl.ControlSystem([kural1, kural2, ..., kural9])
model = ctrl.ControlSystemSimulation(kontrol_sistemi)
```

<h3>6. Girdi değerleri atanır ve çıktı hesaplanır</h3>
  
```python
model.input['bulaşık miktarı'] = 50
model.input['kirlilik seviyesi'] = 80
model.compute()
```

<h3>7. Sonuç yazdırılır</h3>

```python
print(model.output['yıkama süresi'])
```

Bu betik, bulanıklık mantığı kullanarak bulaşık yıkama süresini belirler. Bu sayede karmaşık sistemlerdeki belirsizliği ve doğrusal olmayan ilişkileri modellemek için kullanılabilir.




## Week 3: Naive Bayes ile Kalp Ritim Tespiti

Bu proje, elektrokardiyogram (EKG) verilerini kullanarak Naive Bayes sınıflandırıcısını uygulamayı amaçlar. EKG sinyalleri, kalp ritminin analizinde kullanılan temel verilerdir.

<h3>Amaç</h3>

EKG sinyallerini işleyerek, sinyaldeki farklı aritmileri (kalp ritim bozuklukları) sınıflandırmak.
Naive Bayes sınıflandırıcısını kullanarak aritmileri doğru bir şekilde tanımlamak.

<h3>Adımlar</h3>

Veri Yükleme: Eğitim ve test veri setleri pandas kütüphanesi kullanılarak yüklenir.
Veri Hazırlığı: Veri setleri özellikler ve etiketler olarak ayrılır.
Model Eğitimi: Naive Bayes sınıflandırıcısı kullanılarak model eğitilir.
Tahminler: Test veri seti üzerinde tahminler yapılır.
Değerlendirme: Modelin performansı, karmaşıklık matrisi ve doğruluk metriği kullanılarak değerlendirilir.

<h3>Kullanılan Kod Parçaları</h3>

<h4>Veri setlerini yükleme ve özellikler ile etiketlerin ayrılması:</h4>

```python
import numpy as np
import pandas as pd

train = pd.read_csv("mitbih_train.csv")
X_train = np.array(train)[:, :187] # Özellikler
y_train = np.array(train)[:, 187]  # Etiketler

test = pd.read_csv("mitbih_test.csv")
X_test = np.array(test)[:, :187] # Özellikler
y_test = np.array(test)[:, 187]  # Etiketler
```

<h4>Model eğitimi ve tahminlerin yapılması:</h4>

```python
from sklearn.naive_bayes import CategoricalNB

gnb = CategoricalNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
```


<h4>Değerlendirme ve sonuçların görselleştirilmesi:</h4>

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

index = ['No' , 'S', 'V', 'F', 'Q']
columns = ['No' , 'S', 'V', 'F', 'Q']
cm_df = pd.DataFrame(cm, columns, index)

plt.figure(figsize=(10,6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")
plt.show()

from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
```


<h4>Kullanılan Kütüphaneler</h4>

<p>♾️numpy: Sayısal hesaplamalar için kullanılır.</p>
<p>📚pandas: Veri manipülasyonu ve analizi için kullanılır.</p>
<p>📖sklearn: Makine öğrenimi algoritmalarını ve metriklerini içerir.</p>
<p>🎨seaborn: Veri görselleştirmesi için kullanılır.</p>
<p>🎨matplotlib: Grafik çizimleri için kullanılır.</p>
