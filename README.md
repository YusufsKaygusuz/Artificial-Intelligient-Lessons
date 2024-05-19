# 🚀🤖 Yapay Zeka Dersi 🦾🚀

<p align="center">
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/cd98b111-b66c-4ddb-b0c4-f62ce0ab8b46" alt="ReLU" width="125"/>
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/7bfa61ee-d340-41b9-8855-dec4c561744f" alt="ReLU" width="200"/> 
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/a4e54abd-9ff4-4d8f-b784-bd0653e9b8f3" alt="ReLU" width="125"/>
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/a90a23b8-0c21-40ee-9617-b17d2858b100" alt="ReLU" width="125"/>
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/705deb43-4977-46c8-8d32-b0c34b4b7b66" alt="ReLU" width="125"/>

</p>


## 📚 İçindekiler
| Hafta | Haftalık İçerik                             |
|-------|--------------------------------------------|
| 📆 Week 1 | [**Iris Veri Seti ile Sınıflandırma**](#week-1-iris-veri-seti-ile-sınıflandırma) |
| 📆 Week 2 | [**Bulaşık Yıkama Süresi Kontrol Sistemi**](#week-2-bulaşık-yıkama-süresi-kontrol-sistemi) |
| 📆 Week 3 | [**Naive Bayes ile Kalp Ritim Tespiti**](#week-3-naive-bayes-ile-kalp-ritim-tespiti) |
| 📆 Week 4 | [**Kalp Ritim Bozukluğu Tespiti ve Hastalıklı Yaprak Analizi**](#week-4-kalp-ritim-bozukluğu-tespiti-ve-hastalıklı-yaprak-analizi) |

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

---

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

---
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

<h3>Naive Bayes Sınıflandırma Alg. <strong>%82,96</strong> Doğruluk Değeri</h3>
<img src="https://github.com/YusufsKaygusuz/Artificial-Intelligient-Lessons/assets/86704802/9b900941-9be1-4e3c-a561-e7a84372e10c" alt="ReLU" width="550"/> 

<h3>Decision Tree Sınıflandırma Alg. <strong>%95,26</strong> Doğruluk Değeri</h3>
<img src="https://github.com/YusufsKaygusuz/Artificial-Intelligient-Lessons/assets/86704802/4119e0ed-e4be-46c4-8429-528d309e4a17" alt="ReLU" width="550"/> 

<h3>Decision Tree Sınıflandırma Alg. (K-Cross = 10) <strong>%95,48</strong> Doğruluk Değeri</h3>
<img src="https://github.com/YusufsKaygusuz/Artificial-Intelligient-Lessons/assets/86704802/62e6f831-72f9-4587-9094-10bc6fc50530" alt="ReLU" width="550"/> 

---

## Week 4: Kalp Ritim Bozukluğu Tespiti ve Hastalıklı Yaprak Analizi

Bu kod, pirinç yaprak hastalıklarını sınıflandırmak için bir makine öğrenimi modeli oluşturur. Aşağıda, kodun her bölümünü ayrıntılı olarak açıkladım.
<h3>Kullanılan Kütüphaneneler</h3>

```python
import numpy as np
import pandas as pd
import os
import PIL.Image as img
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
```

- numpy: Sayısal işlemler için kullanılır.
- pandas: Veri analizi ve veri manipülasyonu için kullanılır.
- os: Dosya ve dizin işlemleri için kullanılır.
- PIL.Image: Görüntü işleme için kullanılır.
- sklearn.model_selection: Veri setini eğitim ve test kümelerine ayırmak için kullanılır.
- sklearn.ensemble: RandomForestClassifier modelini oluşturmak için kullanılır.
- sklearn.metrics: Modelin doğruluğunu ölçmek için kullanılır.

<h3>Dosya ve Dizin İşlemleri</h3>

```python
bakteri_yaprak_yanik = "rice_leaf_diseases/Bacterial leaf blight/"
kahve_nokta = "rice_leaf_diseases/Brown spot/"
yaprak_isi = "rice_leaf_diseases/Leaf smut"

def dosya(yol):
    return [os.path.join(yol, f) for f in os.listdir(yol)]
```

- bakteri_yaprak_yanik, kahve_nokta, yaprak_isi: Farklı hastalık türlerine ait görüntülerin bulunduğu dizinlerin yolları.
- dosya: Belirtilen yoldaki tüm dosyaların tam yolunu döndüren bir fonksiyon.


<h3> Veri Dönüştürme </h3>

```python
def veri_donusturme(klasor_adi, sinif_adi):
    goruntuler = dosya(klasor_adi)
    
    goruntu_sinif = []
    for goruntu in goruntuler:
        goruntu_oku = img.open(goruntu).convert('L')
        gorunu_boyutlandirma = goruntu_oku.resize((28, 28))
        goruntu_donusturme = np.array(gorunu_boyutlandirma).flatten()
        if sinif_adi == "bakteri_yaprak_yanik":
            veriler = np.append(goruntu_donusturme, [0])
        elif sinif_adi == "kahve_nokta":
            veriler = np.append(goruntu_donusturme, [1])
        elif sinif_adi == "yaprak_isi":
            veriler = np.append(goruntu_donusturme, [2])
        else:
            continue
        goruntu_sinif.append(veriler)

    return goruntu_sinif
```

- veri_donusturme: Belirtilen klasördeki görüntüleri okuyup, 28x28 boyutuna getirerek düzleştirir ve sınıf etiketleriyle birlikte bir listeye ekler.

<h3> Verilerin Data Setlerinden Yüklenmesi ve Birleştirilmesi </h3>

```python
yanik_veri = veri_donusturme(bakteri_yaprak_yanik, "bakteri_yaprak_yanik")
yanik_veri_df = pd.DataFrame(yanik_veri)

kahve_nokta_veri = veri_donusturme(kahve_nokta, "kahve_nokta")
kahve_nokta_veri_df = pd.DataFrame(kahve_nokta_veri)

yaprak_isi_veri = veri_donusturme(yaprak_isi, "yaprak_isi")
yaprak_isi_veri_df = pd.DataFrame(yaprak_isi_veri)

tum_veri = pd.concat([yanik_veri_df, kahve_nokta_veri_df, yaprak_isi_veri_df])
```

- Her bir hastalık sınıfı için veri_donusturme fonksiyonu kullanılarak veriler okunur ve bir DataFrame'e dönüştürülür.
- Tüm veriler birleştirilir.


<h3>Giriş ve Çıkış Verilerinin Hazırlanması</h3>

```python
Giris = np.array(tum_veri)[:,:784]
Cikis = np.array(tum_veri)[:,784]
```

- Giris: Görüntü verilerini içerir.
- Cikis: Sınıf etiketlerini içerir.


<h3>Veri Setinin Eğitim(Train) ve test Kümelerine Ayrılması</h3>

```python
Giris_train, Giris_test, Cikis_train, Cikis_test = train_test_split(Giris, Cikis, test_size=0.2, random_state=109)
```

- Veri seti %80 eğitim ve %20 test olacak şekilde ayrılır.



<h3>Modelin Eğitilmesi ve Test Edilmesi</h3>

```python
model = RandomForestClassifier()
model.fit(Giris_train, Cikis_train)
```

- RandomForestClassifier modeli oluşturulur ve eğitim verileriyle eğitilir.


<h3>Tahmin Yapılması ve Doğruluk Ölçümü</h3>

```python
Cikis_pred = model.predict(Giris_test)
print("Doğruluk:", metrics.accuracy_score(Cikis_test, Cikis_pred))
```

- Test verileri üzerinde tahmin yapılır ve modelin doğruluğu ölçülür.


<h3>Özetle Neyi Hedefledik? </h3>

<p>Kod, pirinç yaprak hastalıklarını sınıflandırmak için bir makine öğrenimi modeli oluşturur ve modelin doğruluğunu ölçer. Bu model, görüntüleri gri tonlamalı yapıp, yeniden boyutlandırarak ve düzleştirerek çalışır. RandomForestClassifier kullanılarak hastalık sınıflandırması yapılır ve test verileri üzerinde doğruluk ölçülür. </p>







