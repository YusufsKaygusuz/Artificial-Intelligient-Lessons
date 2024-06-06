# 🚀🤖 Yapay Zeka Dersi 🦾🚀

<p align="center">
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/a4e54abd-9ff4-4d8f-b784-bd0653e9b8f3" alt="ReLU" width="125"/>
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/a90a23b8-0c21-40ee-9617-b17d2858b100" alt="ReLU" width="125"/>
<img src="https://github.com/YusufsKaygusuz/Artificial-Intelligient-Lessons/assets/86704802/cf3cd24d-6feb-47f0-9117-d9d305b6a7d7" alt="ReLU" width="150"/>
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/705deb43-4977-46c8-8d32-b0c34b4b7b66" alt="ReLU" width="125"/>
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/7bfa61ee-d340-41b9-8855-dec4c561744f" alt="ReLU" width="200"/> 
<img src="https://github.com/YusufsKaygusuz/Deneyap_Software_Techn/assets/86704802/cd98b111-b66c-4ddb-b0c4-f62ce0ab8b46" alt="ReLU" width="125"/>
</p>


## 📚 İçindekiler
| Hafta | Haftalık İçerik                             |
|-------|--------------------------------------------|
| 📆 Week 1 | [**Iris Veri Seti ile Sınıflandırma**](#week-1-iris-veri-seti-ile-sınıflandırma) |
| 📆 Week 2 | [**Bulaşık Yıkama Süresi Kontrol Sistemi**](#week-2-bulaşık-yıkama-süresi-kontrol-sistemi) |
| 📆 Week 3 | [**Naive Bayes ile Kalp Ritim Tespiti**](#week-3-naive-bayes-ile-kalp-ritim-tespiti) |
| 📆 Week 4 | [**Kalp Ritim Bozukluğu Tespiti ve Hastalıklı Yaprak Analizi**](#week-4-kalp-ritim-bozukluğu-tespiti-ve-hastalıklı-yaprak-analizi) |
| 📆 Week 5 | [**Yapay Sinir Ağları ile Isıtma ve Soğutma Yükü Tahmini**](#week-5-yapay-sinir-ağları-ile-isıtma-ve-soğutma-yükü-tahmini) |
| 📆 Week 6 | [**Q-Learning ile Kargo Teslimatı**](#week-6-q-learning-ile-kargo-teslimatı) |
| 📆 Week 7 | [**Fonksiyon Optimizasyonu için Pygad ile Genetik Algoritma**](#week-7-fonksiyon-optimizasyonu-için-pygad-ile-genetik-algoritma) |


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
<h3>1. Kullanılan Kütüphaneneler</h3>

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

<h3>2. Dosya ve Dizin İşlemleri</h3>

```python
bakteri_yaprak_yanik = "rice_leaf_diseases/Bacterial leaf blight/"
kahve_nokta = "rice_leaf_diseases/Brown spot/"
yaprak_isi = "rice_leaf_diseases/Leaf smut"

def dosya(yol):
    return [os.path.join(yol, f) for f in os.listdir(yol)]
```

- bakteri_yaprak_yanik, kahve_nokta, yaprak_isi: Farklı hastalık türlerine ait görüntülerin bulunduğu dizinlerin yolları.
- dosya: Belirtilen yoldaki tüm dosyaların tam yolunu döndüren bir fonksiyon.


<h3>3. Veri Dönüştürme </h3>

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

<h3>4. Verilerin Data Setlerinden Yüklenmesi ve Birleştirilmesi </h3>

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


<h3>5. Giriş ve Çıkış Verilerinin Hazırlanması</h3>

```python
Giris = np.array(tum_veri)[:,:784]
Cikis = np.array(tum_veri)[:,784]
```

- Giris: Görüntü verilerini içerir.
- Cikis: Sınıf etiketlerini içerir.


<h3>6. Veri Setinin Eğitim(Train) ve test Kümelerine Ayrılması</h3>

```python
Giris_train, Giris_test, Cikis_train, Cikis_test = train_test_split(Giris, Cikis, test_size=0.2, random_state=109)
```

- Veri seti %80 eğitim ve %20 test olacak şekilde ayrılır.



<h3>7. Modelin Eğitilmesi ve Test Edilmesi</h3>

```python
model = RandomForestClassifier()
model.fit(Giris_train, Cikis_train)
```

- RandomForestClassifier modeli oluşturulur ve eğitim verileriyle eğitilir.


<h3>8. Tahmin Yapılması ve Doğruluk Ölçümü</h3>

```python
Cikis_pred = model.predict(Giris_test)
print("Doğruluk:", metrics.accuracy_score(Cikis_test, Cikis_pred))
```

- Test verileri üzerinde tahmin yapılır ve modelin doğruluğu ölçülür.


<h3>9. Özetle Neyi Hedefledik? </h3>

<p>Kod, pirinç yaprak hastalıklarını sınıflandırmak için bir makine öğrenimi modeli oluşturur ve modelin doğruluğunu ölçer. Bu model, görüntüleri gri tonlamalı yapıp, yeniden boyutlandırarak ve düzleştirerek çalışır. RandomForestClassifier kullanılarak hastalık sınıflandırması yapılır ve test verileri üzerinde doğruluk ölçülür. </p>



## Week 5: Yapay Sinir Ağları ile Isıtma ve Soğutma Yükü Tahmini

<p>Bu proje, Tsanas ve Xifara (2012) tarafından sağlanan 768 örnekten oluşan bir veri setini kullanarak çeşitli giriş parametrelerine göre binaların ısıtma ve soğutma yüklerini yapay sinir ağları (YSA) kullanarak tahmin etmeyi amaçlamaktadır. </p>

<h2>Veri Seti</h2>
Veri seti aşağıdaki giriş parametrelerini içermektedir:

- 🧱Rölatif sıkılık
- Yüzey alanı
- Duvar alanı
- Çatı alanı
- Bina yüksekliği
- Oryantasyon
- Cam alanı
- Cam alan dağılımı

<h3>Hedef çıktılar ise</h3>

- Isıtma yükü
- Soğutma yükü

<h3>🏗️ Proje Yapısı</h3>
Proje aşağıdaki adımları içermektedir:

- Verinin yüklenmesi ve ön işlenmesi
- Verinin eğitim ve test setlerine ayrılması
- Girdi özelliklerinin ölçeklendirilmesi
- YSA modelinin oluşturulması
- Modelin eğitilmesi
- Modelin performansının değerlendirilmesi


<h2>🛠️ Kurulum</h2>
Bu projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

- numpy
- pandas
- scikit-learn
- keras
- tensorflow
- matplotlib

Bu kütüphaneleri pip kullanarak yükleyebilirsiniz:

```python
pip install numpy pandas scikit-learn keras tensorflow matplotlib
```

<h2>Kullanım</h2>
<p>Veri seti bir Excel dosyasından yüklenir ve giriş özellikleri (X) ve hedef çıktılar (y) olarak ayrılır. Veri daha sonra eğitim ve test setlerine bölünür.</p>
<img src="https://github.com/YusufsKaygusuz/Artificial-Intelligient-Lessons/assets/86704802/8f5925cd-8ea8-4fd1-a61e-5e935117899a" alt="ReLU" width="400"/>





<h3>Verinin Ölçeklendirilmesi</h3>
Girdi özellikleri StandardScaler kullanılarak ölçeklendirilir, böylece verinin ortalaması 0 ve standart sapması 1 olur. Bu, sinir ağlarının eğitimi için önemlidir.

<h3>YSA Modelinin Oluşturulması</h3>
Isıtma ve soğutma yüklerini tahmin etmek için ortak bir yol ve iki ayrı yol kullanan bir sinir ağı modeli Keras kullanılarak oluşturulur.

```python
# Girdi katmanını tanımla, veri kümesindeki özellik sayısına göre şekil belirle
input_layer = Input(shape=(data_x_train_scaled.shape[1],), name='Input_Layer')

# İlk yoğun katmanı tanımla, 128 birim ve 'relu' aktivasyon fonksiyonu kullan
common_path = Dense(units=128, activation='relu', name='First_Dense')(input_layer)

# Aşırı öğrenmeyi önlemek için dropout katmanı ekle, dropout oranı %30
common_path = Dropout(0.3)(common_path)

# İkinci yoğun katmanı tanımla, yine 128 birim ve 'relu' aktivasyon fonksiyonu kullan
common_path = Dense(units=128, activation='relu', name='Second_Dense')(common_path)

# Aşırı öğrenmeyi önlemek için ikinci dropout katmanı ekle, dropout oranı %30
common_path = Dropout(0.3)(common_path)

# İlk çıkış katmanını tanımla, bir birim ile (Isıtma yükü tahmini için)
first_output = Dense(units=1, name='First_Output__Last_Layer')(common_path)

# İkinci çıkış yolu için ilk yoğun katmanı tanımla, 64 birim ve 'relu' aktivasyon fonksiyonu kullan
second_output_path = Dense(units=64, activation='relu', name='Second_Output__First_Dense')(common_path)

# Aşırı öğrenmeyi önlemek için üçüncü dropout katmanı ekle, dropout oranı %30
second_output_path = Dropout(0.3)(second_output_path)

# İkinci çıkış katmanını tanımla, bir birim ile (Soğutma yükü tahmini için)
second_output = Dense(units=1, name='Second_Output__Last_Layer')(second_output_path)

# Modeli tanımla, giriş katmanı ve iki çıkış katmanını belirt
model = Model(inputs=input_layer, outputs=[first_output, second_output])

```

<h3>Modelin Eğitilmesi</h3>
Model SGD optimizer ve ortalama kare hata (MSE) kaybı ile derlenir. Aşırı öğrenmeyi önlemek için erken durdurma kullanılır.

```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss={'First_Output__Last_Layer': 'mse', 'Second_Output__Last_Layer': 'mse'},
              metrics={'First_Output__Last_Layer': tf.keras.metrics.RootMeanSquaredError(),
                       'Second_Output__Last_Layer': tf.keras.metrics.RootMeanSquaredError()})

history = model.fit(x=data_x_train_scaled, y=data_y_train, verbose=0, epochs=500, batch_size=10,
                    validation_split=0.3, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
```


<h3>Modelin Değerlendirilmesi</h3>
Modelin performansı test seti üzerinde R-kare metriği kullanılarak değerlendirilir.

```python
y_pred = np.array(model.predict(data_x_test_scaled))
print("İlk çıkışın R2 değeri:", r2_score(data_y_test[:, 0], y_pred[0, :].flatten()))
print("İkinci çıkışın R2 değeri:", r2_score(data_y_test[:, 1], y_pred[1, :].flatten()))
```


<h3>Sonuçların Gösterilmesi</h3>
Eğitim ve doğrulama setleri için RMSE kayıplarının grafiği çizilerek modelin performansı görselleştirilir.

```python
import matplotlib.pyplot as plt
plt.plot(history.history['First_Output__Last_Layer_root_mean_squared_error'])
plt.plot(history.history['val_First_Output__Last_Layer_root_mean_squared_error'])
plt.title("İlk Çıkış için RMSE Değerleri")
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper right')
plt.show()

plt.plot(history.history['Second_Output__Last_Layer_root_mean_squared_error'])
plt.plot(history.history['val_Second_Output__Last_Layer_root_mean_squared_error'])
plt.title("İkinci Çıkış için RMSE Değerleri")
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper right')
plt.show()
```

<h2>📝 Sonuç</h2>
Bu proje, çeşitli parametrelere dayalı olarak binaların ısıtma ve soğutma yüklerini tahmin etmek için yapay sinir ağlarının nasıl kullanılacağını göstermektedir. Sonuçlar, böyle bir regresyon görevi için çok çıkışlı bir model kullanmanın etkinliğini göstermektedir.

<h4>İlk çıkışın R2 değeri : 0.937754312982972</h4>
<h4>İkinci çıkışın R2 değeri: 0.878525945856873</h4>






## Week 6: Q-Learning ile Kargo Teslimatı

<p>Bu proje, bir 11x11 ızgara ortamında Q-Learning algoritması kullanarak bir robotun kargo teslimatı yapmasını simüle etmektedir. Robot, belirli geçit noktalarından geçerek bir ödül noktasına ulaşmayı amaçlamaktadır. </p> 

![Ekran Resmi 2024-05-30 20 13 02](https://github.com/YusufsKaygusuz/Artificial-Intelligient-Lessons/assets/86704802/51f309f2-8d59-45fe-bdea-1c7f873d8f9a)


<h2>Proje Hakkında</h2>

<p>Bu proje, Q-learning algoritmasını kullanarak bir robotun kargo teslimatı yapmasını simüle eder. 11x11 bir ızgara ortamında robot, geçit noktalarından geçerek belirli bir ödül noktasına ulaşmaya çalışır. Ödül matrisi başlangıçta -100 ile başlatılır ve ödül noktası 0,5 koordinatında 100 ödül değeri taşır. Geçit noktalarında ödül -1'dir. Python dosyasını çalıştırarak Q-learning algoritmasının eğitimini tamamlayabilir ve ardından robotun kargo noktasına ulaşacağı rotayı belirleyebilirsiniz.
</p>

<h2>Fonksiyonlar</h2>

<p>Bu bölümde, projede kullanılan ana fonksiyonlar ve görevleri açıklanmaktadır. Ortamın boyutlarını belirler ve Q değerlerini sıfırla başlatır. Hareketleri tanımlar ve ödül matrisini -100 ile başlatır. Ödül noktası 0,5 koordinatında 100 ödül değeri taşır.</p>

```python
# Ortam boyutlarını belirle
ortam_satir_sayisi = 11
ortam_sutun_sayisi = 11

# Q değerlerini sıfırla başlat
q_degerleri = np.zeros((ortam_satir_sayisi, ortam_sutun_sayisi, 4))

# Hareketleri tanımla
hareketler = ['yukari', 'sag', 'asagi', 'sol']

# Ödül matrisini -100 ile başlat ve ödül noktasını tanımla
oduller = np.full((ortam_satir_sayisi, ortam_sutun_sayisi), -100.)
oduller[0,5] = 100.
```

<h3>🚇 Geçit Noktaları</h3>
<p>Geçit noktalarını ve ödüllerini tanımlar ve ödül matrisine ekler.</p>

```python
# Geçit Noktalarını Tanımla
gecitler = {}
gecitler[1] = [i for i in range (1,10)]
gecitler[2] = [1, 7, 9]
gecitler[3] = [i for i in range(1,8)]
gecitler[3].append(9)
gecitler[4] = [3, 7]
gecitler[5] = [i for i in range(11)]
gecitler[6] = [5]
gecitler[7] = [i for i in range(1, 10)]
gecitler[8] = [3, 7]
gecitler[9] = [i for i in range(11)]

# Geçit noktalarını ödül matrisine ekle
for satir_indeks in range(1,10):
    for sutun_indeks in gecitler[satir_indeks]:
        oduller[satir_indeks, sutun_indeks] = -1.
```

<h3>🚧 Engel Kontrol Fonksiyonu</h3>

<p>Verilen bir konumda engel olup olmadığını kontrol eder.</p>

```python
def engel_mi(gecerli_satir_indeks, gecerli_sutun_indeks):
    if oduller[gecerli_satir_indeks, gecerli_sutun_indeks] == -1.:
        return False
    else:
        return True
```

<h3>✅ Rastgele Başlangıç Noktası Belirleme</h3>

<p>Robotun başlayacağı rastgele bir başlangıç noktasını belirler.</p>

```python
def baslangic_belirle():
    gecerli_satir_indeks = np.random.randint(ortam_satir_sayisi)
    gecerli_sutun_indeks = np.random.randint(ortam_sutun_sayisi)
    while engel_mi(gecerli_satir_indeks, gecerli_sutun_indeks):
        gecerli_satir_indeks = np.random.randint(ortam_satir_sayisi)
        gecerli_sutun_indeks = np.random.randint(ortam_sutun_sayisi)
    return gecerli_satir_indeks, gecerli_sutun_indeks
```



<h3>🦾 Sonraki Hareketi Belirleme</h3>

<p>Robotun bir sonraki hareketini epsilon-greedy stratejisi kullanarak belirler.</p>

```python
def sonraki_hareket_belirle(gecerli_satir_indeks, gecerli_sutun_indeks, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_degerleri[gecerli_satir_indeks, gecerli_sutun_indeks])
    else:
        return np.random.randint(4)
```



<h3>📍 Sonraki Noktaya Git</h3>

<p>Robotun bir sonraki adımda hangi noktaya gideceğini belirler.</p>

```python
def sonraki_noktaya_git(gecerli_satir_indeks, gecerli_sutun_indeks, hareket_indeks):
    yeni_satir_indeks = gecerli_satir_indeks
    yeni_sutun_indeks = gecerli_sutun_indeks

    if hareketler[hareket_indeks] == 'yukari' and gecerli_satir_indeks > 0:
        yeni_satir_indeks -= 1
    elif hareketler[hareket_indeks] == 'sag' and gecerli_sutun_indeks < ortam_sutun_sayisi - 1:
        yeni_sutun_indeks += 1
    elif hareketler[hareket_indeks] == 'asagi' and gecerli_satir_indeks < ortam_satir_sayisi - 1:
        yeni_satir_indeks += 1
    elif hareketler[hareket_indeks] == 'sol' and gecerli_sutun_indeks > 0:
        yeni_sutun_indeks -= 1
    return yeni_satir_indeks, yeni_sutun_indeks
```



<h3>🏁 En Kısa Mesafeyi Belirleme</h3>

<p>Verilen başlangıç noktasından ödül noktasına giden en kısa mesafeyi belirler.</p>

```python
def en_kisa_mesafe(basla_satir_indeks, basla_sutun_indeks):
    if engel_mi(basla_satir_indeks, basla_sutun_indeks):
        return []
    else:
        gecerli_satir_indeks, gecerli_sutun_indeks = basla_satir_indeks, basla_sutun_indeks
        en_kisa = []
        en_kisa.append([gecerli_satir_indeks, gecerli_sutun_indeks])
        while not engel_mi(gecerli_satir_indeks, gecerli_sutun_indeks):
            hareket_indeks = sonraki_hareket_belirle(gecerli_satir_indeks, gecerli_sutun_indeks, 1.)
            gecerli_satir_indeks, gecerli_sutun_indeks = sonraki_noktaya_git(gecerli_satir_indeks, 
                                                                       gecerli_sutun_indeks, hareket_indeks)
            en_kisa.append([gecerli_satir_indeks, gecerli_sutun_indeks])
        return en_kisa
```



<h3>🥷🏻 Q-Learning Algoritması</h3>

<p>Q-learning algoritmasının ana döngüsü. Algoritma, verilen adım sayısı boyunca ortamda hareket ederek Q değerlerini günceller.</p>

```python
# Q-learning parametreleri
epsilon = 0.9
azalma_degeri = 0.9
ogrenme_orani = 0.9

# Q-learning algoritmasını çalıştır
for adim in range(1000):
  satir_indeks, sutun_indeks = baslangic_belirle()
  while not engel_mi(satir_indeks, sutun_indeks):
    hareket_indeks = sonraki_hareket_belirle(satir_indeks, sutun_indeks, epsilon)
    eski_satir_indeks, eski_sutun_indeks = satir_indeks, sutun_indeks
    satir_indeks, sutun_indeks = sonraki_noktaya_git(satir_indeks, sutun_indeks, hareket_indeks)
    odul = oduller[satir_indeks, sutun_indeks]
    eski_q_degeri = q_degerleri[eski_satir_indeks, eski_sutun_indeks, hareket_indeks]
    fark = odul + (azalma_degeri * np.max(q_degerleri[satir_indeks, sutun_indeks])) - eski_q_degeri
    yeni_q_degeri = eski_q_degeri + (ogrenme_orani * fark)
    q_degerleri[eski_satir_indeks, eski_sutun_indeks, hareket_indeks] = yeni_q_degeri
print('Eğitim tamamlandı.')
```


<h3>🤖 Robotun Hareketi</h3>

```python
ogr_sonrasi_satir = input('Robotun harekete başlayacağı satır indeksini giriniz: ')
ogr_sonrasi_sutun = input('Robotun harekete başlayacağı sütun indeksini giriniz: ')

# En kısa mesafeyi hesapla ve ekrana yazdır
print('Kargo noktasına giden rota: ', en_kisa_mesafe(int(ogr_sonrasi_satir), int(ogr_sonrasi_sutun)))

```

<h3>🛫 En kısa mesafeyi hesapla ve ekrana yazdır</h3>

```python
print('Kargo noktasına giden rota: ', en_kisa_mesafe(int(ogr_sonrasi_satir), int(ogr_sonrasi_sutun)))
```




## Week 7: Fonksiyon Optimizasyonu için Pygad ile Genetik Algoritma

<p>Bu depo, bir fonksiyonu optimize etmek için pygad kütüphanesini kullanarak genetik algoritmayı uygulayan bir Python betiği içermektedir. Amaç, belirli girdilere uygulandığında istenen çıktıyı üreten en iyi değişkenler kümesini bulmaktır.</p>

<img src="https://github.com/YusufsKaygusuz/Artificial-Intelligient-Lessons/assets/86704802/692e257e-5c75-4b30-9f31-11fc5a4eb932" alt="ReLU" width="450"/> 

<p>Bu proje, bir fonksiyon için en uygun çözümü bulmak amacıyla genetik algoritmanın nasıl kullanılacağını göstermektedir. Genetik algoritma, bir fitness fonksiyonuna göre en iyi bireyleri seçerek çoklu nesiller boyunca çözümler popülasyonunu geliştirir. Fitness fonksiyonu, çözümün istenen çıktıya ne kadar yakın olduğunu ölçer.</p>


<h2>Kod Açıklaması</h2>

<h3>1. Fonksiyon Girişleri ve İstenen Çıktı</h3>
<p>Fonksiyon girişleri, optimize edilmek istenen fonksiyonun değişkenleridir. İstenen çıktı, bu fonksiyonun ulaşması gereken hedef değeri belirtir.</p>

```python
function_inputs = [4, -2, 3.5, 5, -11, -4.7]
desired_output = 44
```

<h3>2. Fitness Fonksiyonu</h3>

```python
def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness
```

<p>Fitness fonksiyonu, her çözümün ne kadar iyi olduğunu belirler. Bu fonksiyon, çözümler ile fonksiyon girişlerinin çarpımının toplamını hesaplar ve bu toplamın istenen çıktıya ne kadar yakın olduğunu ölçer.</p>

<h3>3. Genetik Algoritma Parametreleri</h3>

<p>Genetik algoritmanın çalışma parametrelerini belirler. Toplam nesil sayısı, eşleşecek ebeveyn sayısı, popülasyon başına çözüm sayısı ve her çözümdeki gen sayısı bu parametreler arasındadır.</p>

```python
num_generations = 100
num_parents_mating = 7
sol_per_pop = 50
num_genes = len(function_inputs)
```

<h3>4. Nesil Özeti Fonksiyonu</h3>

<p>Genetik algoritma örneğini oluşturur ve belirlenen parametreler ile çalıştırır.</p>

```python
def nesil_ozeti(ga_instance):
    global last_fitness
    print(f"Nesil = {ga_instance.generations_completed}")
    print(f"Fonksiyon Sonucu = {ga_instance.best_solution()[1]}")
    print(f"Degisim = {ga_instance.best_solution()[1] - last_fitness}")
    last_fitness = ga_instance.best_solution()[1]
```

<h3>5. En İyi Çözümün Alınması ve Gösterilmesi</h3>

<p>Genetik algoritma tarafından bulunan en iyi çözümü, bu çözümün fitness değerini ve bu çözümün bulunduğu nesli alır ve gösterir.</p>

```python
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"En uygun cozum degisken degerleri : {solution}")
print(f"En uygun cozumu veren birey indeks no.: {solution_idx}")
prediction = numpy.sum(numpy.array(function_inputs) * solution)
print(f"En uygun cozum ile fonksiyon sonucu : {prediction}")
if ga_instance.best_solution_generation != -1:
    print(f"En uygun cozum {ga_instance.best_solution_generation} nesil sonra elde edildi.")
```


<h3>6. Sonuçlar</h3>
Kod, genetik algoritma tarafından bulunan en iyi çözümü, bu çözümün fitness değerini ve en iyi çözümün bulunduğu nesli çıktı olarak verir. Süreç, çözümlerin evrimini zamanla gözlemleyebilmek için her nesilden sonra günlük bilgileri içerir.


