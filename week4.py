import numpy as np
import pandas as pd
import os
import PIL.Image as img
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

bakteri_yaprak_yanik = "rice_leaf_diseases/Bacterial leaf blight/"
kahve_nokta = "rice_leaf_diseases/Brown spot/"
yaprak_isi = "rice_leaf_diseases/Leaf smut"

def dosya(yol):
    return [os.path.join(yol, f) for f in os.listdir(yol)]

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

yanik_veri = veri_donusturme(bakteri_yaprak_yanik, "bakteri_yaprak_yanik")
yanik_veri_df = pd.DataFrame(yanik_veri)

kahve_nokta_veri = veri_donusturme(kahve_nokta, "kahve_nokta")
kahve_nokta_veri_df = pd.DataFrame(kahve_nokta_veri)

yaprak_isi_veri = veri_donusturme(yaprak_isi, "yaprak_isi")
yaprak_isi_veri_df = pd.DataFrame(yaprak_isi_veri)

tum_veri = pd.concat([yanik_veri_df, kahve_nokta_veri_df, yaprak_isi_veri_df])

Giris = np.array(tum_veri)[:,:784]
Cikis = np.array(tum_veri)[:,784]

# Veri setinin eğitim ve test kümelerine ayrılması
Giris_train, Giris_test, Cikis_train, Cikis_test = train_test_split(Giris, Cikis, test_size=0.2, random_state=109)

# Random Forest modelinin oluşturulması ve eğitilmesi
model = RandomForestClassifier()
model.fit(Giris_train, Cikis_train)

# Test verileri üzerinde tahmin yapılması
Cikis_pred = model.predict(Giris_test)

# Doğruluk ölçümü
print("Doğruluk:", metrics.accuracy_score(Cikis_test, Cikis_pred))
