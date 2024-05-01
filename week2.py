import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Giriş ve çıkış değişkenlerinin tanımlanması
bulaşık_miktarı = ctrl.Antecedent(np.arange(0, 100, 1), 'bulaşık miktarı')
kirlilik = ctrl.Antecedent(np.arange(0, 100, 1), 'kirlilik seviyesi')
yıkama_süresi = ctrl.Consequent(np.arange(0, 180, 1), 'yıkama süresi')

# Üyelik fonksiyonlarının tanımlanması
bulaşık_miktarı['az'] = fuzz.trimf(bulaşık_miktarı.universe, [0, 0, 30])
bulaşık_miktarı['normal'] = fuzz.trimf(bulaşık_miktarı.universe, [10, 30, 60])
bulaşık_miktarı['çok'] = fuzz.trimf(bulaşık_miktarı.universe, [50, 60, 100])

kirlilik['az'] = fuzz.trimf(kirlilik.universe, [0, 0, 30])
kirlilik['normal'] = fuzz.trimf(kirlilik.universe, [10, 30, 60])
kirlilik['çok'] = fuzz.trimf(kirlilik.universe, [50, 60, 100])

yıkama_süresi['kısa'] = fuzz.trimf(yıkama_süresi.universe, [0, 0, 50])
yıkama_süresi['normal'] = fuzz.trimf(yıkama_süresi.universe, [40, 50, 100])
yıkama_süresi['uzun'] = fuzz.trimf(yıkama_süresi.universe, [60, 80, 180])

# Kuralların tanımlanması
kural1 = ctrl.Rule(bulaşık_miktarı['az'] & kirlilik['az'], yıkama_süresi['kısa'])
kural2 = ctrl.Rule(bulaşık_miktarı['normal'] & kirlilik['az'], yıkama_süresi['normal'])
kural3 = ctrl.Rule(bulaşık_miktarı['çok'] & kirlilik['az'], yıkama_süresi['normal'])
kural4 = ctrl.Rule(bulaşık_miktarı['az'] & kirlilik['normal'], yıkama_süresi['normal'])
kural5 = ctrl.Rule(bulaşık_miktarı['normal'] & kirlilik['normal'], yıkama_süresi['uzun'])
kural6 = ctrl.Rule(bulaşık_miktarı['çok'] & kirlilik['normal'], yıkama_süresi['uzun'])
kural7 = ctrl.Rule(bulaşık_miktarı['az'] & kirlilik['çok'], yıkama_süresi['normal'])
kural8 = ctrl.Rule(bulaşık_miktarı['normal'] & kirlilik['çok'], yıkama_süresi['uzun'])
kural9 = ctrl.Rule(bulaşık_miktarı['çok'] & kirlilik['çok'], yıkama_süresi['uzun'])

# Kontrol sistemi tanımlaması
kontrol_sistemi = ctrl.ControlSystem([kural1, kural2, kural3, kural4, kural5, kural6, kural7, kural8, kural9])
model = ctrl.ControlSystemSimulation(kontrol_sistemi)

# Girdi değerlerinin ataması ve çıktının hesaplanması
model.input['bulaşık miktarı'] = 50
model.input['kirlilik seviyesi'] = 80
model.compute()

# Çıktının yazdırılması
print(model.output['yıkama süresi'])
