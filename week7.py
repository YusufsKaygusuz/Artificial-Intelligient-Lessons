import pygad  # pygad kütüphanesini ekle
import numpy  # numpy kütüphanesini ekle

# Fonksiyon girişleri ve istenen çıktıyı tanımlar
function_inputs = [4, -2, 3.5, 5, -11, -4.7] 
desired_output = 44 

# Fitness fonksiyonunu tanımlar. Bu fonksiyon, genetik algoritmanın çözümlerini değerlendirir
def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * function_inputs) # Çözüm ve girişlerin çarpımının toplamını hesaplar
    fitness = 1.0 / numpy.abs(output - desired_output) # Fitness değerini hesaplar, istenen çıktıya ne kadar yakın olduğunu ölçer
    return fitness

fitness_function = fitness_func # Fitness değerini hesaplar, istenen çıktıya ne kadar yakın olduğunu ölçer

# Genetik algoritma parametrelerini tanımlar
num_generations = 100  # Toplam nesil sayısı
num_parents_mating = 7  # Eşleşecek ebeveyn sayısı

sol_per_pop = 50 # Popülasyon başına çözüm sayısı
num_genes = len(function_inputs) # Her çözümdeki gen (değişken) sayısı

last_fitness = 0 # Son fitness değerini saklamak için bir değişken

# Her nesilden sonra özet bilgi veren fonksiyonu tanımlar
def nesil_ozeti(ga_instance):
    global last_fitness # global değişkeni kullanır
    print("Nesil = {generation}".format(generation=ga_instance.generations_completed)) # Şu anki nesli yazdırır
    print("Fonksiyon Sonucu = {fitness}".format(fitness=ga_instance.best_solution()[1])) # En iyi çözümün fitness değerini yazdırır
    print("Degisim = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))# Fitness değerindeki değişimi yazdırır,

    last_fitness = ga_instance.best_solution()[1] # Son fitness değerini günceller

# Genetik algoritma örneği oluşturur ve parametreleri atar
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       on_generation=nesil_ozeti)

# Genetik algoritmayı çalıştırır
ga_instance.run()

# En iyi çözümü ve fitness değerini alır
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("En uygun cozum degisken degerleri : {solution}".format(solution=solution)) # En iyi çözümdeki değişken değerlerini yazdırır
print("En uygun cozumu veren birey indeks no.: {solution_idx}".format(solution_idx=solution_idx)) # En iyi çözümü veren bireyin indeksini yazdırır

prediction = numpy.sum(numpy.array(function_inputs) * solution)
print("En uygun cozum ile fonksiyon sonucu : {prediction}".format(prediction=prediction))

# En iyi çözümün hangi nesilde elde edildiğini kontrol eder ve yazdırır
if ga_instance.best_solution_generation != -1:
    print("En uygun cozum {best_solution_generation} nesil sonra elde edildi."
          .format(best_solution_generation=ga_instance.best_solution_generation))
