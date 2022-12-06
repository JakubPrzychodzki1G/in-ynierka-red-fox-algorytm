import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class RED_FOXES:
    def __init__(self, maxx, minx, wymiar, fitness, alfa):
        self.wymiar = wymiar
        self.position = np.empty(wymiar)
        self.mi = random.uniform(0, 1)
        self.fitness = fitness
        self.alfa = alfa
        for j in range(int(wymiar)):
            if j == 0:
                self.position[j] = random.uniform(minx[0], maxx[0])
            else:
                self.position[j] = random.uniform(minx[j], maxx[j])

    def __repr__(self):
        return '{' +str(self.fitness) + ', ' + str(self.position) + ', ' + str(self.mi) + ', '+ str(self.alfa) +'}'



def fitness_function(x, nazwa_funckji_celu): #  x to lista, ale nie musibyc w zaleznosci od funkcji lista napewno to bedzie w przypadku powella i brown
    lista_pomocnicza1 = np.empty(lisy[1].wymiar)
    lista_pomocnicza2 = np.empty(lisy[1].wymiar)
    zmienna_pomocnicza1 = 0
    zmienna_pomocnicza2 = 0
    result = 0
    a = 0
    b = 0
    if nazwa_funckji_celu == "brown":
        for j in range(lisy[1].wymiar-1):
            zmienna_pomocnicza1 = math.pow(x[0], 2)
            zmienna_pomocnicza2 = math.pow(x[1], 2) + 1
            lista_pomocnicza1[j] = math.pow(zmienna_pomocnicza1, zmienna_pomocnicza2)
        a = math.pow(x[0], 2) + 1
        b = math.pow(x[1], 2)
        result = (np.sum(lista_pomocnicza1)) + (math.pow((b), (a)))
    elif nazwa_funckji_celu == "csendes":
        for j in range(lisy[1].wymiar):
            lista_pomocnicza1[j] = math.pow(x[j], 6)*(2+np.sin(1/x[j]))
        result = np.sum(lista_pomocnicza1)
    elif nazwa_funckji_celu == "griewank":
        for j in range(lisy[1].wymiar):
            lista_pomocnicza1[j] = ((math.pow(x[j], 2)) / 4000)
            lista_pomocnicza2[j] = np.cos((x[j] / np.sqrt(j+1)))
        lista_pomocnicza1[0] = np.sum(lista_pomocnicza1)
        lista_pomocnicza2[0] = np.prod(lista_pomocnicza2)
        result = 1+lista_pomocnicza1[0] - lista_pomocnicza2[0]
    elif nazwa_funckji_celu == "powell":
        for j in range(lisy[1].wymiar/4):
            lista_pomocnicza1[j] = ((math.pow((x[(4*j) -3]+10*(x[(4*j)-2])), 2))+(5*math.pow((x[(4*j)-1]-x[4*j]),2))+(math.pow((x[(4*j)-2]-2*x[(4*j)-1]),4))+(10*math.pow((x[(4*j)-3]-x[4*j]),2)))
        result = np.sum(lista_pomocnicza1)
    elif nazwa_funckji_celu == "powell sum":
        for j in range(lisy[1].wymiar):
            lista_pomocnicza1[j] = math.pow(np.abs(x[j]),j+1)
        result = np.sum(lista_pomocnicza1)
    elif nazwa_funckji_celu == "rastragin":
        for j in range(lisy[1].wymiar):
            lista_pomocnicza1[j] = (math.pow(x[j], 2))-(10*np.cos(2*math.pi*x[j]))
        result = (10*lisy[1].wymiar) + np.sum(lista_pomocnicza1)
    elif nazwa_funckji_celu == "rotated hyper-ellipsoid":
        for l in range(lisy[1].wymiar):
            for j in range(lisy[1].wymiar):
                lista_pomocnicza1[j] = math.pow(x[j], 2)
            lista_pomocnicza2[l] = np.sum(lista_pomocnicza1)
        result = np.sum(lista_pomocnicza2)
    elif nazwa_funckji_celu == "mishra":
        for j in range(lisy[1].wymiar):
            lista_pomocnicza1[j] = np.abs(x[j])
        result = math.pow(((1/lisy[1].wymiar) * (np.sum(lista_pomocnicza1)))-(math.pow(np.abs(x[j]), 1/lisy[1].wymiar)), 2)
    elif nazwa_funckji_celu == "salomon":
        for j in range(lisy[1].wymiar):
            lista_pomocnicza1[j] = math.pow(x[j], 2)
            lista_pomocnicza2[j] = math.pow(x[j], 2)
        result = 1 - (np.cos(np.sum(lista_pomocnicza1))) + (0.1*np.sqrt(np.sum(lista_pomocnicza2)))
    else:
        print("zla nazwa funkcji")
        result = 0
    # t1 = np.sum(x ** 2) / 4000
    # t2 = np.prod([np.cos(x[idx] / np.sqrt(idx + 1)) for idx in range(0, 2)])
    return result


def obser_angle(observation_radius, fi, weather):
    if fi != 0:
        return observation_radius*(np.sin(fi)/fi)
    if fi == 0:
        return weather


def odleglosc_euklidesowa(v, u):
   return np.linalg.norm(v - u)

weather = random.uniform(0,1)
wymiar = 2
minima = [-10, -10]
maxima = [10, 10]
ilosc_liskow = 50
#generowanie liskow w roznych zakresach
lisy = []
lisy1 = []
y = 0
liczba_powtorzen_calego_algorytmu = 20
while y < ilosc_liskow:
    lisy.append(RED_FOXES(maxima, minima, wymiar, 0, 0))
    y = y+1


print(lisy)
#liczba iteracji
T = 80

fitness_prawdziwe = np.empty(ilosc_liskow)

#przelicza przez funkcje celu
for i in range(int(ilosc_liskow)):
    lisy[i].fitness = fitness_function(lisy[i].position, "griewank")

#posortowana przeliczona funkcja celu

#for l in range(int(ilosc_liskow)):
    #print(str(l) + ". " + str(lisy[l].position) + " / " + str(lisy[l].fitness))

#print(len(lisy))
lisy1 = copy.deepcopy(lisy)
#for l in range(int(ilosc_liskow)):
    #print(str(l) + ". " + str(lisy1[l].position) + " / " + str(lisy1[l].fitness))
#print(len(lisy1))
lista_obliczenia = np.empty(liczba_powtorzen_calego_algorytmu)
lista_wykres = np.empty((T*2, ilosc_liskow))
#glowna petla while

for z in range(liczba_powtorzen_calego_algorytmu):
    wykr = 0
    t=0
    lisy.clear()
    lisy=copy.deepcopy(lisy1)
    while t < T:
        zmienna = 0
        kappa = random.uniform(0, 1)
        fi_obser = np.empty(len(lisy)+1)
        fi_obser[0] = random.uniform(0, 2 * np.pi)
        lisy.sort(key=lambda x: np.abs(x.fitness))
        x_best = lisy[0].position
        a = random.uniform(0, 0.2)

        for o in range(len(lisy)):
            lisy[o].alfa = random.uniform(0, odleglosc_euklidesowa(lisy[o].position, x_best))
            fi_obser[o+1] = random.uniform(0, 2 * np.pi)
            x_z_kreska = lisy[o].position
            x_z_kreska = x_z_kreska + lisy[o].alfa*np.sign(x_best - x_z_kreska)
            if np.abs(lisy[o].fitness) > np.abs(fitness_function(x_z_kreska, "griewank")):
                lisy[o].position = x_z_kreska
                lisy[o].fitness = fitness_function(x_z_kreska, "griewank")

            noticing = lisy[o].mi
            if noticing > 0.75:
                r = obser_angle(a, fi_obser[0], weather)
                ar_cos = 0
                for h in range(0, wymiar):
                    ar_cos = ar_cos + a * r * np.cos(fi_obser[o+1])
                    lisy[o].position[h] = ar_cos + lisy[o].position[h]
                lisy[o].fitness = fitness_function(lisy[o].position, "griewank")

        lisy.sort(key=lambda x: np.abs(x.fitness))
        old_lisy_len = len(lisy)
        for g in range(0, math.ceil(0.05 * len(lisy))):
            lisy.pop(len(lisy)-1 - g)
        new_lisy_len = len(lisy)
        habitat_center = (lisy[0].position + lisy[1].position)/2
        habitat_diameter = np.sqrt(np.linalg.norm(lisy[0].position-lisy[1].position))
        for c in range(0,old_lisy_len - new_lisy_len):
            if kappa >= 0.45:
                nowe_minx = [habitat_center[0]+habitat_diameter, habitat_center[0]+habitat_diameter]
                lisy.append(RED_FOXES(maxima, nowe_minx, wymiar, 0, 0))
                lisy[-1].fitness = fitness_function(lisy[-1].position, "griewank")
            else:
                new_from_alfa_foxes_position = kappa*habitat_center
                max_min_new_from_alfa_foxes_position = [new_from_alfa_foxes_position[0], new_from_alfa_foxes_position[1]]
                lisy.append(RED_FOXES(max_min_new_from_alfa_foxes_position, max_min_new_from_alfa_foxes_position, wymiar, 0 ,0))
                lisy[-1].fitness = fitness_function(lisy[-1].position, "griewank")
        t = t + 1
        for h in range(len(lisy)):
            lista_wykres[wykr][h] = lisy[h].position[0]
            lista_wykres[wykr+1][h] = lisy[h].position[1]
        #plt.figure(t)
        wykr += 2
        print("generacja: "+str(t)+", najlepsze: "+str(lisy[0]))
        #plt.plot(lista_wykres[0], lista_wykres[1], 'ro')
    lisy.sort(key=lambda x: np.abs(x.fitness))
    print("generacja:"+str(t)+"najlepsze: "+str(lisy[0]))
    lista_obliczenia[z] = lisy[0].fitness
print("odchylenie standardowe: "+str(np.std(lista_obliczenia[0:45]))+", srednia: "+str(np.average(lista_obliczenia[0:45])))

fig, ax = plt.subplots()
print(lista_wykres[2])


def animate(i):
    x_axes = lista_wykres[i]
    y_axes = lista_wykres[i+1]

    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.autoscale(enable=False, axis='both', tight=None)
    i+=2
    return ax.plot(x_axes, y_axes, 'o')


ani = animation.FuncAnimation(fig, animate, frames=T, interval=100, repeat=False)

plt.show()

#ani.save('location.gif', writer=animation.ImageMagickFileWriter()) odkomentować jeśli ma zapisać animacje


