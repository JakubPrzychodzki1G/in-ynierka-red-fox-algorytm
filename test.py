import numpy
import numpy as np
import random
import math


def fitness_function(x): #  x to lista, ale nie musibyc w zaleznosci od funkcji lista napewno to bedzie w przypadku powella i brown
    lista_pomocnicza1 = np.empty(2)
    lista_pomocnicza2 = np.empty(2)
    zmienna_pomocnicza1 = 0
    zmienna_pomocnicza2 = 0
    result = 0
    a = 0
    b = 0
    for j in range(2):
        lista_pomocnicza1[j] = ((math.pow(x[j], 2)) / 4000)
        lista_pomocnicza2[j] = math.cos((x[j] / np.sqrt(j+1)))
    lista_pomocnicza1[0] = np.sum(lista_pomocnicza1)
    lista_pomocnicza2[0] = np.prod(lista_pomocnicza2)
    result = lista_pomocnicza1[0] - lista_pomocnicza2[0]
    print(result)
    return result


print(fitness_function([0.000017366565, 0.000012598809]))