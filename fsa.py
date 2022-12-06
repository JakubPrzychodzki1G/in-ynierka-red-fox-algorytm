import numpy as np
import math
from mealpy.swarm_based.WOA import OriginalWOA
import matplotlib.pyplot as plt


# func.evaluate(func.create_solution())
# print(func.evaluate(func.create_solution()))


def fitness_function(x):
    lista_pomocnicza1 = np.empty(2)
    lista_pomocnicza2 = np.empty(2)
    for j in range(2):
        lista_pomocnicza1[j] = ((math.pow(x[j], 2)) / 4000)
        lista_pomocnicza2[j] = np.cos((x[j] / np.sqrt(j + 1)))
    lista_pomocnicza1[0] = np.sum(lista_pomocnicza1)
    lista_pomocnicza2[0] = np.prod(lista_pomocnicza2)
    result = 1 + lista_pomocnicza1[0] - lista_pomocnicza2[0]
    return result

problem_dict1 = {
    "fit_func": fitness_function,
    "lb": [-10, -10],
    "ub": [10, 10],
    "minmax": "min",
    "save_population": True,
}

epoch = 80
pop_size = 50
best_iteration = np.empty(20)
for i in range(0,20):
    model = OriginalWOA(epoch, pop_size)
    best_position, best_fitness = model.solve(problem_dict1)
    all_best = model.history.list_global_best
    new_arry = np.empty(45)
    x = np.arange(0, 45, 1)
    best_iteration[i]=best_fitness
    for a in range(0,45):
        new_arry[a]=all_best[a][1][0]
    plt.figure(i)
    plt.plot(x, new_arry)
    print('best fitness: '+str(best_fitness))

print("odchylenie standardowe: "+str(np.std(best_iteration))+", srednia: "+str(np.average(best_iteration)))
plt.show()

