import math
import numpy as np
import matplotlib.pyplot as plt

alpha1 = 20  # вт/(м^2*град)
alpha2 = 50  # вт/(м^2*град)
t1 = 25
t2 = 1000

h = 0.01 #м
A = 35 #вт/(м*град)
B = 10 #вт/(м^2*град)
C = -30 #вт/м^3
D = 100 #м
E = 1

n = 500 #размер вычислительной сетки

deltaX = h / n #шаг вычислений
derK = B #производная от k

#Функция для вычисления переменного коэффициента теплопроводности
def k(x):
    return A + B * x

#Функция для вычисления f(x)
def f(x):
    return C * math.exp(-D * ((x - h / E) ** 2))

#Функция для вычисления диагональных коэффициентов и коэффициентов теплопроводности
def calcCoeffs():
    a, c, b, kx = [], [], [], []
    kx.append(k(0))
    a.append(0)
    c.append(-alpha1 - kx[0] / deltaX)
    b.append(kx[0] / deltaX)
    for i in range(1, n - 1):
        kx.append(k(i * deltaX))
        a.append(kx[i] / (deltaX ** 2) - derK / (2 * deltaX))
        c.append(-2 * kx[i] / (deltaX ** 2))
        b.append(kx[i] / (deltaX ** 2) + derK / (2 * deltaX))
    kx.append(k((n - 1) * deltaX))
    a.append(kx[n - 1] / deltaX)
    c.append(-alpha2 - kx[n - 1] / deltaX)
    b.append(0)
    return a, c, b, kx

#Функция для вычисления массива свободных коэффициентов (функций f(x) во всех точках)
def calcFreeCoeffs():
    fc = []
    fc.append(-alpha1 * t1)
    for i in range(1, n - 1):
        fc.append(f(i * deltaX))
    fc.append(-alpha2 * t2)
    return fc

#Метод прогонки
def numericMethod(a, b, c, fc):
    P, Q, X = [], [], [0] * n
    for i in range(n):
        if i == 0:
            P.append(-b[i] / c[i])
            Q.append(fc[i] / c[i])
        else:
            P.append(-b[i] / (a[i] * P[i - 1] + c[i]))
            Q.append((fc[i] - a[i] * Q[i - 1]) / (a[i] * P[i - 1] + c[i]))

    for i in P:
        if abs(i) > 1:
            return np.asarray([0])

    for i in range(n - 1, -1, -1):
        if i == n - 1:
            X[i] = Q[i]
        else:
            X[i] = P[i] * X[i + 1] + Q[i]
    return np.asarray(X)

#Функция вычисления теплового потока
def calcFlow(t, kx):
    flow = []
    for i in range(0, n):
        if i == 0:
            flow.append(-kx[i] * (t[i + 1] - t[i]) / deltaX)
        elif i == n - 1:
            flow.append(-kx[i] * (t[i] - t[i - 1]) / deltaX)
        else:
            flow.append(-kx[i] * (t[i + 1] - t[i - 1]) / (2 * deltaX))
    return flow

#Получение коэффициентов для построения матрицы
a, c, b, kx = calcCoeffs()
fc = calcFreeCoeffs()

#Заполнение матрицы
mainMatrix = np.zeros((n, n))
mainMatrix[0, 0: 2] = [c[0], b[0]]
for i in range(1, n - 1):
    mainMatrix[i, i - 1] = a[i]
    mainMatrix[i, i] = c[i]
    mainMatrix[i, i + 1] = b[i]
mainMatrix[n - 1, n - 2:] = [a[n - 1], c[n - 1]]
print(mainMatrix)

diagonalDominance = True

for i in range(n):
    if abs(c[i]) >= abs(c[i]) + abs(b[i]):
        diagonalDominance = False

#Шаг для построения графиков
stepX = []
for i in range(n):
    stepX.append(deltaX * i)

#Построение графика k(x)
plt.plot(stepX, kx)
plt.title("Переменный коэффициент теплопроводности k(x)")
plt.xlabel("x, М")
plt.ylabel("k(x)")
plt.show()

#Построение графика f(x)
plt.plot(stepX[1:-1], fc[1:-1])
plt.title("Распределенные источники тепла f(x)")
plt.xlabel("x, М")
plt.ylabel("f(x)")
plt.show()

#Получение решения СЛАУ методом NumPy
npTemp = np.linalg.solve(mainMatrix, fc)
#Получение решения СЛАУ методом прогонки
numericMethodTemp = numericMethod(a, b, c, fc)

#Построение графика распределения температуры (метод библиотеки NumPy)
plt.plot(stepX, npTemp)
plt.title("График распределения температуры (метод библиотеки NumPy)")
plt.xlabel("x, М")
plt.ylabel("T, C")
plt.show()

if diagonalDominance:
    #Построение графика распределения температуры (метод прогонки)
    plt.plot(stepX, numericMethodTemp)
    plt.title("График распределения температуры (метод прогонки)")
    plt.xlabel("x, М")
    plt.ylabel("T, C")
    plt.show()

    #Вычисление погрешности между двумя решениями
    tempDifference = []
    for i in range(n):
        tempDifference.append(abs(npTemp[i] - numericMethodTemp[i]))

    #Построение графика погрешности между двумя температурными результатами
    plt.plot(stepX, tempDifference)
    plt.title("Погрешность в вычислениях температуры методом прогонки и методом библиотеки NumPy")
    plt.xlabel("x, М")
    plt.ylabel("E")
    plt.show()
else:
    print("Метод прогонки не использован - не выполнено условие диагонального преобладания")

#Вычисление теплового потока (решение NumPy)
npFlow = calcFlow(npTemp, kx)

#Построение графика распределения теплового потока q(x) (метод NumPy)
plt.plot(stepX, npFlow)
plt.title("Распределение теплового потока q(x) (метод библиотеки NumPy)")
plt.xlabel("x, М")
plt.ylabel("q(x), Вт/М^2")
plt.show()

if diagonalDominance:
    #Вычисление теплового потока (решение методом прогонки)
    numericMethodFlow = calcFlow(numericMethodTemp, kx)

    #Построение графика распределения теплового потока q(x) (метод прогонки)
    plt.plot(stepX, numericMethodFlow)
    plt.title("Распределение теплового потока q(x) (метод прогонки)")
    plt.xlabel("x, М")
    plt.ylabel("q(x), Вт/М^2")
    plt.show()

    #Вычисление погрешности между двумя тепловыми потоками
    flowDifference = []
    for i in range(n):
        flowDifference.append(abs(npFlow[i] - numericMethodFlow[i]))

    # Построение графика погрешности между двумя температурными результатами
    plt.plot(stepX, flowDifference)
    plt.title("Погрешность в вычислениях теплового потока методом прогонки и методом библиотеки NumPy")
    plt.xlabel("x, М")
    plt.ylabel("E")
    plt.show()