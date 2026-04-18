#Problema 1: Dinámica cuántica y complejidad computacional

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time


#Inciso a
# 

#Inciso b


#se definen las matrices de Pauli para los operadores locales
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])


#Función para construir los operadores locales en el espacio de Hilbert de N espines
def op_local(N, j, pauli):
    return np.kron(np.eye(2**j), np.kron(pauli, np.eye(2**(N-j-1))))


#Constreucción de la matriz Hamiltoniana para el modelo de Ising con campo magnético transversal
def hamiltonian(N, J, B):
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N-1):
        H += J * np.dot(op_local(N, i, sx), op_local(N, i+1, sx))
    for i in range(N):
        H += B * op_local(N, i, sz)
    return H

#Evolución temporal del estadio inicial bajo el hamiltoniano H
def simulate_evolution(N, J, B, t_array):
    psi_0 = np.zeros(2**N, dtype=complex)
    psi_0[-1] = 1 
    
    H = hamiltonian(N, J, B)
    probabilities = []
    

    for t in t_array:
        U = expm(-1j * H * t)
        psi_t = np.dot(U, psi_0)
        p_t = np.abs(np.vdot(psi_t, psi_0))**2
        probabilities.append(p_t)
        
    return probabilities

# Inciso c
N_spins = 6
tiempos = np.linspace(0, 10, 100)

plt.figure(figsize=(10, 6))

# Graficación para los csos solicitados: B/J = 0.1, 1.0, 10.0

for label, ratio in [("B/J=0.1", 0.1), ("B/J=1.0", 1.0), ("B/J=10.0", 10.0)]:
    p_t = simulate_evolution(N_spins, 1.0, ratio, tiempos)
    plt.plot(tiempos, p_t, label=label)

plt.xlabel("Tiempo (t)")
plt.ylabel("Probabilidad p(t)")
plt.legend()
plt.title(f"Evolución Temporal para N={N_spins} espines")
plt.grid(True)
plt.show() 

#Inciso d y e

#Para mayor comodidad, usaremos las funciones anteriores para medir el tiempo 
#Para construir y diagonalizar el hamiltonia para tamaños N=4,5,6,7,8
#Por comodidad se realizará el inciso e de forma conjunta

print("Inciso d y e: Medición de tiempos y gráfico")
N_particulares = [4, 5, 6, 7, 8]
t_pronedio= []
num_reañizaciones = 5

for n in N_particulares:
    tiempos_n = []
    for _ in range(num_reañizaciones):
        start_time = time.time()
        H = hamiltonian(n, J=1.0, B=1.0)
        evals, evecs= np.linalg.eigh(H)
        end_time = time.time()
        tiempos_n.append(end_time - start_time)

    avg= np.mean(tiempos_n)
    t_pronedio.append(avg)
    print(f"N={n}, Tiempo promedio: {avg:.4f} segundos")

#Graficar inciso e

plt.figure(figsize=(8, 5))
plt.plot(N_particulares, t_pronedio, marker='o', color= 'blue', label='Tiempo medido')
plt.xlabel('Número de espines (N)')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.title('Escalamiento del tiempo de cálculo vs N')
plt.legend()
plt.grid(True)
plt.show()

#Inciso f (discusión)
#Para estimar el tiempo para N= 20,50,100 podemos usar la información presentada en el inciso a
#Donde sabemos que la dimensión del espacio de Hilbert crece como 2^N, donde para
# 2**8 el tiempo promedio fue de 0.18 segundos, nos será útil mencionar que el 
# orden de complejidad de la diagonalización de una matriz es O(d^3), donde d es la dimensión de la matriz.
#  Por lo tanto, el tiempo de cálculo para N espines se puede estimar como T(N) ~ (2^N)^3 = 2^(3N).
# Usando esta relación, podemos estimar el tiempo para N=20, 50 y 100:

print("Inciso f: Estimación de tiempos para N=20, 50, 100")
N_estimar = [20, 50, 100]
tiempos_estimados = [0.18 * (2**(3*(n-8))) for n in N_estimar]
for n, t in zip(N_estimar, tiempos_estimados):
    print(f"N={n}, Tiempo estimado: {t:.2e} segundos")

#De esos resultados los pasamos a escalas de tiempo más comprensibles:

for n, t in zip(N_estimar, tiempos_estimados):
    if t < 60:
        print(f"N={n}, Tiempo estimado: {t:.2e} segundos")
    elif t < 3600:
        print(f"N={n}, Tiempo estimado: {t/60:.2f} minutos")
    elif t < 86400:
        print(f"N={n}, Tiempo estimado: {t/3600:.2f} horas")
    else:
        print(f"N={n}, Tiempo estimado: {t/86400:.2f} días")

#Con esto podemos evidenciar que la dimensionalidad del espacio de Hilbert posee una expansión
# de forma exponencial, lo que hace que el tiempo de cálculo para sistemas con un número grande de espines sea prohibitivo,
# incluso con los recursos computacionales más avanzados disponibles en la actualidad.