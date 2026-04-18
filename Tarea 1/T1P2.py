#PRoblema 2: Transformada de Fourier y complejidad algorítmica

#Inciso a 

import numpy as np
import matplotlib.pyplot as plt

f1 = 50 
f2 = 120  
A1 = 1
A2 = 0.5 
t = np.linspace(0, 2, 1000)  
senal1 = A1 * np.sin(2 * np.pi * f1 * t)
senal2 = A2 * np.sin(2 * np.pi * f2 * t)
x_n = senal1 + senal2

#Inciso b (Implementación de transformada discreta de fourier)

def dft(x):
    N= len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

#inciso c (graficar el espectro obtenido)

X_k = dft(x_n)
frecuencias = np.fft.fftfreq(len(x_n), d=t[1] - t[0])
plt.figure(figsize=(10, 6))
plt.plot(frecuencias[:len(frecuencias)//2], np.abs(X_k)[:len(X_k)//2])
plt.title("Espectro de la señal compuesta")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()
plt.show()

#Inciso d (Calculo de la transformada )

import time

start_time = time.time()
X_k_fft = np.fft.fft(x_n)
end_time = time.time()
print(f"Tiempo de ejecución de FFT: {end_time - start_time} segundos")

#inciso e (comparar tiempo de ejecución entre DFT y FFT para señales N= 10^n con n=2,3,4,5)

N_values = [10**2, 10**3, 10**4, 10**5]
dft_times = []
fft_times = []
for N in N_values:
    random_x_n = np.random.rand(N)   
    start_time = time.time()
    dft(random_x_n)
    end_time = time.time()
    dft_times.append(end_time - start_time)
    
    start_time = time.time()
    np.fft.fft(random_x_n)
    end_time = time.time()
    fft_times.append(end_time - start_time)
    print(f"N={N}: DFT tiempo = {dft_times[-1]:.4f} s, FFT tiempo = {fft_times[-1]:.4f} s")


# Inciso f: Graficar los tiempos de ejecución de N para ambos algoritmos

plt.figure(figsize=(10, 6))
plt.plot(N_values, dft_times, label='DFT', marker='o')
plt.plot(N_values, fft_times, label='FFT', marker='o')
plt.xlabel('Tamaño de la señal (N)')
plt.ylabel('Tiempo de ejecución (s)')
plt.title('Comparación de tiempos de ejecución entre DFT y FFT')
plt.legend()
plt.grid()
plt.show()

# Inciso g (Repetir graficaciòn en escala logarítmica)

plt.figure(figsize=(10, 6))
plt.plot(N_values, dft_times, label='DFT', marker='o')
plt.plot(N_values, fft_times, label='FFT', marker='o')
plt.xlabel('Tamaño de la señal (N)')
plt.ylabel('Tiempo de ejecución (s)')
plt.title('Comparación de tiempos de ejecución entre DFT y FFT')
plt.legend()
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.show()
expoesc_dft = np.polyfit(np.log(N_values), np.log(dft_times), 1)[0]
expoesc_fft = np.polyfit(np.log(N_values), np.log(fft_times), 1)[0]
print(f"Exponente de escalamiento para DFT: {expoesc_dft:.2f}")
print(f"Exponente de escalamiento para FFT: {expoesc_fft:.2f}")

#Inciso h: Hallar N para que FFT sea 100 veces más rápida que DFT

for N, dft_time, fft_time in zip(N_values, dft_times, fft_times):
    if dft_time / fft_time >= 100:
        print(f"Para N={N}, la FFT es al menos 100 veces más rápida que la DFT.")
        break