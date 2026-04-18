# Tarea 1: Física Computacional (FIS205)

Este repositorio contiene las soluciones computacionales para la Tarea 1 del curso FIS205. Se abordan problemas de dinámica cuántica, procesamiento de señales y mecánica clásica utilizando Python.

## Contenidos del Repositorio

* **`T1P1.py`**: Simulación del modelo de Ising transversal para una cadena de $N$ espines $1/2$. Incluye el cálculo de la evolución temporal y estimaciones de complejidad computacional ($O(2^{3N})$).
* **`T1P2.py`**: Comparación de rendimiento entre la Transformada Discreta de Fourier (DFT) y la Transformada Rápida de Fourier (FFT), analizando el escalamiento algorítmico $O(N^2)$ vs $O(N \log N)$.
* **`T1P3.py`**: Simulación de trayectorias balísticas interceptoras. Implementa un modelo de resolución de ecuaciones diferenciales (ODE) considerando:
    * Gravedad constante.
    * Roce aerodinámico cuadrático ($F \propto v^2$).
    * Fuerza de Coriolis (Sistema de referencia rotatorio a 45°N).

## Requisitos

Para ejecutar los scripts, es necesario contar con un entorno de Python 3.x y las siguientes librerías:

```bash
pip install numpy matplotlib scipy
