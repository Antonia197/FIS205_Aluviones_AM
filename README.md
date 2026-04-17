# Modelamiento de Reología Transiente en Aluviones mediante PINNs
**Estudiante:** Antonia Migliassi
**Curso:** FIS205 Física Computacional - USM

## Estado del Avance 1 (Preliminar)
Este repositorio contiene la arquitectura base de una Red Neuronal Informada por la Física (PINN) para resolver la dinámica de flujos granulares.

### Implementación actual:
* **Solver Físico:** Motor basado en PyTorch que utiliza Diferenciación Automática (Autograd) para evaluar el residuo de la ecuación de momentum de Rojas (2015).
* **Condiciones de Borde:** Implementación de penalización para condición de no-deslizamiento en el fondo ($y=0$).
* **Optimización:** Loop de entrenamiento inicial utilizando el algoritmo Adam.

### Cómo ejecutar:
1. Instalar dependencias: `pip install -r requerimientos.txt`
2. Ejecutar: `python modelado_aluvion.py`
