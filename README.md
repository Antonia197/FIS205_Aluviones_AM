# Modelado de Aluviones mediante Redes Neuronales Informadas por la Física (PINNs)
**Estudiante:** Antonia Migliassi
**Profesores:** Ariel Norambuena; Nicolás Viaux
**Institución:** Universidad Técnica Federico Santa María  
**Hito:** Segundo Avance

## Descripción del Proyecto
Este repositorio contiene el marco computacional desarrollado en Python (PyTorch) para resolver de forma continua la ecuación de momento transiente de un flujo granular denso sobre un canal inclinado, incorporando la reología no-lineal $\mu(I)$ de Pouliquen y las correcciones de presión hidrostática variable de la tesis de Rojas.

## Arquitectura del Código
El script principal está estructurado de forma modular en 6 bloques lógicos:
1. **Muestreo Estocástico (Monte Carlo):** Generación de puntos de colocación en el dominio interno y condiciones iniciales ($t=0$) parametrizadas ante geometrías aleatorias.
2. **Condiciones de Borde Dinámicas:** Adaptación automática del lecho ($y=0$, no deslizamiento) y la superficie libre ($y=h$, esfuerzo nulo) según el espesor generado.
3. **Red Neuronal Configurable:** Clase orientada a objetos que permite modificar dinámicamente el número de capas ocultas y neuronas utilizando activaciones suaves (`Tanh`).
4. **Operador Físico (Autograd):** Cálculo de residuos diferenciales exactos mediante diferenciación automática, incluyendo estabilización numérica con `torch.clamp`.
5. **Función de Pérdida Compuesta:** Bucle de entrenamiento que pondera el residuo de la EDP, condiciones de contorno e iniciales mediante factores de regularización ($\lambda_{ic}, \lambda_{bc}$).
6. **Módulo del Laboratorio:** Automatización de experimentos comparativos y herramientas de diagnóstico visual.


## Ejecución
Para correr los ensayos del laboratorio de hiperparámetros y generar los análisis de convergencia y robustez, ejecute:

1. Instalar dependencias: `pip install -r requerimientos.txt`
2. Ejecutar: `python modelado_aluvion.py`
