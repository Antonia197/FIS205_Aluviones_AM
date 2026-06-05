import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# =============================================================================
# BLOQUE 1: PARAMETRIZACIÓN ALEATORIA Y MUESTREO DE MONTE CARLO
# =============================================================================

# 1. Definición aleatoria de la geometría del canal para este experimento
H_LAYER = np.random.uniform(0.5, 3.0)       # Espesor total aleatorio (en metros)
ANGULO_DEG = np.random.uniform(10.0, 25.0)  # Ángulo aleatorio (en grados)
T_MAX = 5.0                                 # Tiempo máximo fijo de simulación (s)

# Convertimos el ángulo a radianes para los cálculos matemáticos de la reología
THETA_RAD = np.radians(ANGULO_DEG)

# 2. Generación de puntos en el interior del lodo mediante Monte Carlo Uniforme
N_f = 2000
y_collocation = torch.rand(N_f, 1) * H_LAYER  # Escala dinámicamente de 0 a el H elegido
t_collocation = torch.rand(N_f, 1) * T_MAX    # Escala de 0 a T_max

# 3. Generación de puntos para la Condición Inicial (t = 0)
N_ic = 500
y_inicial = torch.rand(N_ic, 1) * H_LAYER
t_inicial = torch.zeros(N_ic, 1)

# 4. Reporte por pantalla de las dimensiones utilizadas para la simulación
print("==========================================================")
print("     CONFIGURACIÓN GEOMÉTRICA GENERADA PARA EL AVANCE     ")
print("==========================================================")
print(f"-> Espesor del aluvión (h) : {H_LAYER:.3f} metros")
print(f"-> Inclinación del canal (θ): {ANGULO_DEG:.2f}° (equivalente a {THETA_RAD:.4f} rad)")
print(f"-> Tiempo de observación (T): {T_MAX} segundos")
print("==========================================================\n")
# =============================================================================
# BLOQUE 2: CONDICIONES DE BORDE DINÁMICAS (Para h aleatorio)
# =============================================================================
N_b = 500
t_borde = torch.rand(N_b, 1) * T_MAX  # El tiempo sigue corriendo en las fronteras

# 1. Condición en el Fondo (y = 0): No deslizamiento -> u(0, t) = 0
y_fondo = torch.zeros(N_b, 1)

# 2. Condición en la Superficie Libre (y = h aleatorio): Esfuerzo nulo -> du/dy = 0
y_superficie = torch.ones(N_b, 1) * H_LAYER

# =============================================================================
# BLOQUE 3: RED NEURONAL CONFIGURABLE
# =============================================================================

class AnalisisPINN(nn.Module):
    def __init__(self, num_capas_ocultas, neuronas_por_capa):
        super(AnalisisPINN, self).__init__()
        
        capas = []
        # Capa de entrada: Recibe 2 variables (y, t)
        capas.append(nn.Linear(2, neuronas_por_capa))
        capas.append(nn.Tanh())
        
        # Capas ocultas intermedias
        for _ in range(num_capas_ocultas - 1):
            capas.append(nn.Linear(neuronas_por_capa, neuronas_por_capa))
            capas.append(nn.Tanh())
            
        # Capa de salida: Entrega 1 variable (Velocidad aproximada u_hat)
        capas.append(nn.Linear(neuronas_por_capa, 1))
        
        # Juntamos todo en un contenedor secuencial
        self.net = nn.Sequential(*capas)
        
    def forward(self, y, t):
        # Concatenamos las columnas de espacio y tiempo para ingresar a la red
        input_data = torch.cat([y, t], dim=1)
        return self.net(input_data)
    
# =============================================================================
# BLOQUE 4: OPERADOR FÍSICO Y DIFERENCIACIÓN AUTOMÁTICA (AUTOGRAD)
# =============================================================================

# Definimos las constantes reológicas de Pouliquen y Rojas como tensores
RHO = torch.tensor(1800.0, dtype=torch.float32)     # Densidad del aluvión (kg/m3)
RHO_S = torch.tensor(2500.0, dtype=torch.float32)   # Densidad de los granos sólidos
D_GRAIN = torch.tensor(0.01, dtype=torch.float32)   # Diámetro medio de partícula (m)
G_ACC = torch.tensor(9.81, dtype=torch.float32)     # Gravedad (m/s2)

MU_1 = torch.tensor(0.32, dtype=torch.float32)      # Fricción estática mínima
MU_2 = torch.tensor(0.60, dtype=torch.float32)      # Fricción máxima a altas tasas
I_0 = torch.tensor(0.30, dtype=torch.float32)       # Número de inercia crítico

def calcular_residuo_aluvion(model, y, t, H_layer, theta_rad):
    """
    Calcula el residuo de la ecuación de momento transiente usando la reología mu(I).
    Recibe la geometría variable (H_layer y theta_rad) calculada en el Bloque 1.
    """
    # Aseguramos que las variables geométricas sean tensores para su uso en Autograd
    if not isinstance(H_layer, torch.Tensor):
        H_layer = torch.tensor(H_layer, dtype=torch.float32)
    if not isinstance(theta_rad, torch.Tensor):
        theta_rad = torch.tensor(theta_rad, dtype=torch.float32)

    # Paso 1: Forzar a PyTorch a rastrear los gradientes de las variables independientes
    y.requires_grad_(True)
    t.requires_grad_(True)
    
    # Paso 2: Evaluación del aproximador universal (PINN)
    u = model(y, t)
    
    # Paso 3: Primeras derivadas exactas con Autograd
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # Paso 4: Cálculo de la Presión Hidrostática Variable (Ajuste del avance 1)
    # Acoplamos las variables aleatorias de inclinación y espesor dinámico
    P = RHO * G_ACC * torch.cos(theta_rad) * (H_layer - y)
    
    # ESTABILIZACIÓN NUMÉRICA CRÍTICA: Evitamos P = 0 en la superficie libre (y = H)
    P = torch.clamp(P, min=1e-3)
    
    # Paso 5: Magnitud de la tasa de deformación por cizalle (gamma punto)
    # También la protegemos para evitar raíces de cero absoluto
    gamma_dot = torch.clamp(torch.abs(u_y), min=1e-5)
    
    # Paso 6: Cálculo del Número de Inercia local transiente (I)
    I = (D_GRAIN * gamma_dot) / torch.sqrt(P / RHO_S)
    
    # Paso 7: Cierre reológico denso mu(I) de Pouliquen
    mu_I = MU_1 + (MU_2 - MU_1) / (I_0 / I + 1)
    
    # Paso 8: Esfuerzo de corte total (tau)
    tau = mu_I * P
    
    # Paso 9: Segunda derivada con Autograd -> Gradiente vertical del esfuerzo (d_tau / d_y)
    tau_y = torch.autograd.grad(tau, y, grad_outputs=torch.ones_like(tau), create_graph=True)[0]
    
    # Paso 10: Ecuación Diferencial Parcial de cantidad de movimiento balanceada
    # residuo = rho * u_t + d_tau/dy - rho * g * sin(theta)
    residuo = RHO * u_t + tau_y - RHO * G_ACC * torch.sin(theta_rad)
    
    return residuo

# =============================================================================
# BLOQUE 5: FUNCIÓN DE PÉRDIDA COMPUESTA Y BUCLE DE OPTIMIZACIÓN
# =============================================================================

def entrenar_experimento_aluvion(model, epochs, lr, lambda_ic, lambda_bc, H_layer, theta_rad):
    """
    Ejecuta el bucle de optimización permitiendo modificar los parámetros de 
    regularización (lambdas) para analizar el rendimiento.
    """

    # Convertimos las variables de geometría a tensores para su uso en Autograd
    H_layer = torch.tensor(H_layer, dtype=torch.float32)
    theta_rad = torch.tensor(theta_rad, dtype=torch.float32)

    # Definimos el optimizador Adam (se puede experimentar variando la tasa de aprendizaje 'lr')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Lista para almacenar el historial de errores y poder graficar la convergencia
    historial_loss = []
    
    print(f"Comenzando entrenamiento con Lambda_IC={lambda_ic} y Lambda_BC={lambda_bc}...")
    
    for epoch in range(epochs):
        # ---------------------------------------------------------------------
        # 1. Pérdida del Interior del Dominio (EDP - Reología de Rojas)
        # ---------------------------------------------------------------------
        # Recuperamos el residuo calculado con Autograd en el Bloque 4
        res_fisica = calcular_residuo_aluvion(model, y_collocation, t_collocation, H_layer, theta_rad)
        
        # El objetivo es que el residuo medio cuadrático tienda a cero
        loss_EDP = torch.mean(res_fisica ** 2)
        
        # ---------------------------------------------------------------------
        # 2. Pérdida de la Condición Inicial (IC: t = 0 -> Reposo)
        # ---------------------------------------------------------------------
        # Evaluamos la velocidad predicha por la red en el instante del arranque
        u_inicial_pred = model(y_inicial, t_inicial)
        
        # Como parte desde el reposo, penalizamos cualquier velocidad distinta de cero
        loss_IC = torch.mean(u_inicial_pred ** 2)
        
        # ---------------------------------------------------------------------
        # 3. Pérdida de Condición de Borde Inferior (BC Fondo: y = 0)
        # ---------------------------------------------------------------------
        # Evaluamos la velocidad en el lecho del canal a lo largo del tiempo
        u_fondo_pred = model(y_fondo, t_borde)
        
        # Condición de no deslizamiento: u(0, t) = 0
        loss_BC0 = torch.mean(u_fondo_pred ** 2)
        
        # ---------------------------------------------------------------------
        # 4. Pérdida de Condición de Borde Superior (BC Superficie: y = h)
        # ---------------------------------------------------------------------
        # Para evaluar el gradiente du/dy en la superficie, necesitamos activar Autograd en y_superficie
        y_superficie.requires_grad_(True)
        u_superficie_pred = model(y_superficie, t_borde)
        
        # Calculamos la derivada espacial en el borde superior
        u_y_superficie = torch.autograd.grad(
            u_superficie_pred, y_superficie, 
            grad_outputs=torch.ones_like(u_superficie_pred), 
            create_graph=True
        )[0]
        
        # Esfuerzo libre implica que el gradiente vertical de velocidad se anula
        loss_BCh = torch.mean(u_y_superficie ** 2)
        
        # ---------------------------------------------------------------------
        # 5. Combinación y Regularización Dinámica de Pérdidas (Hiperparámetros)
        # ---------------------------------------------------------------------
        # Aquí es donde aplicamos los parámetros de regularización lambda_ic y lambda_bc para ajustar la importancia relativa de cada término
        # Multiplicamos las infracciones de las fronteras por sus respectivos pesos lambdas.
        loss_total = loss_EDP + (lambda_ic * loss_IC) + lambda_bc * (loss_BC0 + loss_BCh)
        
        # ---------------------------------------------------------------------
        # 6. Algoritmo de Retropropagación y Actualización de Pesos
        # ---------------------------------------------------------------------
        optimizer.zero_grad()   # Limpiamos los gradientes acumulados del ciclo anterior
        loss_total.backward()   # Backpropagation: calcula los gradientes de la pérdida compuesta
        optimizer.step()        # El optimizador Adam ajusta los pesos de las capas del Bloque 3
        
        # Guardamos el registro numérico de la época
        historial_loss.append(loss_total.item())
        
        # Imprimimos el progreso cada cierto tramo para monitorear el rendimiento
        if epoch % 2 == 0:
            print(f"  Época {epoch:02d} | Pérdida Total: {loss_total.item():.4e} [EDP: {loss_EDP.item():.4e}, IC: {loss_IC.item():.4e}]")
            
    return historial_loss

# =============================================================================
# BLOQUE 6: EJECUCIÓN DEL LABORATORIO Y COMPARATIVA DE HIPERPARÁMETROS
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    # Definimos las épocas para las pruebas del laboratorio
    EPOCAS_TEST = 15
    TASA_APRENDIZAJE = 0.005

    print("Iniciando experimentos en el Laboratorio PINN...")
    cronometro_inicio = time.time()

    # -------------------------------------------------------------------------
    # Experimento 1: Configuración Base (Red estándar y regularización unitaria)
    # -------------------------------------------------------------------------
    print("\n--- EJECUTANDO EXPERIMENTO 1: Red Base (3 capas, 20 neuronas) ---")
    red_base = AnalisisPINN(num_capas_ocultas=3, neuronas_por_capa=20)
    historial_exp1 = entrenar_experimento_aluvion(
        model=red_base, epochs=EPOCAS_TEST, lr=TASA_APRENDIZAJE,
        lambda_ic=1.0, lambda_bc=1.0, H_layer=H_LAYER, theta_rad=THETA_RAD
    )

    # -------------------------------------------------------------------------
    # Experimento 2: Incremento de Complejidad
    # -------------------------------------------------------------------------
    print("\n--- EJECUTANDO EXPERIMENTO 2: Red Profunda (5 capas, 40 neuronas) ---")
    red_profunda = AnalisisPINN(num_capas_ocultas=5, neuronas_por_capa=40)
    historial_exp2 = entrenar_experimento_aluvion(
        model=red_profunda, epochs=EPOCAS_TEST, lr=TASA_APRENDIZAJE,
        lambda_ic=1.0, lambda_bc=1.0, H_layer=H_LAYER, theta_rad=THETA_RAD
    )

    # -------------------------------------------------------------------------
    # Experimento 3: Alteración de Regularización (Alta penalización en contornos)
    # -------------------------------------------------------------------------
    print("\n--- EJECUTANDO EXPERIMENTO 3: Alta Regularización (Lambdas = 10.0) ---")
    red_regularizada = AnalisisPINN(num_capas_ocultas=3, neuronas_por_capa=20)
    historial_exp3 = entrenar_experimento_aluvion(
        model=red_regularizada, epochs=EPOCAS_TEST, lr=TASA_APRENDIZAJE,
        lambda_ic=10.0, lambda_bc=10.0, H_layer=H_LAYER, theta_rad=THETA_RAD
    )

    tiempo_total = time.time() - cronometro_inicio
    print(f"\n==========================================================")
    print(f"Laboratorio completado con éxito en {tiempo_total:.2f} segundos.")
    print(f"==========================================================")

    # -------------------------------------------------------------------------
    # GENERACIÓN DEL GRÁFICO COMPARATIVO
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(historial_exp1, 's-', color='tab:blue', label='Exp 1: Base (3 capas, 20 neuronas, $\lambda=1$)')
    plt.plot(historial_exp2, 'o-', color='tab:orange', label='Exp 2: Profunda (5 capas, 40 neuronas, $\lambda=1$)')
    plt.plot(historial_exp3, '^-', color='tab:green', label='Exp 3: Sobre-Regularizado ($\lambda=10$)')

    plt.yscale('log') # Escala logarítmica para evaluar órdenes de magnitud de error
    plt.xlabel('Iteraciones de Optimización (Épocas)')
    plt.ylabel('Función de Pérdida Compuesta $\mathcal{L}_{total}$ (MSE)')
    plt.title('Estudio Comparativo de Hiperparámetros - Modelo Friccional Transiente')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(loc='upper right')
    
    # Guarda el gráfico automáticamente en tu carpeta del proyecto
    plt.savefig('convergencia_hiperparametros.png', dpi=300, bbox_inches='tight')
    print("Gráfico 'convergencia_hiperparametros.png' guardado exitosamente.")
    plt.show()

    # =========================================================================
    # ANÁLISIS EXTRA: MAPA DE CALOR DEL RESIDUO EN EL ESPACIO-TIEMPO
    # =========================================================================
    print("\nGenerando Mapa de Calor del Residuo Físico...")

    # 1. Creamos ejes regulares fijos de 100 puntos para el espacio y el tiempo
    y_eje = np.linspace(0, H_LAYER, 100)
    t_eje = np.linspace(0, T_MAX, 100)

    # 2. Construimos la malla bidimensional (Meshgrid)
    T_malla, Y_malla = np.meshgrid(t_eje, y_eje)

    # 3. Transformamos la malla a vectores columna de PyTorch para el operador físico
    # Usamos flatten() para estirarlos y luego los convertimos en tensores
    y_tensor_malla = torch.tensor(Y_malla.flatten(), dtype=torch.float32).view(-1, 1)
    t_tensor_malla = torch.tensor(T_malla.flatten(), dtype=torch.float32).view(-1, 1)

    # 4. Evaluamos el residuo físico exacto en los 10.000 puntos usando la red profunda
    with torch.set_grad_enabled(True): # Necesario para que Autograd pueda derivar en la malla
        residuo_malla = calcular_residuo_aluvion(
            red_profunda, y_tensor_malla, t_tensor_malla, H_LAYER, THETA_RAD
        )
        
        # Pasamos el resultado a valor absoluto numérico de NumPy
        error_absoluto = torch.abs(residuo_malla).detach().numpy()

    # 5. Redimensionamos el error para que vuelva a tener la forma de la malla (100x100)
    Error_2D = error_absoluto.reshape(100, 100)

    # 6. Graficamos el mapa de calor continuo
    plt.figure(figsize=(8, 6))
    
    # pcolormesh creará el mapa de calor usando los ejes y la matriz de errores
    pcm = plt.pcolormesh(T_malla, Y_malla, Error_2D, cmap='inferno', shading='auto')
    
    # Agregamos la barra de colores para medir la magnitud del error
    plt.colorbar(pcm, label='Magnitud Absoluta del Residuo de la EDP')
    
    plt.xlabel('Eje de Tiempo $t$ [segundos]')
    plt.ylabel('Altura sobre el lecho del canal $y$ [metros]')
    plt.title('Distribución Espacio-Temporal del Error Físico (Análisis de Robustez)')
    plt.grid(True, ls='--', alpha=0.3)
    
    # Guardamos el gráfico
    plt.savefig('mapa_calor_residuo.png', dpi=300, bbox_inches='tight')
    plt.show()
