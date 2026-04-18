import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#definimos parámetros

g=9.81 #m/s^2
rho=1.225 #kg/m^3
C_d=0.3
A=0.05 #m^2
m=100 #kg
omega_t= 7.292e-5 #rad/s
latitud= np.radians(45) #grados a radianes
omega= omega_t * np.array([0,np.cos(latitud),np.sin(latitud)]) #vector de velocidad angular de la Tierra

def modelo_fisico(t,y):
    v = y[3:]
    v_magnitud = np.linalg.norm(v)

    a_g= np.array([0, 0, -g])
    a_r= -0.5 * rho * C_d * A * v_magnitud * v / m
    a_c= -2 * np.cross(omega, v)

    a_tiotal= a_g + a_r + a_c
    return [v[0], v[1], v[2], a_tiotal[0], a_tiotal[1], a_tiotal[2]]



#definimos las condiciones iniciales del misil 1

v0= 500 #m/s
teta_1= np.radians(45) #grados a radianes
psi1= np.radians(30) #grados a radianes
vx1= v0 * np.cos(teta_1) * np.sin(psi1)
vy1= v0 * np.cos(teta_1) * np.cos(psi1)
vz1= v0 * np.sin(teta_1)
y0_misil1= [0, 0, 0, vx1, vy1, vz1]

def impacto_suelo(t,y):
    return y[2] #altura del misil
impacto_suelo.terminal= True
impacto_suelo.direction= -1

t_span= (0, 200) #tiempo de simulación
sol_misil1= solve_ivp(modelo_fisico, t_span, y0_misil1, events=impacto_suelo, max_step=0.1)

#Grafico la trayectoria del misil 1
plt.figure(figsize=(8,5))
plt.plot(sol_misil1.y[0], sol_misil1.y[1], label='Misil 1')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trayectoria del Misil 1')
plt.legend()
plt.grid(True)
plt.show()

print(f"Impacto detectado a los {sol_misil1.t[-1]:.2f} segundos")

#Inciso c
#Ahora para determinar los valores restantes del segundo misil, usaremos minimize de scipy para encontrar los valores de teta_2 y psi_2 que minimicen la distancia entre los dos misiles en el tiempo, con la condición de que el segundo misil se lance a los 10 segundos.


from scipy.optimize import minimize
from scipy.interpolate import interp1d


pos_1= interp1d(sol_misil1.t, sol_misil1.y[:3], axis=1,kind='cubic', fill_value="extrapolate")

def distancia_misil2(params):
    v0_2, teta_2, psi_2 = params
    teta_2= np.radians(teta_2)
    psi_2= np.radians(psi_2)
    vx2= v0_2 * np.cos(teta_2) * np.sin(psi_2)
    vy2= v0_2 * np.cos(teta_2) * np.cos(psi_2)
    vz2= v0_2 * np.sin(teta_2)
    y0_misil2= [5000, 2000, 0, vx2, vy2, vz2]
    
    sol_m2= solve_ivp(modelo_fisico, (10, 200), y0_misil2, events=impacto_suelo, max_step=0.5)

    p2= sol_m2.y[:3].T
    p1= pos_1(sol_m2.t).T

    distancia= np.linalg.norm(p1 - p2, axis=1)
    return np.min(distancia)

#Valores iniciales para la optimización
test= [500, 45, 30] #v0_2, teta_2, psi_2

res= minimize(distancia_misil2, test, method='Nelder-Mead', tol=1e-2)   

v0_2_opt, teta_2_opt, psi_2_opt = res.x
print(f"Valores óptimos para el misil 2: v0={v0_2_opt:.2f} m/s, teta={teta_2_opt:.2f} grados, psi={psi_2_opt:.2f} grados")
print(f"Distancia mínima entre los misiles: {res.fun:.2f} metros")

#Inciso d
# Ahora determinamos el tiempo desde misil objetivo hasta el impacto

teta_2_opt_rad= np.radians(teta_2_opt)
psi_2_opt_rad= np.radians(psi_2_opt)
vx2_opt= v0_2_opt * np.cos(teta_2_opt_rad) * np.sin(psi_2_opt_rad)
vy2_opt= v0_2_opt * np.cos(teta_2_opt_rad) * np.cos(psi_2_opt_rad)
vz2_opt= v0_2_opt * np.sin(teta_2_opt_rad)
y0_misil2_opt= [5000, 2000, 0, vx2_opt, vy2_opt, vz2_opt]
sol_m2_opt= solve_ivp(modelo_fisico, (10, 200), y0_misil2_opt, events=impacto_suelo, max_step=0.1)
p2= sol_m2_opt.y[:3].T
p1= pos_1(sol_m2_opt.t).T
distancia= np.linalg.norm(p1 - p2, axis=1)
impacto_index= np.argmin(distancia)
t_colision= sol_m2_opt.t[impacto_index]
print(f"Tiempo desde el lanzamiento del misil objetivo hasta el impacto: {t_colision:.2f} segundos")

# Inciso e
# Graficamos las trayectorias de ambos misiles en un gráfico estático 3D

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
x_col= sol_m2_opt.y[0][impacto_index]
y_col= sol_m2_opt.y[1][impacto_index]
z_col= sol_m2_opt.y[2][impacto_index]
ax.scatter(x_col, y_col, z_col, color='red', label='Colisión', s=100, marker='X')
ax.plot(sol_m2_opt.y[0, : impacto_index+1],
        sol_m2_opt.y[1, : impacto_index+1],
        sol_m2_opt.y[2, : impacto_index+1], label='Misil 2 hasta impactar', color='blue')
ax.plot(sol_misil1.y[0], sol_misil1.y[1], sol_misil1.y[2], label='Misil 1', color='green', alpha=0.6)  
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Trayectorias de los Misiles')
ax.legend()
plt.show()

# Inciso f
# Animación de las trayectorias de ambos misiles
from matplotlib.animation import FuncAnimation

pos_2_func= interp1d(sol_m2_opt.t, sol_m2_opt.y[:3], axis=1, kind='cubic', fill_value="extrapolate")
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
t_anim= np.linspace(0, t_colision, 100)

def update(frame):
    ax.clear()
    t_actual= t_anim[frame]
    
    #misil 1
    
    p1= pos_1(t_anim[:frame+1]).T
    ax.plot(p1[:,0], p1[:,1], p1[:,2],'g-', label='Misil 1', alpha=0.6)

    #misil 2
    if t_actual >= 10:
        t_m2_range= t_anim[(t_anim >= 10) & (t_anim <= t_actual)]
        if len (t_m2_range) > 0:
            p2= pos_2_func(t_m2_range).T
            ax.plot(p2[:,0], p2[:,1], p2[:,2], 'b-', label='Misil 2')


    #Colision
    if t_actual >= t_colision:
        ax.scatter(x_col, y_col, z_col, color='red', label='Colisión', s=100, marker='X')

    ax.set_xlim(0, 6000); ax.set_ylim(0, 9000); ax.set_zlim(0, 4000)
    ax.set_title(f'Tiempo: {t_actual:.1f} s')
    ax.legend()

ani= FuncAnimation(fig, update, frames=len(t_anim), interval=50)
plt.show()