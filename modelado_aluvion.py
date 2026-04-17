import torch
import torch.nn as nn
import numpy as np

class AluvionPINN(nn.Module):
    def __init__(self):
        super(AluvionPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 3) 
        )

    def forward(self, y, t):
        return self.net(torch.cat([y, t], dim=1))

def f_loss(model, y, t):
    rho = 2000.0  # kg/m3 
    g = 9.81
    theta = np.radians(20)
    y.requires_grad_(True)
    t.requires_grad_(True)
    
    pred = model(y, t)
    u = pred[:, 0:1]
    P = pred[:, 1:2]
    tau = pred[:, 2:3]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    tau_y = torch.autograd.grad(tau, y, torch.ones_like(tau), create_graph=True)[0]
    res_momentum = rho * u_t + tau_y - rho * g * np.sin(theta)
    y_bottom = torch.zeros_like(t) # y = 0
    u_bottom = model(y_bottom, t)[:, 0:1] # Velocidad en el fondo
    loss_bc = torch.mean(u_bottom**2) # Queremos que u sea 0 en el fondo
    
    return res_momentum + loss_bc

if __name__ == "__main__":
    print("Iniciando el motor de física PINN...")
    model = AluvionPINN()
    y_test = torch.rand(100, 1) * 0.1
    t_test = torch.rand(100, 1) * 5.0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for step in range(10):
        optimizer.zero_grad()
        loss = f_loss(model, y_test, t_test)
        loss.backward()
        optimizer.step()
        if step % 2 == 0:
            print(f"Paso {step}: Pérdida = {loss.item():.2f}")
    # -----------------------------------------------

    print(f"Modelo cargado y optimización inicial completa.")
    print(f"Pérdida final de este test: {loss.item():.6f}")