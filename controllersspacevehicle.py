import pygame
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from control.matlab import lqr
from control.matlab import ctrb
from numpy.linalg import matrix_rank

# System parameters
global x, u
global raio


# Initialise pygame
pygame.init()

# Set the screen size
WIDTH = 800
HEIGHT = 600

# Define as cores
WHITE = (255, 255, 255)
RED   = (188, 39, 50)
GREEN = (0, 255, 0)

# Run the Pygame window
WIN   = None

# variables
ixx = WIDTH/2
iyy = HEIGHT/2
izz = 100
ixy = 0
ixz = 0
iyx = 0
iyz = 0
izx = 0
izy = 0

#platform mass
mass = 20 # kg

#gravity
g = 1.81

# The forces are 2 Newton! (fz is not taken into account)
fx = 0
fy = 0
fz = 0

# Rotation according to N! (L and M are not taken into account)
n = 0

#All the necessary variables
phi   = 0
theta = 0
psi   = 90 * np.pi / 180
l     = 0
m     = 0
p     = 0
q     = 0
r     = 0
pn    = 0
pe    = 0
u     = 0
v     = 0
w     = 0

# Initialize linearization matrices
A, B = None, None

# Linearisation (X and initial U)
# x  = [phi, theta, psi, p, q, r, pn, pe, h, u, v, w]
# u  = [fx fy fz l m n]
t         = 0
x0        = [psi, r, pn, pe, u, v]
u_control = [n, fx, fy]


# X equivalent
pn_eq  = float(input("Digite o pn_eq desejado: "))
pe_eq  = float(input("Digite o pe_eq desejado: "))
psi_eq = float(input("Digite o psi_eq desejado [em graus]: "))

x_eq = np.array([psi_eq * np.pi /180, 0, pn_eq, pe_eq, 0, 0])


################################################################################
################################################################################


def f(t, x, u_control):

    global g, mass, ixx, ixy, ixz, iyx, iyy, iyz, izx, izy, izz
    global theta, phi, q, p, l, m, fz , w, mass


    #x0 = [psi0, r0, pn0, pe0, u0, v0]
    psi = x[0]
    r   = x[1]
    pn  = x[2]
    pe  = x[3]
    u   = x[4]
    v   = x[5]
    #print(psi, r, pn, pe, u,)

    # u_control = [n, fx, fy]
    n  = u_control[0]
    fx = u_control[1]
    fy = u_control[2]
    #print(n, fx, fy)

     

    # Kinematics equations (only phi_dot is needed)

    # phi_dot   = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    # theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot   = (q * np.sin(phi) + r * np.cos(phi)) * (1 / np.cos(theta))
    #print(psi_dot)

    # p = phi_dot - psi_dot * np.sin(theta)
    # q = theta_dot * np.cos(phi) + psi_dot * np.cos(theta) * np.sin(phi)
    # r = psi_dot * np.cos(theta) * np.cos(phi) - theta_dot * np.sin(phi)


    # Rotation equations (only r_dot is needed)

    #q_dot = m - (ixx -izz) * p* r + ixz * (p**2 - r**2) 
    #p_dot = l + ixz * (r_dot) + ixz * p * q - (izz - iyy) * r * q
    #r_dot = (ixx * ((ixx - iyy) + (ixz**2)) + (ixz**2)) * p * q - ixz * (ixx - iyy + ixz) * q * r + ixz * l + ixx * n
    r_dot = n/izz



    # Matrix of direct cosines

    mrc = np.array([[np.cos(theta) * np.cos(psi) ,   np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi) ,  np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
                    [np.cos(theta) * np.sin(psi) ,   np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi) ,  np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
                    [-np.sin(theta)              ,   np.sin(phi) * np.cos(theta)                                           ,  np.cos(phi) * np.cos(theta)]])


    cal = np.dot(mrc, np.array([[u, v, w]]).T)

    uf = cal[0]
    vf = cal[1]

    pn_dot = uf
    pe_dot = vf


    # Translation Equation
    u_dot = (fx/mass) - g * np.sin(theta) + v * r - w * q
    v_dot = (fy/mass) + g * np.sin(phi) * np.cos(theta) - u * r + w * p
    #w_dot = g * np.cos(phi) * np.cos(theta) + fz + u * q - v * p


    # Result of the derivative of X in a vector
    x_dot = np.array([psi_dot, r_dot, pn_dot.item(), pe_dot.item(), u_dot, v_dot])


    return x_dot


################################################################################
################################################################################

def calcular_u_eq(x_eq):
    def fun(u):
        return f(0, x_eq, u)


    u0 = [1, 1, 1]
    result = least_squares(fun, u0)
    u_eq = result.x
    return u_eq

u_eq = calcular_u_eq(x_eq) 
print("u_eq:", u_eq)


################################################################################
################################################################################


def linearizacao(x_eq, u_eq):
    h = 1e-5
    t = 0

    num_col_A = len(x_eq)
    num_col_B = len(u_eq)

    A = np.zeros((num_col_A, num_col_A))
    B = np.zeros((num_col_A, num_col_B))

    Ha = np.eye(num_col_A, num_col_A)
    for j in range(num_col_A):
        v = Ha[:,j]
        A[:, j] = (f(t, x_eq + h*v, u_eq) - f(t, x_eq, u_eq)) / h

    Hb = np.eye(num_col_B, num_col_B)
    for j in range(num_col_B):
        v = Hb[:,j]
        B[:, j] = (f(t, x_eq, u_eq + h*v) - f(t, x_eq, u_eq)) / h

    return A, B

A, B = linearizacao(x_eq, u_eq) # 3rd to be executed
print("Matriz A:")
print(A)
print("Matriz B:")
print(B)

################################################################################
################################################################################

def controlador_lqr(A, B):
    rho = 100
    rho = 0.000001

    # controllability
    Mc      = ctrb(A, B)
    rank_Mc = matrix_rank(Mc)

    Q = np.eye(6)
    R = np.eye(3) * rho

    K = lqr(A, B, Q, R)[0]
    print(K)

    Acl = A - np.dot(B, K)
    lambda_Acl = np.linalg.eig(Acl)[0]
    print(f"lambda_Acl = \n {lambda_Acl}")

    return K

K = controlador_lqr(A, B)

################################################################################
################################################################################

def f_real(t, x, un_next_real):
    global g, mass, ixx, ixy, ixz, iyx, iyy, iyz, izx, izy, izz
    global theta, phi, q, p, l, m, fz, w

    raio = 1

    #x0 = [psi, r, pn, pe, u, v]
    psi = x[0]
    r   = x[1]
    pn  = x[2]
    pe  = x[3]
    u   = x[4]
    v   = x[5]

    
    f1, f2, f3, f4, f5, f6, f7, f8 = un_next_real

    # Now we can use f1, f2, f3, f4, f5, f6, f7 and f8 to calculate other variables
    fx = f2 + f4 - f1 - f3 
    fy = f5 + f7 - f6 - f8
    n  = ( (f2 + f3) - (f1 + f4) + (f5 + f8) - (f6 + f7) ) * raio

    # Kinematics equations (only phi_dot is needed)

    #phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    #theta_dot = q * np.cos(phi) - r * np.sin(phi)
    psi_dot = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

    # Rotation equations (only r_dot is needed)

    #p_dot = ixz * (ixx - iyy + izz) * p * q - (izz * (izz - iyy) + ixz**2) * r * q + izz * l + ixz * n
    #q_dot = (izz - ixx) * p * r - ixz * (p**2 - r**2) + m
    #r_dot = (ixx * ((ixx - iyy) + (ixz**2)) + (ixz**2)) * p * q - ixz * (ixx - iyy + ixz) * q * r + ixz * l + ixx * n
    
    r_dot = (ixx / (ixx * izz - ixz**2)) * (n + (ixz / ixx)*l + ((ixz**2) / ixx) * p * q - (izz - iyy) * r * q - (iyy - ixx) * p * q - ixz * q * r)

    # Matrix of direct cosines
    mrc = np.array([[np.cos(theta) * np.cos(psi) ,   np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi) ,  np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
                    [np.cos(theta) * np.sin(psi) ,   np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi) ,  np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
                    [-np.sin(theta)              ,   np.sin(phi) * np.cos(theta)                                           ,  np.cos(phi) * np.cos(theta)]])


    cal = np.dot(mrc, np.array([[u, v, w]]).T)


    uf = cal[0]
    vf = cal[1]


    pn_dot = uf
    pe_dot = vf


    # Translation Equation
    u_dot = (fx/mass) - g * np.sin(theta) + v * r - w * q
    v_dot = (fy/mass) + g * np.sin(phi) * np.cos(theta) - u * r + w * p
    #w_dot = g * np.cos(phi) * np.cos(theta) + fz + u * q - v * p


    # Result of the derivative of X in a vector
    x_dot = np.array([psi_dot, r_dot, pn_dot.item(), pe_dot.item(), u_dot, v_dot])


    return x_dot



################################################################################
################################################################################


def matrix_distribution(un):

    r     =  1
    Mz    = un[0]
    Fx    = un[1]
    Fy    = un[2]
    

    A_fm = np.array([[-1, 1, -1,  1, 0,  0,  0,  0],
                    [ 0, 0,  0,  0, 1, -1,  1, -1],
                    [-r, r,  r, -r, r, -r, -r,  r]]
                    )    


    # Pseudo-inverso de A_fm
    A_fm_pinv = np.linalg.pinv(A_fm)

    b_fm = np.array([Fx, Fy, Mz])

    # calcular o resultado X
    X = np.dot(A_fm_pinv, b_fm)

    f1 = X[0]
    f2 = X[1]
    f3 = X[2]
    f4 = X[3]
    f5 = X[4]
    f6 = X[5]
    f7 = X[6]
    f8 = X[7]


    for f in X:
        if f < 0:
            f = 0


    return f1, f2, f3, f4, f5, f6, f7, f8

################################################################################
################################################################################


def PWN(un_real, periodo, temp):

    f1, f2, f3, f4, f5, f6, f7, f8 = un_real
    f_max = 10
    Dc_max = 100

    un_next_real = []

    for f in un_real:


        Dc = (f * Dc_max) / f_max
        # Calculate the time the signal is ON
        T_on = (Dc * periodo) * 100
        T_off = periodo - T_on

        if temp < T_on:
            f = f_max
            un_next_real.append(f)
        else:
            f = 0
            un_next_real.append(f)

    return un_next_real



def RK(x0, x_eq, u_eq, K):
    t0 = 0
    tf = 30
    tf = 60
    h  = 0.01

    tn = t0
    xn = x0
    un = -np.dot(K, (xn - x_eq)) + u_eq

    un_real = matrix_distribution(un)

    lista_tempo     = []
    lista_x         = []
    lista_u_control = []

    k = 1
    temp = t0

    for t in np.arange(t0, tf, h):
        lista_tempo.append(tn)
        lista_x.append(xn)
        lista_u_control.append(un_real)

        k1 = h * np.array(f_real(tn, xn, un_real))
        k2 = h * np.array(f_real(tn + (h / 4), xn + (k1 / 4), un_real))
        k3 = h * np.array(f_real(tn + (h / 4), xn + (k1 / 8) + (k2 / 8), un_real))
        k4 = h * np.array(f_real(tn + (h / 2), xn - (k2 / 2) + k3, un_real))
        k5 = h * np.array(f_real(tn + (3 * h / 4), xn + (3 * k1 / 16) + (9 * k4 / 16), un_real))
        k6 = h * np.array(f_real(tn + h, xn - (3 * k1 / 7) + (2 * k2 / 7) + (12 * k3 / 7) - (12 * k4 / 7) + (8 * k5 / 7), un_real))

        xn_next = xn + (1 / 90) * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6)

        un_next      = -np.dot(K, (xn_next - x_eq)) + u_eq
        un_next_real = matrix_distribution(un_next)

        periodo = 1/100
        if temp == periodo * k:

            un_next_real = PWN(un_next_real, periodo, temp)
            k = k + 1
        
        else: 
            un_next_real = matrix_distribution(un_next)

        temp_next    = temp + h
        temp         = temp_next
        tn_next = tn + h

        tn      = tn_next
        xn      = xn_next
        un_real = un_next_real

    return np.array(lista_tempo), np.array(lista_x), np.array(lista_u_control)

t, x, u = RK(x0, x_eq, u_eq, K)


################################################################################
################################################################################

def fazer_grafico(x0, u_control, x_eq, u_eq):
    global t, x, u

    # variaveis de controlo
    n_data  = u[:, 0]
    fx_data = u[:, 1]
    fy_data = u[:, 2]

    #variaveis de estado
    psi_data = x[:, 0]
    r_data   = x[:, 1]
    pn_data  = x[:, 2]
    pe_data  = x[:, 3]
    u_data   = x[:, 4]
    v_data   = x[:, 5]


    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, fx_data, 'r')
    plt.title('Força em X')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Força (N)')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, fy_data, 'g')
    plt.title('Força em Y')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Força (N)')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, n_data, 'b')
    plt.title('Rotação no eixo do Z')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Rotação no eixo do Z (rad/s)')
    plt.grid(True)

    plt.tight_layout()

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, psi_data, 'r')
    plt.title('Rotação no eixo do Z')
    plt.xlabel('Tempo (s)')
    plt.ylabel('\Psi (rad)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, r_data, 'g')
    plt.title('Velocidade Angular no eixo do Z')
    plt.xlabel('Tempo (s)')
    plt.ylabel('r (rad/s)')
    plt.grid(True)

    plt.tight_layout()

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, pn_data, 'b')
    plt.title('Posição no eixo do X')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição no eixo do X (m)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, pe_data, 'r')
    plt.title('Posição no eixo do Y')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição no eixo do Y (m)')
    plt.grid(True)

    plt.tight_layout()

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, u_data, 'g')
    plt.title('Velocidade no eixo do X')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocidade no eixo do X (m/s)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, v_data, 'b')
    plt.title('Velocidade no eixo do Y')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocidade no eixo do Y (m/s)')
    plt.grid(True)

    plt.figure(figsize=(12, 8))

    plt.plot(pe_data, pn_data)
    plt.title('Posição no eixo do X e Y')
    plt.grid(True)
    plt.xlim([-2, 2])
    plt.ylim([-3, 3])

    plt.tight_layout()
    plt.show()


fazer_grafico(x0, u_control, x_eq, u_eq) # 4º a ser executado




def simulation(WIDTH, HEIGHT):
    global WIN
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Project III")

# Platform class
class Platform:
    def __init__(self, x, y, size, color):
        self.x = x
        self.y = y
        self.size = size
        self.color = color

    def draw_circle(self, win):
        # Draw the circle
        pygame.draw.circle(win, self.color, (self.x, self.y), self.size)
        
        # Draw the cross in the centre of the circle
        cross_color = (255, 255, 255)  # White colour for the cross
        line_length = self.size
        # Horizontal line
        pygame.draw.line(win, cross_color, (self.x - line_length, self.y), (self.x + line_length, self.y), 2)
        # Vertical line
        pygame.draw.line(win, cross_color, (self.x, self.y - line_length), (self.x, self.y + line_length), 2)

# Main function
def main(WIDTH, HEIGHT, WHITE, GREEN):
    global x
    pygame.init()  # Initialise Pygame
    clock = pygame.time.Clock()
    simulation(WIDTH, HEIGHT)

    platform_size = 20
    platform_color = GREEN

    font = pygame.font.Font(None, 24)  # Source for the text

    platform = Platform(WIDTH // 2, HEIGHT // 2, platform_size, platform_color)  # Start the platform in the centre

    run = True
    start_time = pygame.time.get_ticks() / 1000  # Time in seconds
    while run:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Calculates the time elapsed since the start
        elapsed_time = pygame.time.get_ticks() / 1000 - start_time

        # Update the physics and get the new state of the platform
        current_index = min(int(elapsed_time * 100), len(x) - 1)  # Controls animation speed
        psi, _, pn, pe, _, _ = x[current_index] #synchronise the progress of the animation with the time elapsed in the simulator

        # Calculates the position of the platform on the screen
        platform.x = int(WIDTH / 2) + int(pn * 10)
        platform.y = int(HEIGHT / 2) - int(pe * 10)

        # Clean the screen
        WIN.fill((0, 0, 0))

        # Design the platform
        platform.draw_circle(WIN)

        # Displays the platform coordinates on the screen
        text_surface = font.render(f"Posição (X, Y): ({pn:.2f}, {pe:.2f})", True, WHITE)
        WIN.blit(text_surface, (10, 10))  # Position of the text on the screen

        # Refresh the screen
        pygame.display.flip()  # Faster alternative to pygame.display.update()

        # Adjust the FPS to control the screen's refresh rate
        clock.tick(120)  # Approximately 120 FPS

    pygame.quit()

main(WIDTH, HEIGHT, WHITE, GREEN)


################################################################################
################################################################################


print("Programme over!")