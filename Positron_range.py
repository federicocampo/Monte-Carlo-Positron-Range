import numpy as np
from matplotlib import pyplot as plt
import time
# Energy in MeV, dx in cm

m_positrc2 = 0.511  #MeV
m_electrc2 = m_positrc2
M_protonc2 = 938 #MeV
r_e = 2.82e-13  #cm - raggio classico elettone
N_av = 6.022e23 #mol^-1 - Numero di avogadro
density = 1  #g/cm3 - densità acqua



Z = 7.42 # Z efficace H2O
A = 18


N_e = density * N_av *Z/A


I = (12*Z + 7)*1e-6 #MeV

# E_kinetic typ da poter usare nelle prove: 0.8 MeV = 800 keV

#Robe utili per il calcolo dei raggi delta:
W_min = 0.01 #[MeV] = 10 keV energia minima del delta, affinchè venga prodotto


def dedx_coll(E_kinetic): #E_kinetic in MeV

    Energy = E_kinetic + m_positrc2
    beta = np.sqrt(Energy**2 - m_positrc2**2)/Energy
    
    coll1 = 2*np.pi * r_e**2 * m_electrc2 /beta**2 * N_av * density*Z/A
    coll2 = np.log(m_electrc2*beta**2 *E_kinetic/(2*I**2 *(1-beta**2)) ) - np.log(2) * (2*np.sqrt(1-beta**2)-1+beta**2) + (1-beta**2) + 1/8 * (1-np.sqrt(1-beta**2))**2
    
    coll = coll1 * coll2
    return coll



def Step(Estep, E_k):
    r = Estep/dedx_coll(E_k)
    return r



def Ndelta(E_kin, step):
    '''
    Prende in input l'energia cinetica della particella incidente (E_kin), 
    e la lunghezza dello step, spazio percorso, e restituisce il numero di 
    elettroni delta prodotti in quel tragitto "step"
    '''
    W_min = 0.01 #[MeV] : 10keV come energia minima di produzione

    if E_kin > W_min:
        Energy = E_kin + m_electrc2
        beta = np.sqrt(Energy**2 - m_positrc2**2)/Energy
        
        A = N_e * 2*np.pi* r_e**2 * m_electrc2 #primo pezzo, senza il beta al denom
        tau = E_kin/m_electrc2
        W_max = 2*tau*(tau +2)*m_electrc2/(2+ 2*(tau+1))
        G = (W_max-W_min)/(Energy)**2
        B = (1/W_min - 1/W_max) - beta**2 *np.log(W_max/W_min)/W_max + G
        dndx = A/beta**2 * B
        N_delta = dndx * step
    else: N_delta = 0
    
    return N_delta


def Phidelta(w, E_kin):
    '''Data l'energia w del delta creato e l'energia E_kin, da cui ricavo beta della
    particella incidente, ricavo l'angolo di emissione del delta: phi
    '''
    Energy = E_kin + m_electrc2
    beta = np.sqrt(Energy**2 - m_positrc2**2)/Energy

    phi = np.arccos(np.sqrt(w/(2*m_electrc2*beta**2)))

    return phi







X = [] #X e Y array che terranno traccia delle posizioni del positr step per step
Y = []
    
x = 0   #Posizione iniziale della particella creata
y = 0
    
E_0 = 0.1 #Mev, energia iniziale positrone creato
Estep = 3e-3 # MeV - Energia persa ad ogni step




def Rotation(theta, vect_prim):
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    #vect_prim = np.array([[x_prim], [y_prim]])
    vect = np.dot(rot_mat, vect_prim)
    vect = np.ravel(vect)
    return vect
    
def Traslation(vect_prim, vect_o_prim):
    vect_trasled = vect_prim + vect_o_prim
    return vect_trasled 
    
 
def Sampling_Wdelta(W, E_kin):
    W_min = 0.01 #[MeV] energia necessaria per creare un delta, limite inferiore dell'intervallo
    tau = E_kin/m_electrc2

    W_max = 2*tau*(tau +2)*m_electrc2/(2+ 2*(tau+1))

    p_w = 1/W**2 - 1/(W*W_max)
    
    return p_w
    
 

    
#if __name__ == “main”: 

# E = energia cinetica della particella (elettrone o positrone) primaria
seed = time.time()
np.random.seed(int(seed))

Tot_Npositr = 10
for npart in range(Tot_Npositr):
    
    X = []
    Y = []
    
    posiz = np.array([0, 0])
    X.append(posiz[0])
    Y.append(posiz[1])
    
    E = E_0
    theta0 = np.random.uniform(0, 2*np.pi)

    step = Step(Estep, E)
    
    x1, y1 = step*np.cos(theta0), step*np.sin(theta0)
    posiz = np.array([x1, y1])
    X.append(posiz[0])
    Y.append(posiz[1])
    
    E -= Estep
    
    while(E > 0):
        
        theta_prim = np.random.normal(scale = 0.4)
        
        step = Step(Estep, E)
        x2_prim, y2_prim = step*np.cos(theta_prim), step*np.sin(theta_prim)
        vett_prim = np.array([x2_prim, y2_prim])
        
        vett = Rotation(theta0, vett_prim) + posiz
        E-=Estep
        theta0 += theta_prim
        posiz = vett 
        
        #Creazione di un raggio delta: viene creato in questo tratto di strada? y/n
        '''Calcolo quanti raggi delta vengono creati, tramite la funzione
        Ndelta. Dato che per ciascuno step, il numero di raggi delta creati, 
        ricavato con la formula, è < 1 (typ: 0.018), per stabilire st un delta 
        viene creato o no, estraggo un numero uniforme tra 0 e 1, e se il numero
        è minore del numero ricavato prima, vuol dire che il raggio delta è 
        stato creato. Cioè uso Ndelta come una sorta di probabilità
        '''
         
        ndelta = Ndelta(E, step)
        #mettere un controllo che ndelta sia < 1 ?
        yndelta = np.random.uniform(0, 1)
        
        if yndelta < ndelta: #se viene creato il delta..
            print('Delta!!')
            #con che energia viene creato il delta?
            tau = E/m_electrc2
            W_max = 2*tau*(tau +2)*m_electrc2/(2+ 2*(tau+1))
            
            flag = False
            while(flag is False):
                '''
                estraggo a caso una energia del delta en_delta, usando
                rigetto elementare. uso una flag per stabilire quando
                ho trovato un valore valido della distribuzione delle 
                energie, cioè un valore valido dell'energia del delta
                '''
                en_delta = np.random.uniform(W_min, W_max)
                p = Sampling_Wdelta(en_delta, E)
                xi2 = np.random.uniform(0, Sampling_Wdelta(W_min, E))
                if xi2 < p:
                    print(en_delta)
                    flag = True
            
            #Calcolo l'angolo di emissione del delta
            phidelta = Phidelta(en_delta, E)
            
            '''
            dato che l'angolo del delta può essere sopra o sotto la direzione di incidenza, 
            estraggo un numero 0 o 1, per stabilire se il delta va a + 0 - phidelta
            '''
            temp = np.random.randint(2)
            if temp == 0:
                phidelta = -phidelta
            print('Phi delta = ', phidelta*180/np.pi)


            
            
            

         
        
        
        X.append(posiz[0])
        Y.append(posiz[1])
    
    
    plt.plot(X, Y)
        
    
   
   
   
   













plt.show()
