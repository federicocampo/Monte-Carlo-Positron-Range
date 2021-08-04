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
    '''
    Data l'energia w del delta creato e l'energia E_kin, da cui ricavo beta della
    particella incidente, ricavo l'angolo di emissione del delta: phi
    '''
    Energy = E_kin + m_electrc2
    beta = np.sqrt(Energy**2 - m_positrc2**2)/Energy

    phi = np.arccos(np.sqrt(w/(2*m_electrc2*beta**2)))

    return phi


def Phipositr(w, e_kin, phidelta):
    '''
    Calcola l'angolo di emissione del positrone dopo l'urto che ha creato un raggio
    delta, imponendo la conservazione dell'impulso,  dato e_kin = energia cinetica
    positrone incidente, w = energia cinetica del delta, phidelta = angolo di emissione del 
    delta, rispetto alla direzione di incidenza del positrone
    '''
    phidelta = np.abs(phidelta)
    energy = e_kin + m_electrc2
    energy_prim = e_kin - w + m_electrc2
    energy2 = w + m_electrc2
    p1 = np.sqrt(energy**2 - m_electrc2**2)
    p1_prim = np.sqrt(energy_prim**2 - m_electrc2**2)
    p2_prim = np.sqrt(energy2**2 - m_electrc2**2)

    cosphi1 = -p2_prim*np.cos(phidelta)/p1_prim
    #print('Cos(phi positrone) = ', cosphi1)
    phipositr = np.arccos(cosphi1)
    return phipositr



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
    
def Delta_position(vett_inizio, vett_fine):
     '''
     Estrae a caso un valore per stabilire il punto di creazione del raggio delta, 
     tra il punto di inizio e il punto di fine dello step fatto dal positrone, 
     scegliendo a caso un valore di x compreso tra i due valori di ascissa, 
     e la corrispondente ordinata data dall'equazione della retta passante per quei
     due punti (inizio e fine step)
     '''
     x_inizio = vett_inizio[0]
     y_inizio = vett_inizio[1]
     x_fine = vett_fine[0]
     y_fine = vett_fine[1]

     x_delta = np.random.uniform(x_inizio, x_fine)
     y_delta = y_inizio + (y_fine - y_inizio)*(x_delta - x_inizio)/(x_fine - x_inizio)

     vett_delta = np.array([x_delta, y_delta])
     return vett_delta
     

    
#if __name__ == “main”: 

# Ekin = energia cinetica della particella (elettrone o positrone) primaria

#seed = time.time()
seed = 42
np.random.seed(int(seed))

    
E_0 = 0.1 #Mev, energia iniziale positrone creato
Estep = 3e-3 # MeV - Energia persa ad ogni step


Tot_Npositr = 50
for npart in range(Tot_Npositr):

    delta_parameters = np.zeros((10, 5))
    i_delta = 0

    first_iteration = True
    #Creo array dove tener conto della posizione della particella punto per punto, per ciascuna particella
    X = []
    Y = []
    #Ekin iniziallizzo l'energia CINETICA (Ekin) della particella con E_0
    Ekin = E_0
    
    while(Ekin > 0):

        step = Step(Estep, Ekin)

        if first_iteration:
            posiz = np.array([0, 0])
            X.append(posiz[0])
            Y.append(posiz[1])
            theta_prim = np.random.uniform(0, 2*np.pi)
            theta0 = 0

            first_iteration = False
            delta = False
        elif delta:
            delta = False  
        else:
            theta_prim = np.random.normal(scale = 0.4)           

        x1_prim, y1_prim = step*np.cos(theta_prim), step*np.sin(theta_prim)
        vett_prim = np.array([x1_prim, y1_prim])
        vett = Rotation(theta0, vett_prim) + posiz

        #Viene creato in questo tratto di strada? y/n
        '''Calcolo quanti raggi delta vengono creati, tramite la funzione
        Ndelta. Dato che per ciascuno step, il numero di raggi delta creati, 
        ricavato con la formula, è < 1 (typ: 0.018), per stabilire st un delta 
        viene creato o no, estraggo un numero uniforme tra 0 e 1, e se il numero
        è minore del numero ricavato prima, vuol dire che il raggio delta è 
        stato creato. 
        '''
        ndelta = Ndelta(Ekin, step)
        #mettere un controllo che ndelta sia < 1 ?
        yndelta = np.random.uniform(0, 1)
        
        if yndelta < ndelta: #se viene creato il delta..
            print('Delta!!')

            # 1: In che punto viene creato il delta.
            delta_position = Delta_position(posiz, vett)
            # 2: Con che energia CINETICA (Ekin_delta) viene creato il delta.
            tau = Ekin/m_electrc2
            W_max = 2*tau*(tau +2)*m_electrc2/(2+ 2*(tau+1))
            flag = False
            while(flag is False):
                '''
                estraggo a caso una energia del delta Ekin_delta, usando
                rigetto elementare. uso una flag per stabilire quando
                ho trovato un valore valido della distribuzione delle 
                energie, cioè un valore valido dell'energia del delta
                '''
                Ekin_delta = np.random.uniform(W_min, W_max)
                p = Sampling_Wdelta(Ekin_delta, Ekin)
                xi2 = np.random.uniform(0, Sampling_Wdelta(W_min, Ekin))
                if xi2 < p:
                    flag = True
            
            # 3: Calcolo l'angolo di emissione del delta
            phidelta = Phidelta(Ekin_delta, Ekin)
            '''
            dato che l'angolo del delta può essere sopra o sotto la direzione di incidenza, 
            estraggo un numero 0 o 1, per stabilire se il delta va a + 0 - phidelta
            '''
            temp = np.random.randint(2)
            if temp == 0:
                phidelta = -phidelta
            # 4: Calcolo il corrispondente angolo di emissione del positrone
            phipositr = Phipositr(Ekin_delta, Ekin, phidelta)

            # Inizia il pezzo in cui stabilisco la traiettoria del positrone dopo l'urto col delta
            posiz = delta_position
            X.append(posiz[0])
            Y.append(posiz[1])
            Ekin -= Ekin_delta
            theta0 += theta_prim
            theta_prim = phipositr
            delta = True
            # E ora salvo i parametri del delta per utilizzarli dopo
            delta_parameters[i_delta][0] = Ekin_delta
            delta_parameters[i_delta][1] = delta_position[0]
            delta_parameters[i_delta][2] = delta_position[1]
            delta_parameters[i_delta][3] = theta0
            delta_parameters[i_delta][4] = phidelta
            print('Sto salvando i parametri e vengono:')
            print(delta_parameters[i_delta][:])
            i_delta += 1
            


        else:
            posiz = vett            
            X.append(posiz[0])
            Y.append(posiz[1])
            Ekin -= Estep
            theta0 += theta_prim
         

    plt.plot(X, Y, color = 'tab:blue')
    if delta_parameters.any() == 0:
    else:
        Tot_delta = i_delta
        for ndelta in range(Tot_delta):
            first_iteration = True
            #Creo array dove tener conto della posizione della particella punto per punto, per ciascuna particella
            X = []
            Y = []
            #Ekin iniziallizzo l'energia CINETICA (Ekin) della particella con E_0
            Ekin = delta_parameters[ndelta][0]
            while(Ekin > 0):

                step = Step(Estep, Ekin)

                if first_iteration:

                    posiz = np.array([delta_parameters[ndelta][1], delta_parameters[ndelta][2]])
                    X.append(posiz[0])
                    Y.append(posiz[1])
                    theta_prim = delta_parameters[ndelta][4]
                    theta0 = delta_parameters[ndelta][3]

                    first_iteration = False 
                else:
                    theta_prim = np.random.normal(scale = 0.4)           

                x1_prim, y1_prim = step*np.cos(theta_prim), step*np.sin(theta_prim)
                vett_prim = np.array([x1_prim, y1_prim])
                vett = Rotation(theta0, vett_prim) + posiz

                posiz = vett            
                X.append(posiz[0])
                Y.append(posiz[1])
                Ekin -= Estep
                theta0 += theta_prim
        plt.plot(X, Y, color = 'tab:red')

    
    
        
plt.show()
