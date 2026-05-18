import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import sys

# Numero di caselle presenti nell'ambiente chiamati STATI
NUMERO_STATI = 10

# Avendo 10 caselle, ho 10 azioni possibili
NUMERO_AZIONI = NUMERO_STATI

# Punto da raggiungere per ottenere reward = 1
PUNTO_REWARD = 9

# Punti che fanno terminare male l'episodio: reward = -1
PUNTI_CIECHI = [2, 6, 8]

# Punti da cui puo iniziare una partita
PUNTI_INIZIALI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Punti validi per la partenza casuale del percorso finale:
# esclude la reward e i punti ciechi.
PUNTI_INIZIALI_CASUALI = [
    stato for stato in PUNTI_INIZIALI
    if stato != PUNTO_REWARD and stato not in PUNTI_CIECHI
]

# Costanti per realizzare il Q-Learning
EPISODI = 5000                    # cicli = 1 percorso
PASSI_MASSIMI = 30                # ripetizioni del ciclo con quello stato iniziale
LEARNING_RATE = 0.8               # peso della nuova informazione rispetto a quella vecchia
DISCOUNT_FACTOR = 0.9             # peso delle ricompense future rispetto a quelle immediate
EPSILON_INIZIALE = 1.0            # probabilità iniziale di esplorare anziché sfruttare l'ambiente
EPSILON_MINIMO = 0.05             # impedisce di non esplorare
RIDUZIONE_EPSILON = 0.995         # permette di passare gradualmente da esplorazione a sfruttamento

"""
    Percorso da realizzare:

           2
           ^
           |
    0 <--> 1 <--> 4 <--> 6
    |             | \
    v             v   \
    3 <---------> 5 -> 7 -> 9
                       |
                       v
                       8
"""

# Dizionario dei collegamenti dell'ambiente
# Ogni chiave e uno stato, ogni lista contiene gli stati raggiungibili
COLLEGAMENTI_AMBIENTE = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1],
    3: [0, 5],
    4: [1, 5, 6, 7],
    5: [3, 4, 7],
    6: [4],
    7: [4, 5, 8, 9],
    8: [7],
    9: [],
}

"""
    Questa classe crea un ambiente personalizzato grazie a Gymnasium

    Essa è la sottoclasse di gym.Env, quindi implementa i metodi richiesti da Gymnasium:
    - __init__: costruttore.
    - reset: resetta l'ambiente.
    - step: esegue un'azione.
"""

class AmbientePercorso(gym.Env): 

    metadata = {"render_modes": ["human"]} 

    # self.observation_space --> spazio degli stati (contiene i numeri da 0 a NUMERO_STATI - 1)
    # self.action_space --> spazio delle azioni (contiene i numeri da 0 a NUMERO_AZIONI - 1)
    # self.stato_corrente --> posizione attuale dell'agente (inizializzato a None, sarà impostato da reset())

    def __init__(self):
        self.observation_space = spaces.Discrete(NUMERO_STATI) 
        self.action_space = spaces.Discrete(NUMERO_AZIONI)     
        self.stato_corrente = None            

    # Spaces è la libreria importata
    # Discrete è una classe che ha come attributo n, utile per usare sample()
    # Il metodo sample() restituisce un numero intero casuale tra 0 e n-1

    def reset(self, seed=None, options=None):
        # Uso il metodo reset della superclasse
        super().reset(seed=seed)

        self.stato_corrente = int(np.random.choice(PUNTI_INIZIALI))

        # Gymnasium richiede di restituire osservazione (stato iniziale) e info (per dare informazioni extra)
        return self.stato_corrente, {}

    def step(self, azione):
        # Converto l'azione in intero per evitare problemi con tipi NumPy
        azione = int(azione)

        # Controlla se azione è tra i valori disponibili di COLLEGAMENTI_AMBIENTE nello stato corrente
        azione_valida = azione in COLLEGAMENTI_AMBIENTE[self.stato_corrente]

        # Se l'azione non e valida, l'agente resta fermo e prende reward -1
        if not azione_valida:
            prossimo_stato = self.stato_corrente
            reward = -1
            terminato = False

        # Se l'azione porta in un punto cieco, l'episodio termina male
        elif azione in PUNTI_CIECHI:
            prossimo_stato = azione
            reward = -1
            terminato = True

        # Se l'azione porta alla reward, l'episodio termina bene
        elif azione == PUNTO_REWARD:
            prossimo_stato = azione
            reward = 1
            terminato = True

        # Se l'azione è valida ma non terminale, il movimento vale 0
        else:
            prossimo_stato = azione
            reward = 0
            terminato = False

        # Aggiorno la posizione attuale dell'agente
        self.stato_corrente = prossimo_stato

        # Non tronco l'episodio dentro l'ambiente: uso PASSI_MASSIMI fuori
        troncato = False

        # Gymnasium richiede: prossimo stato, reward, terminato, troncato, info
        return prossimo_stato, reward, terminato, troncato, {}

# -------------------------------- fine della classe -----------------------------------

def crea_tabella_reward():      
    #full crea una matrice di dimensione NUMERO_STATI x NUMERO_AZIONI piena di -1
    tabella_reward = np.full((NUMERO_STATI, NUMERO_AZIONI), -1, dtype=int)

    # COLLEGAMENTI_AMBIENTE è un dizionario quindi in stato ci finisce la chiave 
    # mentre in stati_raggiungibili invece ci sono i valori
    for stato, stati_raggiungibili in COLLEGAMENTI_AMBIENTE.items():
        for prossimo_stato in stati_raggiungibili:
            if prossimo_stato == PUNTO_REWARD:
                tabella_reward[stato, prossimo_stato] = 1 
            elif prossimo_stato not in PUNTI_CIECHI:
                tabella_reward[stato, prossimo_stato] = 0   
    # I punti ciechi restano -1, quindi non serve riscriverli
    return tabella_reward

def addestra_agente(ambiente):
    # zeros crea una matrice di dimensione NUMERO_STATI x NUMERO_AZIONI piena di 0
    q_table = np.zeros((NUMERO_STATI, NUMERO_AZIONI), dtype=float)

    # Ripeto l'addestramento per molti episodi
    for episodio in range(EPISODI):
        # Ogni episodio parte da un punto casuale
        stato, _ = ambiente.reset()

        # Epsilon parte alto e poi scende: prima esploro, poi sfrutto
        epsilon = max(EPSILON_MINIMO, EPSILON_INIZIALE * (RIDUZIONE_EPSILON ** episodio))

        # Limito il numero di passi per evitare episodi infiniti
        for passo in range(PASSI_MASSIMI):
            # Con probabilita epsilon scelgo un'azione casuale
            if np.random.random() < epsilon:
                azione = ambiente.action_space.sample()

            # Altrimenti scelgo l'azione migliore secondo la Q-table
            else:
                azione = int(np.argmax(q_table[stato]))

            # Eseguo l'azione nell'ambiente
            prossimo_stato, reward, terminato, troncato, _ = ambiente.step(azione)

            # Se l'episodio finisce, non considero reward future
            if terminato:
                migliore_valore_futuro = 0

            # Se l'episodio continua, guardo il miglior valore Q del prossimo stato
            else:
                migliore_valore_futuro = np.max(q_table[prossimo_stato])

            # Applico la formula del Q-Learning:
            # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
            q_table[stato, azione] = q_table[stato, azione] + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * migliore_valore_futuro - q_table[stato, azione]
            )

            stato = prossimo_stato

            # Se l'agente ha raggiunto reward o vicolo cieco, termino l'episodio
            if terminato or troncato:
                break

    return q_table

def scegli_stato_iniziale_percorso():
    nome_programma = os.path.splitext(os.path.basename(sys.argv[0]))[0].lower()

    if nome_programma == "partenza_da_9":
        return PUNTO_REWARD

    if nome_programma == "partenza_da_2":
        return 2

    if nome_programma == "partenzacasuale":
        return int(np.random.choice(PUNTI_INIZIALI_CASUALI))

    return None

def trova_percorso(ambiente, q_table, stato_iniziale=None):
    # Faccio partire il percorso finale da un punto scelto o casuale
    if stato_iniziale is None:
        stato, _ = ambiente.reset()
    else:
        stato = int(stato_iniziale)
        ambiente.stato_corrente = stato

    # Creo la lista che conterra tutti i punti attraversati
    percorso = [stato]

    if stato == PUNTO_REWARD:
        return percorso

    # Trova l'azione migliore secondo la Q-table
    for passo in range(PASSI_MASSIMI):
        # argmax restituisce l'indice della colonna con il valore massimo, quindi l'azione migliore
        azione = int(np.argmax(q_table[stato]))

        # Eseguo l'azione nell'ambiente per controllare se è valida
        prossimo_stato, reward, terminato, troncato, _ = ambiente.step(azione)

        # Aggiungo il nuovo punto al percorso
        percorso.append(prossimo_stato)

        # Aggiorno lo stato corrente
        stato = prossimo_stato

        # Se arrivo alla reward o in un vicolo cieco, il percorso termina
        if terminato or troncato:
            break

    # Restituisco il percorso trovato
    return percorso

#--------------------------- funzioni per stampare i risultati in modo chiaro --------------------

def stampa_mappa():
    # Stampo i collegamenti dell'ambiente
    print("\nMAPPA DELL'AMBIENTE")
    print("Ogni riga indica i punti raggiungibili con una singola azione.")

    # Scorro tutti gli stati
    for stato in range(NUMERO_STATI):
        # Se lo stato e la reward, lo scrivo chiaramente
        if stato == PUNTO_REWARD:
            print(f"{stato}: REWARD finale")

        # Se lo stato e un punto cieco, lo scrivo chiaramente
        elif stato in PUNTI_CIECHI:
            print(f"{stato}: vicolo cieco")

        # Altrimenti mostro i collegamenti disponibili
        else:
            print(f"{stato}: {COLLEGAMENTI_AMBIENTE[stato]}")

def stampa_tabella(nome, tabella, decimali):
    # Stampo il nome della tabella
    print(f"\n{nome}")

    # Creo e stampo l'intestazione con i numeri delle azioni
    intestazione = "stato | " + " ".join(f" {azione:>5}" for azione in range(NUMERO_AZIONI))
    print(intestazione)

    # Stampo una linea divisoria
    print("-" * len(intestazione))

    # Scorro tutte le righe della tabella
    for stato in range(NUMERO_STATI):
        # Formatto i valori con il numero di decimali richiesto
        valori = " ".join(f"{valore:>6.{decimali}f}" for valore in tabella[stato])

        # Stampo la riga corrispondente allo stato
        print(f"{stato:>5} | {valori}")

def stampa_percorso(percorso):
    # Converto il percorso in una stringa con frecce
    percorso_con_frecce = " -> ".join(str(stato) for stato in percorso)

    # Stampo il percorso migliore trovato
    print("\nPERCORSO MIGLIORE TROVATO")
    print(percorso_con_frecce)

    # Se l'ultimo punto e la reward, il programma ha raggiunto l'obiettivo
    if percorso[-1] == PUNTO_REWARD:
        print("Risultato: reward raggiunta, programma terminato.")

    # Se l'ultimo punto e un vicolo cieco, il percorso finale e fallito
    elif percorso[-1] in PUNTI_CIECHI:
        print("Risultato: vicolo cieco raggiunto.")

    # Se non e arrivato a un punto terminale, ha raggiunto il limite dei passi
    else:
        print("Risultato: limite passi raggiunto.")

def attendi_tasto_se_exe():
    if getattr(sys, "frozen", False) and os.environ.get("SALTA_PAUSA_EXE") != "1":
        print("\nPremi un tasto per chiudere...", end="", flush=True)
        try:
            import msvcrt
            msvcrt.getch()
        except ImportError:
            input()

def main():
    ambiente = AmbientePercorso()                                    # Creo l'ambiente personalizzato Gymnasium.
    reward_table = crea_tabella_reward()                             # Creo la tabella delle reward dell'ambiente.
    q_table = addestra_agente(ambiente)                              # Addestro l'agente e ottengo la Q-table finale.
    stato_iniziale = scegli_stato_iniziale_percorso()                # Scelgo la partenza in base al nome del programma.
    percorso = trova_percorso(ambiente, q_table, stato_iniziale)     # Trovo il percorso migliore.

    stampa_mappa()                                                   # Stampo la mappa dell'ambiente.
    stampa_tabella("TABELLA REWARD DELL'AMBIENTE", reward_table, 0)  # Stampo la tabella delle reward.
    stampa_tabella("Q-TABLE IMPARATA CON Q-LEARNING", q_table, 2)    # Stampo la Q-table imparata con Q-Learning.
    stampa_percorso(percorso)                                        # Stampo il percorso finale con frecce.
    attendi_tasto_se_exe()

if __name__ == "__main__":
    main()
