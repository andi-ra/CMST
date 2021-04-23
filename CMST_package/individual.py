import random
import sys


class Individual:
    """
        Questa classe rappresenta l'individuo cioè la soluzione (che nel mio caso sarà uno spanning tree).

        Ogni soluzione ha dentro di sè la sua fitness che io posso impostare nel file principale o nella funzione
        fitness_evaluation(). Ho cercato di scrivere l'oggetto Individual nel modo più "pythonico" possibile per
        facilitare le operazioni nei file population_generator e hall_of_fame.

        _tree : list
            lista di archi ordinata che implementa l'albero ricoprente
        _fitness : int
            valore che rappresenta la bontà della soluzione
        optim_type : int
            indicatore del tipo di ottimizzazione che faccio (max o min)
        genotype : list[int]
            stringa di interi di lunghezza variabile
    """


    def __init__(self, size, min, iter_range):
        """
        Inizializzo l'individuo, per ora è soltanto una struttra dati vuota con dei

        :param size: Dimensione dell'individuo cioè quanti archi avrà
        :param min: Sto minimizzando/massimizzando
        :param iter_range: Massimo numero che posso assegnare all'inizializzazione
        """
        if min:
            self.optim_type = 1
        else:
            self.optim_type = -1
        self._fitness = 0
        self.genotype = [random.randint(0, iter_range) for i in range(size)]
        self.tree = []

    def __iter__(self):
        """
        Restituisci te stesso, cioè la soluzione come se fossi un oggetto iterabile

        :return:
        """
        return iter(self.genotype)

    @property
    def fitness(self):
        """
        Metodo getter per comodità quando uso l'IDE è bene avercelo

        :return: **_fitness** questo è un valore "privato"
        """
        return self._fitness * self.optim_type

    @fitness.setter
    def fitness(self, value):
        """
        Metodo setter per comodità quando uso l'IDE è bene avercelo

        :param value:
        :return:
        """
        self._fitness = value * self.optim_type

    # Overide delle funzioni di get/set degli oggetti
    def __getitem__(self, index):
        """
        Quando faccio l'accesso alla struttura Individual[index] restituisce una tupla

        :param index:
        :return:
        """
        return self.genotype[index]

    def __setitem__(self, index, value):
        """
        Quando voglio settare un arco (implementato singolarmente come tupla) uso questo

        :param index:
        :param value:
        :return:
        """
        self.genotype[index] = value

    def __eq__(self, other):
        """
        Metodo che mi implementa il confronto, come fai a confrontare due individui, cioè due alberi?
        Ho dovuto riscrivere questa operazione

        :param other:
        :return:
        """
        temp = 0
        for i in range(len(self)):
            temp += 1 if self[i] == other[i] else 0
        return temp == len(self)

    def __cmp__(self, other):
        """
        Come fai a confrontare due alberi? Su cosa basi il confronto? Uso la fitness dell'individuo
        (calcolata come il peso cumulativo del branch o albero). In sostanza è un confronto fra interi
        in cui "collasso"/riassumo tutta la soluzione in un unico parametro concentrato.

        :param other:
        :return:
        """
        return cmp(int(self._fitness), int(other._fitness))

    def __len__(self):
        """
        Poiché genotype è l'iterabile che funziona da "handle" per arrivare alla struttura vera e propria
        dell'albero che in pratica è l'individuo, per calcolarmi la lunghezza dell'individuo uso la lunghezza
        dell'albero a sua volta calcolata come lunghezza dell'iterabile (è più semplice come implementazione)

        :return:
        """
        return len(self.genotype)

    def __str__(self):
        """
        Questa funzione mi permette di visualizzare la soluzione quando faccio print(individuo).

        :return:
        """
        ret = ''
        for i in self.genotype:
            ret += str(i) + " "
        return ret

    def __lt__(self, other):
        """
        Operatore Less_Than, anche questo lo uso per i confronti

        :param other:
        :return:
        """
        return int(self._fitness) < int(other._fitness)

    def reshape2matrix(self, NUM_NODES=4):
        """
        Funzione che implementa la conversione da genotipo a matrice di adiacenza in un grafo. È in certi versi simile a
        ``numpy.reshape(shape)`` ma non sto lavorando con array o matrici numpy dunque mi devo reimplementare la mia versione
        di matrix_reshaping per "riarrotolare" il genotipo in una matrice di adiacenza che rappresenta un grafo. Ciò che
        ottengo in uscita è una matrice quadrata, simmetrica e con la diagonale da ignorare (è un +inf)::

            import individual
            ind = individual.Individual(4, 1, 256)
            ind.genotype
            Out[11]: [193, 87, 43, 27]
            ind.reshape2matrix(3)
            Out[16]:
            [[9223372036854775807, 367, 323],
            [367, 9223372036854775807, 173],
            [323, 173, 9223372036854775807]]

        Dall'esempio si osserva:

        1. La diagonale, cioè da un nodo verso se stesso ho un arco con peso infinito in (pratica ``sys.maxsize``) per
           dire che archi in loopback non sono da esaminare.
        2. La matrice è simmetrica dunque il grafo è non diretto

        Un altro pezzo saliente della funzione è l'operazione di riarrotolamento del genotipo::

            L = max(row, col)
            S = min(row, col)
            index = L * (L - 1) / 2
            index += S - 1
            index -= L
            matrix[col][row] = genome[row]
            matrix[col][row] += genome[col]
            matrix[col][row] += genome[int(index)]

        In pratica prendo due nodi e sommando i loro "pesi" ottengo il peso dell'arco. Di solito si :emphasis:`sottraggono`
        i pesi, in questo caso ho preferito sommarli per evitare il problema degli archi con pesi negativi, altrimenti
        tocca usare algoritmi label-correcting e diventa tutto più complicato...

        :param NUM_NODES: numero colonne e righe per il reshaping
        :return: matrice di adiacenza ottenuta riarrotolando il genotipo
        """
        matrix = [[sys.maxsize for x in range(NUM_NODES)] for x in range(NUM_NODES)]
        genome = self.genotype
        for col in range(len(matrix)):
            for row in range(len(matrix[col])):
                if row != col:
                    L = max(row, col)
                    S = min(row, col)
                    index = L * (L - 1) / 2
                    index += S - 1
                    index -= L
                    index += NUM_NODES
                    matrix[col][row] = genome[row]
                    matrix[col][row] += genome[col]
                    matrix[col][row] += genome[int(index)]
        return matrix
