import queue as q

from CMST_package.individual import Individual


class Hof:
    """Questa classe contiene la Hall of Fame

    In questa classe metto tutto quello che serve per memorizzare la cosiddetta Hall of Fame. Questa classe semplicemente
    conterrà le migliori soluzioni fino ad ora viste.

    .. note::
        L'idea della struttura dati della Hall of Fame proviene dal framework DEAP python.
    """

    def __init__(self, size):
        """
        Inizializzo l'oggetto Hall of Fame il quale conterrà le migliori soluzioni (da 1 fino a size) nella sua coda
        interna che ho chiamato ``individuals``. Questa coda è ordinata in ordine di fitness.

        .. seealso:: :class:`Individual` per vedere i confronti come sono implementati

        :param individuals: la coda "privata" che conterrà le soluzioni
        :param size: the first value
        :param paths: the first value
        """
        self.individuals = q.Queue(size)
        self.size = size
        self.paths = []

    def add(self, individual):
        """
        Aggiungi un nuovo membro alla hall of fame. Ho osservato una nuova soluzione e devo valutare se è il caso di
        immetterlo nella hall of fame. Creo una coda ordinata (in ordine di fitness), inserisco gli individui mai visti
        e quelli vecchi e poiché le dimensioni fra HOF e popolazione potrebbero essere diverse, fintanto che una delle
        due non è vuota continua a immettere gli elementi dalla coda di appoggio alla HOF.

        :type individual: Individual
        :param individual: Individuo da aggiungere nella HOF
        :return:
        """
        temp = q.PriorityQueue()
        while not self.individuals.empty():
            old_ind = self.individuals.get()
            if individual == old_ind:
                continue
            temp.put(old_ind)
        # Se nella coda temporanea non c'è questo individuo allora aggiungi
        temp.put(individual)
        # Finché hai individui a disposizione e posto nella coda hall of fame, aggiungi in ordine di fitness
        while not self.individuals.full() and not temp.empty():
            self.individuals.put(temp.get())

    def update(self, population):
        """
        Aggiorna la hall of fame poiché è cambiata la popolazione a seguito di un ciclo evolutivo. Il procedimento è il
        seguente:
        1. Ordina la popolazione in base alla fitness
        2. Seleziona i migliori individui (i quali stanno in cima alla coda)
        3. Per ognuno di questi individui migliori rimettili nella hall of fame

        :type population: list
        :param population: popolazione su cui valutare la hall of fame
        :return:
        """
        population.sort()
        best_candidates = population[0:self.size]  # Slicing della lista e selezione dei migliori (da 0 a N)
        for individual in best_candidates:
            self.add(individual)

    def __str__(self):
        """
        Funzione di utility molto utile quando devo fare le print(). Qui visualizzo le informazioni sulla hall of fame

        :return:
        """
        ret = ''
        while not self.individuals.empty():
            ind = self.individuals.get()
            ret += "Fitness: " + str(ind.fitness) + "\n"
            ret += "Tree: " + str(ind.tree)
            self.paths.append(ind.tree)
            ret += "\n\n"
        return ret
