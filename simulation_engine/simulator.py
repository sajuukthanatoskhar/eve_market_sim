import market_models.item_model
import logging

logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
DEBUG = True


class SimStates:
    RUNNING = "running"
    STOPPED = "stopped"
    NULL = None


class SimulationComponent:
    def __init__(self):
        pass

    def step(self):
        logging.warning('SimulationComponent.step:Not Implemented')


class MarketComponent(SimulationComponent):
    """
    Just a component
    """

    def __init__(self, stepsize=1):
        self.items: list[market_models.item_model.Item] = []
        self.precursor_changes = []
        self.quantity_changes = []
        self.max_supply_changes = []
        super().__init__()
        self.stepsize = stepsize

    def step(self):
        self.update_quantities()
        self.change_precursors()
        self.update_prices()
        self.update_histories()
        if DEBUG:
            print("Market Step process completed")

    def update_quantities(self):
        if len(self.quantity_changes) == 0:
            return True

        # check if key exists

    def change_precursors(self):
        """
        Whenever the precursor requirements to produce an item changes - this is where those changes happen
        :return:
        """
        if len(self.precursor_changes) != 0:
            for item in self.items:
                logging.warning('Not Implemented')
        logging.warning('Not Implemented')

    def update_prices(self):
        for item in self.items:
            item.recalculate_price()
        return 1

    def update_histories(self):
        """
        Updates all item histories
        :return:
        """
        item: market_models.item_model.Item
        for item in self.items:
            item.save_history_entry()

        logging.warning('Not Implemented')

    def update_max_supply(self):

        logging.warning('Not Implemented')

    def add_max_supply_entry(self, value: dict):
        # dict is item_name: value
        if type(value) != dict:
            return False
        return self.Check_if_change_exists(self.max_supply_changes, value)

    def add_precursor_entry(self, value: dict):
        # dict is item_name: value
        if type(value) != dict:
            return False
        return self.Check_if_change_exists(self.precursor_changes, value)


    def add_quantities_entry(self, value: dict):
        # dict is item_name: value
        if type(value) != dict:
            return False
        return self.Check_if_change_exists(self.quantity_changes, value)

    def Check_if_change_exists(self, dump_var, value):
        """
        Checks if a item's value that is being inserted into the list of changes already exists for
        Uses Key of the value
        :param dump_var:
        :param value:
        :return:
        """
        for akey in value:
            existing_keys: dict
            for existing_keys in dump_var:
                if existing_keys.get(akey):
                    existing_keys[akey] = value[akey]
                    return True
        dump_var.append(value)
        return True

    def add_item(self, item):
        """
        Adds item to the market component
        :param item:
        :return:
        """
        self.items.append(item)

    def remove_item(self, item_name: str):
        """
        Removes the item from the market component
        :param item_name:
        :return:
        """
        raise NotImplementedError

    def add_precursor_to_item(self, precursor, item_to_modify:str):
        """
        Adds a precursor item to an item
        :param precursor:
        :param item_to_modify:
        :return: True if the item was added successfully
        """
        for item in self.items:
            if item.name == item_to_modify:
                return item.add_prec_pointer(precursor)
        return False


class Simulator:
    def __init__(self, stepsize=1):
        self.state = SimStates.NULL
        self.stepsize = stepsize
        self.components = []

    def run(self):
        print("Running Simulation")
        self.start = SimStates.RUNNING
        component: SimulationComponent
        for component in self.components:
            component.step()

    def stop(self):
        print("Stopping Simulation")
        self.state = SimStates.STOPPED

    def add_component(self, component: SimulationComponent):
        self.components.append(component)
