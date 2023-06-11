import functools
import random
from functools import partial
from typing import Dict, Any, Type

from matplotlib import pyplot as plt

from market_models.Inventoryf import Inventory
from market_models.p_q_models import Resource_Input_Model, Extraction_PQ_Model, Precursor_PQ_Model, InputMaterial, Precursor, base_p_q_model, \
    PQModelFactory


class ModelManager:
    def __init__(self):
        """
        Manages models from p_q_models.p_base_p_q_model
        """
        self.models = []
        self.location = ""
        self.random_inputs : list[Resource_Input_Model] = []
        self.random_input_models : list[Resource_Input_Model] = []
        self.extraction_inputs = dict



    def add_model(self, model: base_p_q_model | PQModelFactory) -> bool:
        """
        Adds a model
        @param model:
        @return:
        """

        if model:
            self.models.append(model)
            return True
        return False

    def get_model_by_name(self, name) -> base_p_q_model | None:
        """
        Gets a model by name
        @param name:
        @return:
        """
        for model in self.models:
            if model.name == name:
                return model
        return None

    def add_precursors_to_model(self, model_name: str, precursor_list: list[Precursor]) -> bool:
        """
        Adds a list of precursors to a Precursor_PQ_Model
        @param model_name:
        @param precursor_list:
        @return:
        """
        a_model = self.get_model_by_name(model_name)

        if isinstance(a_model, Precursor_PQ_Model):
            for a_precursor in precursor_list:
                a_model.precursor_list.append(a_precursor)
            return True
        return False

    def add_connection(self, output_name, input_name) -> bool:
        """
        Adds a connection from one model, to another
        @param output_name:
        @param input_name:
        @return:
        """
        output_model = self.get_model_by_name(output_name)
        input_model = self.get_model_by_name(input_name)
        if output_model is not None and input_model is not None:
            output_model.add_output_connection(input_model)
            return True
        return False

    @staticmethod
    def make_single_random_value(min_val, max_val, discrete: bool = True) -> float | int:
        """
        Generates a single random value within the given range.

        :param min_val: Minimum value of the range (inclusive).
        :param max_val: Maximum value of the range (inclusive).
        :param discrete: If True, the output will be an integer; if False, it will be a float.
        :return: A single random value within the given range.
        """
        if discrete:
            return random.randint(min_val, max_val)
        else:
            return random.uniform(min_val, max_val)

    def add_discrete_extraction_input(self, name, min, max) -> None:
        """
        Adds a discrete extraction input function, this is called in a Extraction_PQ_Model.

        !!Deprecated!!
        @param name:
        @param min:
        @param max:
        @return:
        """
        self.extraction_inputs[name] = partial(manager.make_material_of_random_value, name, min, max)

    def make_material_of_random_value(self, name, min_value: int | float, max_value: int | float) -> InputMaterial:
        """

        @param name:
        @param min_value:
        @param max_value:
        @return:
        """
        return InputMaterial(name, self.make_single_random_value(min_value, max_value))

    def run_simulation(self, sim_len : int, t_interval : float = 1) -> None:
        """
        Runs the simulation and poops out alot of graphs
        @param sim_len: is the simulation max length, typically thought of as number of days, but can be minutes, hours, whatever.
        :param t_interval: at the moment, this should be set to 1 since python floats do weird things and I haven't tested it.  If you want to simulate hours, set sim_len to number of hours.
        @return: None
        """

        Timet = list(range(0, sim_len, t_interval))
        self.run_core_simulation(Timet)
        self.create_graphs_and_show(Timet)

    def run_core_simulation(self, Timet: object) -> object:
        """
        Runs the collection of resources at timeT
        :param Timet:
        :return:
        """
        for a_time_unit in Timet:
            collected_resources = self.collect_resources_from_rims()
            self.iterate_p_q_models(collected_resources)
            self.increment_p_q_model_timers()
            self.attacks_on_resources()

    def iterate_p_q_models(self, collected_resources):
        for model in self.models:
            if isinstance(model, Extraction_PQ_Model):
                model.iterate_model(collected_resources)  # input model function is called here
            else:
                model.iterate_model()

    def increment_p_q_model_timers(self):
        base_p_q_model.set_Timet()

    def create_graphs_and_show(self, Timet):
        for model in self.models:
            self.create_Greedgain_graph(Timet, model)
            self.create_quantity_output_graph(Timet, model)
            self.create_output_graph(Timet, model)
        plt.show()

    def create_output_graph(self, Timet, model):
        model.show_output(Timet, model.p_out, label="Price Output")

    def create_quantity_output_graph(self, Timet, model):
        model.show_output(Timet, model.q_out, label="Quantity Output")

    def create_Greedgain_graph(self, Timet, model):
        model.show_output(Timet, model.greedgain, label="Greed Gain")

    def generate_extraction_inputs(self) -> dict:
        return self.extraction_inputs

    def create_random_input_model(self, r_i_m_name, amount, time_interval, timer, quantity, resource):
        self.random_inputs.append(Resource_Input_Model(name = r_i_m_name, amount = amount,
                                                       interval = time_interval, timer = timer, quantity = quantity,
                                                       resource = resource))

    def collect_resources_from_rims(self) -> list[InputMaterial]:
        """
        Collects all resources (doesn't collate or combine)
        :return:
        """
        rim_model: Resource_Input_Model
        resources_collected = []
        for rim_model in self.random_inputs:
            resources_collected.append(rim_model.output_resource())

        return resources_collected

    def attacks_on_resources(self):
        """
        Unimplemented simulation of attacks on collectors, which will decrease the total amount of incoming resources

        :return:
        """
        return None


def generate_extraction_inputs(manager) -> dict:
    """
    Generates random
    @param manager:
    @return:
    """
    make_coal_material = partial(manager.make_material_of_random_value, 'Coal', 0, 12)
    make_iron_material = partial(manager.make_material_of_random_value, 'Iron', 0, 5)
    make_trit = partial(manager.make_material_of_random_value, 'Iron', 0, 5)

    extraction_inputs = {
        'Coal': make_coal_material,
        'Iron': make_iron_material,
        'Tritanium': make_trit
    }

    return extraction_inputs


if __name__ == '__main__':
    # Create a new ModelManager instance
    manager = ModelManager()

    # Create Extraction_PQ_Model objects for coal and iron, and add them to the manager
    coal_extraction = Extraction_PQ_Model(p0=10, q0=800, name="Coal")
    iron_extraction = Extraction_PQ_Model(p0=10, q0=2000, name="Iron")
    tritanium = Extraction_PQ_Model(p0=5, q0=200000, name="Tritanium")
    manager.add_model(coal_extraction)
    manager.add_model(iron_extraction)

    # Create Precursor_PQ_Model objects for steel and hard steel, and add them to the manager
    steel_precursor = Precursor_PQ_Model(p0=500, q0=1000, name="Steel",
                                         precursors=[Precursor("Iron", 5), Precursor("Coal", 20)])
    hard_steel_precursor = Precursor_PQ_Model(p0=500, q0=1000, name="Hard Steel",
                                              precursors=[Precursor("Iron", 1), Precursor("Coal", 1)])
    # Add the two Precursor_PQ_Model objects to the manager
    manager.add_model(steel_precursor)
    manager.add_model(hard_steel_precursor)

    # Connect the Extraction_PQ_Model objects to both Precursor_PQ_Model objects
    manager.add_connection("Coal", "Hard Steel")
    manager.add_connection("Iron", "Hard Steel")
    manager.add_connection("Coal", "Steel")
    manager.add_connection("Iron", "Steel")

    manager.run_simulation(100)
