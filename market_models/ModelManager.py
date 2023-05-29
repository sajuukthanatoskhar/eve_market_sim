import random
from functools import partial

from matplotlib import pyplot as plt

from market_models.Inventoryf import Inventory
from market_models.p_q_models import Extraction_PQ_Model, Precursor_PQ_Model, InputMaterial, Precursor, base_p_q_model


class ModelManager:
    def __init__(self):
        """
        Manages models from p_q_models.p_base_p_q_model
        """
        self.models = []
        self.location = ""

    def add_model(self, model: base_p_q_model):
        """
        Adds a model
        @param model:
        @return:
        """
        self.models.append(model)

    def get_model_by_name(self, name):
        """
        Gets a model by name
        @param name:
        @return:
        """
        for model in self.models:
            if model.name == name:
                return model
        return None

    def add_connection(self, output_name, input_name):
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

    def make_material_of_random_value(self, name, min_value: int | float, max_value: int | float) -> InputMaterial:
        return InputMaterial(name, self.make_single_random_value(min_value, max_value))

    def run_simulation(self, sim_len, inputs : dict):
        """

        @param sim_len:
        @return:
        """

        t_int = 1  # days
        Timet = list(range(0, sim_len, t_int))

        for a_time_unit in Timet:
            for model in self.models:
                if isinstance(model, Extraction_PQ_Model):

                    model.iterate_model([inputs[model.name]()])
                else:
                    model.iterate_model()
            base_p_q_model.set_Timet()

        for model in self.models:
            model.show_output(Timet, model.greedgain, label="Greed Gain")
            model.show_output(Timet, model.q_out, label="Quantity Output")
            model.show_output(Timet, model.p_out, label="Price Output")


        plt.show()


def generate_extraction_inputs(manager):
    make_coal_material = partial(manager.make_material_of_random_value, 'Coal', 0, 12)
    make_iron_material = partial(manager.make_material_of_random_value, 'Iron', 0, 5)

    extraction_inputs = {
        'Coal': make_coal_material,
        'Iron': make_iron_material
    }

    return extraction_inputs

if __name__ == '__main__':
    # Create a new ModelManager instance
    manager = ModelManager()

    # Create Extraction_PQ_Model objects for coal and iron, and add them to the manager
    coal_extraction = Extraction_PQ_Model(p0=10, q0=800, name="Coal")
    iron_extraction = Extraction_PQ_Model(p0=10, q0=2000, name="Iron")
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


    manager.run_simulation(100, generate_extraction_inputs(manager) )

