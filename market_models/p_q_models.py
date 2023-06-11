"""
Price Quantity Models

There are 3


Extraction Only

Extraction + Precursor

Precursor Only
"""
import abc
import dataclasses

from matplotlib import pyplot as plt

from basic import generate_white_gaussian_noise


@dataclasses.dataclass
class Precursor:
    """
    Data class supporting precursor based models
    """
    name: str
    required: int


@dataclasses.dataclass
class InputMaterial:
    name: str
    amount: int
    origin : str = "Unknown"


@dataclasses.dataclass
class InProduction:
    t_started: float
    t_duration: float
    output_quantity: int


def make_gaussian_input_material_input(matname: str, min, max, size) -> list[InputMaterial]:
    wngen = generate_white_gaussian_noise(min, max, size)
    input_mats = []
    for val in wngen:
        input_mats.append(InputMaterial(matname, val))

    return input_mats


class base_p_q_model:
    time_t = [0]
    no_of_plots = 0

    @classmethod
    def inc_no_of_plots(cls):
        cls.no_of_plots += 1

    @classmethod
    def set_Timet(cls):
        cls.time_t.append(cls.time_t[-1] + 1)

    def __init__(self, p0=0, q0=0, name="A_Commodity"):
        self.requested_materials: list[InputMaterial] = [InputMaterial(name, 0)]
        self.p0: float = p0
        self.q0: int = q0
        self.q_out = [q0]
        self.p_out = [p0]
        self.q_dot = 0
        self.p_dot = 0
        self.delay = 0
        self.greedgain = [1]
        self.q_dot_memory = [0]
        self.name = name
        self.input_deliveries: list(InputMaterial) = []

        self.output_connections: list[base_p_q_model] = []
        self.input_connection: list[base_p_q_model] = []

    def add_output_connection(self, connection):
        if connection in [conn for conn in self.output_connections]:
            return None
        self.output_connections.append(connection)
        connection.add_input_connection(self)

    def add_input_connection(self, connection):
        self.input_connection.append(connection)

    @abc.abstractmethod
    def iterate_model(self):
        """
        Iterates through one time
        """
        # self.get_greedGain()

        raise NotImplementedError

    @abc.abstractmethod
    def solve_price(self):
        """
        Solves for the price
        """
        raise NotImplementedError

    @abc.abstractmethod
    def solve_quantity(self, quantity_inputs: list):
        raise NotImplementedError

    @staticmethod
    def get_greedGain(p_0: float, current_p: float, a: float = 1, b: float = 2) -> float:
        """
        An amazing concept in EVE Online, where when the price of something drops
        below its original price at the start of the sim, people will start to not mine it

        @param p_0: price value that was before,
        @param current_p: current price value
        @param a: value that can be updated by an observer controller
        @param b: value that can be updated by an observer controller
        @return:

        """
        greedGain = (a * (current_p / p_0)) ** b
        return greedGain

    def show_output(self, time_x_set: list, y_set: list, label="Graph Title!"):
        plt.figure()
        plt.plot(time_x_set, y_set[:-1])
        plt.title(f"{self.name} {label}")
        self.inc_no_of_plots()

    @abc.abstractmethod
    def get_requested_materials(self, matname):
        NotImplementedError

    def get_sum_of_input_deliveries(self):
        total_requested = []
        for i in self.input_deliveries:
            if i.name == self.name:
                total_requested.append(i.amount)

    def check_availability_of_materials(self, requested_material: InputMaterial, runs):

        if self.q_out[self.time_t[-1]] >= int(requested_material.amount) * runs:
            return True  # If there is enough
        return False

    def get_sum_of_requested_material(self):
        """
        Get the sum of the requested materials that were made
        @return:
        """
        total = sum([mat.amount for mat in self.requested_materials])
        self.requested_materials = []
        return total


@staticmethod
def safely_get_nth_newest_value(single_d_list: list[float | int], n: int) -> float:
    """
    Utility function
    Gets the n-th newest value from single_d_list

    Args:
    single_d_list: A list of numeric values.
    n: The index of the value to retrieve.

    Returns:
    The n-th newest value in single_d_list, if it is greater than or equal to 0.

    Raises:
    ValueError: If the n-th newest value in single_d_list is less than 0.
    IndexError: If there are not enough values in single_d_list to retrieve the n-th newest value.
    """
    try:
        value = single_d_list[n]
    except IndexError:
        if value < 0:
            value = single_d_list[0]

        if value > len(single_d_list):
            value = single_d_list[-1]

    return value


@dataclasses.dataclass
class RequestedMaterial:
    name: str
    amount: int
    FromWhere: base_p_q_model

    def check_requested_amount_is_there(self):
        """
        Checks the FromWhere object if there is enough of the RequestedMaterial
        @return:
        """
        req_conn = self.FromWhere
        if (req_conn.q_out[req_conn.time_t] - (req_conn.get_sum_of_input_deliveries())) >= self.amount:
            return True
        return False


class Extraction_PQ_Model(base_p_q_model):
    """
    Extraction_PQ_Model
    """

    def get_requested_materials(self):
        """
        Not needed to be implemented as an Extraction_PQ_Model has people mine/collect things from elsewhere.
        It relies on entire stochastic inputs.
        @return:
        """
        raise NotImplementedError

    def __init__(self, p0=0, q0=0, name="A_Commodity"):
        super().__init__(p0=p0, q0=q0, name=name)

    def solve_price(self):
        """
        Solves the price of the extracted materials
        @return:
        """

        self.p_out.append(self.p_out[-1] - self.p_out[-1] * ((self.q_dot_memory[-1]) / self.q_out[-1]))

    def solve_quantity(self, quantity_inputs: list[InputMaterial]):

        i: InputMaterial | int
        for i in quantity_inputs:
            try:
                if i.name == self.name:
                    if i.amount > 0:
                        self.q_dot += i.amount * self.greedgain[self.time_t[-1]]
                    else:
                        self.q_dot += i.amount
            except AttributeError:
                self.q_dot += i

        self.q_dot -= sum([mat.amount for mat in self.requested_materials])  # removes
        self.reset_requested_materials()
        self.q_dot_memory.append(self.q_dot)

        self.q_out.append(max(self.q_dot + self.q_out[-1], 1))
        self.q_dot = 0

    def reset_requested_materials(self):
        self.requested_materials = []

    def iterate_model(self, inputs: list[InputMaterial | float | int] = []):
        self.greedgain.append(self.get_greedGain(self.p_out[0], self.p_out[-1], a=1))
        self.solve_quantity(inputs)
        self.solve_price()

    def get_sum_of_requested_material(self):
        """
        Gets the total amount of requested material
        @return:
        """
        requested_mats: list[InputMaterial] = [0]
        conn: Extraction_PQ_Model | Precursor_PQ_Model

        for mat in requested_mats:
            pass
        return sum(requested_mats)


class Precursor_PQ_Model(base_p_q_model):

    def __init__(self, p0=0, q0=0, name="A_Commodity", precursors: list[Precursor] = [],
                 production_time=1, no_of_factory_lines=1, build_output=100,
                 production_set_quantity=1):
        """

        @param p0: price at time 0
        @param q0: quantity at time 0
        @param name: Name of the Resource
        @param precursors:
        @param production_time:
        @param no_of_factory_lines:
        @param build_output:
        @param production_set_quantity:
        """
        super().__init__(p0=p0, q0=q0, name=name)
        self.precursor_list: list[Precursor] = precursors
        self.no_of_factories: int = no_of_factory_lines  # For the precursor stuff
        self.production_time_to_build: int = production_time
        self.production_set_quantity: int = production_set_quantity
        self.stuff_in_production: list[InProduction] = []
        self.build_output = build_output
        self.required_materials_from_connections: list[InputMaterial] = []

    def get_requested_materials(self, matname):

        return [input.amount for input in self.required_materials_from_connections if input.name == matname]

    def iterate_model(self, inputs: list[InputMaterial | float | int] = []):
        """
        This is different from the extraction model
        @param inputs:
        @param requested_materials: are output materials that have been requested else where.
        @return:
        """
        self.greedgain.append(self.get_greedGain(self.p_out[0], self.p_out[-1]))

        self.deliver_stuff_into_production(inputs)

        outputs = self.deliver_stuff_out_of_production()
        sum_of_inputs = sum([input.amount for input in inputs if input.name == self.name])
        sum_of_requested_material = self.get_sum_of_requested_material()
        sum_of_outputs = sum([an_output.output_quantity for an_output in outputs])
        self.q_dot = sum_of_outputs - sum_of_requested_material + sum_of_inputs
        # todo : waste products!
        self.q_dot_memory.append(self.q_dot)

        self.q_out.append(self.q_dot + self.q_out[-1])

        self.solve_price()

    def solve_price(self):
        self.p_out.append(self.p_out[-1] - self.p_out[-1] * ((self.q_dot) / self.q_out[-1]))

    def solve_quantity(self, quantity_inputs: list):
        pass

    def deliver_stuff_into_production(self, inputs: list[InputMaterial]):
        """
        Puts the stuff into 'production'
        required_materials_from_connections needs to be checked
        @param inputs:
        @return:
        """

        self.required_materials_from_connections = []  # reset each time
        self.set_required_input_materials_for_prod(inputs)
        # Get the values of the required build
        if self.confirm_input_availability_with_input_connections():
            self.remove_inputmats_from_stockpile()

            # Make the building of the stuff, sets the time for it
            self.stuff_in_production.append(InProduction(self.time_t[-1],
                                                         self.get_timeduration(),
                                                         self.get_total_output()))
        else:
            pass  # Nothing was placed into production!

    def deliver_stuff_out_of_production(self) -> list[InProduction]:
        """
        Takes stuff out of production when it has reached its time
        @param inputs:
        @return:
        """
        outputs = []
        a_production_slot: InProduction
        self.stuff_in_production: list
        for a_production_slot in self.stuff_in_production:
            if a_production_slot.t_started + a_production_slot.t_duration >= self.time_t[-1]:
                outputs.append(a_production_slot)
                self.stuff_in_production.remove(a_production_slot)

        return outputs

    def set_required_input_materials_for_prod(self, inputs) -> bool:
        """
        Sets the required amount of input materials
        @param inputs:
        @return: Nothing is returned and a return is forced if there arne't enough materials to begin a build
        """
        self.required_materials_from_connections = []  # resets each time
        conn: Extraction_PQ_Model | Precursor_PQ_Model

        for a_precursor in self.precursor_list:
            self.required_materials_from_connections.append(
                InputMaterial(a_precursor.name, self.precursor_amount_required_for_build(a_precursor)))

    def precursor_amount_required_for_build(self, a_precursor):
        return a_precursor.required * self.get_build_runs()

    def get_build_runs(self):
        return self.production_set_quantity * self.no_of_factories

    def get_total_output(self) -> float | int:
        return self.build_output * self.no_of_factories

    def get_timeduration(self):
        return self.get_build_runs() * self.production_time_to_build

    def confirm_input_availability_with_input_connections(self) -> bool:
        """ Confirms with the input connections that the requested material """
        no_of_input_conns = len(self.input_connection)
        confirm_req_mat_there = []
        for input_conn in self.input_connection:
            for requested_material in self.required_materials_from_connections:
                if input_conn.name == requested_material.name and input_conn.check_availability_of_materials(
                        requested_material, self.get_build_runs()):
                    confirm_req_mat_there.append(True)

        if len(confirm_req_mat_there) == len(self.required_materials_from_connections):
            return True  # if the connections have the required amounts and were successfully subtracted
        return False

    def remove_inputmats_from_stockpile(self):
        for input_conn in self.input_connection:
            for requested_material in self.required_materials_from_connections:
                if input_conn.name == requested_material.name:
                    input_conn.requested_materials.append(
                        InputMaterial(requested_material.name, int(requested_material.amount) * self.get_build_runs()))


class Modeltype:
    model_type_extraction = 'extraction'
    model_type_precurser = 'precursor'


class PQModelFactory:
    """
    Factory for P_Q Model creation
    """

    @staticmethod
    def create(model_type, p0=10, q0=10, name="A resource", **kwargs) -> base_p_q_model:




        if model_type == Modeltype.model_type_extraction:
            return Extraction_PQ_Model(p0 = p0, q0 = q0, name = name)
        elif model_type == Modeltype.model_type_precurser:
            return Precursor_PQ_Model(p0 = p0, q0 = q0, name = name, **kwargs)
        else:
            return None


if __name__ == '__main__':
    sim_len = 100
    t_int = 1  # days
    Timet = list(range(0, 100, 1))
    some_coal = Extraction_PQ_Model(p0=10, q0=800, name="Coal")
    some_iron = Extraction_PQ_Model(p0=10, q0=2000, name="Iron")
    iron_input = make_gaussian_input_material_input('Iron', 5, 0, len(Timet) - 1)
    coal_input = make_gaussian_input_material_input('Coal', 5, 0, len(Timet) - 1)

    some_steel = Precursor_PQ_Model(p0=500, q0=1000, name="Steel",
                                    precursors=[Precursor("Iron", "5"),
                                                Precursor("Coal", "20")])
    some_hard_steel = Precursor_PQ_Model(p0=500, q0=1000, name="Hard Steel",
                                         precursors=[Precursor("Iron", "10"),
                                                     Precursor("Coal", "1")])
    some_coal.add_output_connection(some_steel)
    some_iron.add_output_connection(some_steel)

    some_coal.add_output_connection(some_hard_steel)
    some_iron.add_output_connection(some_hard_steel)

    for t in Timet:
        some_coal.iterate_model([coal_input[t - 1]])
        some_iron.iterate_model([iron_input[t - 1]])
        some_steel.iterate_model()
        some_hard_steel.iterate_model()
        base_p_q_model.set_Timet()

    some_coal.show_output(Timet, some_coal.q_out, label="Quantity Output")
    some_coal.show_output(Timet, some_coal.p_out, label="Price Output")

    some_iron.show_output(Timet, some_iron.q_out, label="Quantity Output")
    some_iron.show_output(Timet, some_iron.p_out, label="Price Output")

    some_steel.show_output(Timet, some_steel.q_out, label="Quantity Output")
    some_steel.show_output(Timet, some_steel.p_out, label="Price Output")

    some_hard_steel.show_output(Timet, some_hard_steel.q_out, label="Quantity Output")
    some_hard_steel.show_output(Timet, some_hard_steel.p_out, label="Price Output")

    # some_coal.show_output(Timet, some_coal.greedgain, label="GreedGain Output")
    # plt.plot(Timet, some_coal.p_out[:-1])
    # plt.title("Price of Goods")
    # plt.figure(2)
    # plt.plot(Timet, some_coal.q_out[:-1])
    # plt.title("Quantity of Goods")
    plt.show()


class Resource_Input_Model:
    """
    This is the group of resource inputs that either provide a regular amount of resources or are random events
    These can be thought of as

    a) Miners providing resources, modelled gains/losses over time

    b) One time events such as EVE Battles, thefts, donations


    They are 'lazily' added to the model manager and require NO connections.  Why?
    Because they are inputs and do not store any quantity of materials.

    """

    def __init__(self, name="a_resource", amount: int = 0, interval: int = 5,
                 timer = 0, quantity = 1, resource = "Tritanium"):
        self.name: str = name
        self.amount: int = amount
        self.interval: int = interval
        self.resource = resource
        self.timer: int = timer
        self.quantity: int = quantity

    def reset_give_timer(self, timer: float | int = 0) -> bool:
        """
        Resets the timer in which the R_I_M uses to calculate when it will drop off resources next

        :return: True if successful.
        """
        if isinstance(timer, int) or isinstance(timer, float):
            self.timer = timer
            return True
        print("Incorrect type, should be int or float")
        return False

    def increment_timer(self) -> None:
        """
        Increments the timer as part of model iteration
        :return:
        """
        self.timer += 1

    def set_resource(self, resource: str) -> bool:
        """
        Sets the resource output of the R_I_M
        :param resource:
        :return:
        """
        if isinstance(resource, str):
            self.resource = resource
            return True
        print("Incorrect type, should be str")
        return False

    def set_quantity(self, quantity: int = 1) -> False:
        """
        Sets the quantity of ships doing the same thing
        :param quantity:
        :return: bool if integer
        """
        if isinstance(quantity, int):
            self.quantity = quantity
            return True
        print("Incorrect type, should be int")
        return False

    def output_resource(self) -> InputMaterial|None:
        """
        Outputs a resource if the time is right, increments timer if not
        Part of iteration
        :return:
        """
        if self.timer == self.interval:
            self.reset_give_timer()
            return InputMaterial(self.resource, self.amount*self.quantity, self.name)
        self.increment_timer()
        return InputMaterial(self.resource, 0*self.quantity, self.name)


