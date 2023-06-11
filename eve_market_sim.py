from market_models.ModelManager import ModelManager
from market_models.p_q_models import PQModelFactory, Modeltype, Precursor, Resource_Input_Model


class EVEMarket():
    def __init__(self, region = "A_Region"):
        self.region = region
        self.model_manager = ModelManager()

    def add_precursors_to_model(self, precursorlist: list[Precursor], name: str):

        return self.model_manager.add_precursors_to_model(model_name=name,
                                                    precursor_list=precursorlist)

    def add_resource(self, name:str, q0:float, p0:float, model_type:str, **kwargs) -> bool:
        """
        Add resource blocks to model
        @param model_type:
        @param name:
        @param q0:
        @param p0:
        @return:
        """
        return self.model_manager.add_model(PQModelFactory.create(model_type, q0=q0, p0=p0, name=name, **kwargs))

    def link_resources(self, output : str = "", input : str = "") -> bool:
        """
        Wrapper method to link resource blocks
        @param input: Name of input
        @param output: Name of output
        @return:
        """
        if self.model_manager.add_connection(output, input):
            return True
        return False

    def add_input(self, r_i_m_name, amount, time_interval, timer, quantity, resource):

        self.model_manager.create_random_input_model(r_i_m_name, amount, time_interval, timer, quantity, resource)


if __name__ == '__main__':
    an_EVEMarket = EVEMarket(region="The Forge")

    an_EVEMarket.add_resource("Tritanium", 28000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Pyerite", 28000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Mexallon", 28000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Isogen", 28000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Megacyte", 28000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Zydrine", 28000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Morphite", 28000, 5, Modeltype.model_type_extraction)

    an_EVEMarket.add_resource("Rifter", 5, 500000, Modeltype.model_type_precurser, build_output = 1)

    an_EVEMarket.add_precursors_to_model([
        Precursor("Tritanium", 28800),
        Precursor("Isogen", 450),
        Precursor("Mexallon", 2250),
        Precursor("Pyerite", 5400)
    ], "Rifter")

    an_EVEMarket.add_input(r_i_m_name = "A Trit collector", amount= 500, time_interval = 3,
                           timer = 0,quantity = 2, resource = "Tritanium" )
    an_EVEMarket.add_input(r_i_m_name = "A Pyerite collector", amount= 750, time_interval = 4,
                           timer = 0,quantity = 2, resource = "Pyerite" )
    an_EVEMarket.add_input(r_i_m_name = "An Isogen collector", amount= 700, time_interval = 3,
                           timer = 0,quantity = 2, resource = "Isogen" )
    an_EVEMarket.add_input(r_i_m_name = "A mexallon collector", amount= 1000, time_interval = 3,
                           timer = 0,quantity = 2, resource = "Mexallon" )

    an_EVEMarket.link_resources("Tritanium","Rifter")
    an_EVEMarket.link_resources("Pyerite","Rifter")
    an_EVEMarket.link_resources("Mexallon","Rifter")
    an_EVEMarket.link_resources("Isogen","Rifter")

    an_EVEMarket.model_manager.run_simulation(100)