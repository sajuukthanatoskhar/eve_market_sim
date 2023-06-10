from market_models.ModelManager import ModelManager
from market_models.p_q_models import PQModelFactory, Modeltype, Precursor


class EVEMarket():
    def __init__(self, region = "A_Region"):
        self.region = region
        self.model_manager = ModelManager()

    def add_precursors_to_model(self, precursorlist: list[Precursor], name: str):

        return self.model_manager.add_precursors_to_model(model_name=name,
                                                    precursor_list=precursorlist)

    def add_resource(self, name:str, q0:float, p0:float, model_type:str) -> bool:
        """
        Add resource blocks to model
        @param model_type:
        @param name:
        @param q0:
        @param p0:
        @return:
        """
        return self.model_manager.add_model(PQModelFactory.create(model_type, name, q0,p0))

    def link_resources(self, input : str, output : str) -> bool:
        """
        Wrapper method to link resource blocks
        @param input: Name of input
        @param output: Name of output
        @return:
        """
        if self.model_manager.add_connection(output_name=output, input_name=input):
            return True
        return False




if __name__ == '__main__':
    an_EVEMarket = EVEMarket(region="The Forge")

    an_EVEMarket.add_resource("Tritanium", 1000000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Pyerite", 1000000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Mexallon", 1000000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Isogen", 1000000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Megacyte", 1000000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Zydrine", 1000000, 5, Modeltype.model_type_extraction)
    an_EVEMarket.add_resource("Morphite", 1000000, 5, Modeltype.model_type_extraction)

    an_EVEMarket.add_resource("Rifter", 20, 500000, Modeltype.model_type_precurser)

    an_EVEMarket.add_precursors_to_model([
        Precursor("Tritanium", 28800),
        Precursor("Isogen", 450),
        Precursor("Mexallon", 2250),
        Precursor("Pyerite", 5400)
    ], "Rifter")

