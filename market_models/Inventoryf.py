import dataclasses

from market_models.p_q_models import InputMaterial


class Inventory:
    def __init__(self, area : str):
        self.area = area




    def update_storage(self, input_materials : list[InputMaterial]):
        for material in input_materials:
            found = False
            for stored in self.storage:
                if stored.name == material.name:
                    stored.amount += material.amount
                    found = True
                    break
            if not found:
                self.storage.append(material)

    def get_storage_material(self, matname:str):
        material = [material for material in self.storage if material.name == matname]
        if len(material) > 1:
            raise ValueError(f"Too many ({len(material)} Input materials of type {matname}, something has gone wrong!")

