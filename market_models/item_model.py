import dataclasses
import datetime


class ItemMarketHistory:
    def __init__(self):
        self.region = ""
        self.history_list: []


@dataclasses.dataclass
class ItemHistoryEntry:
    """
    For the day
    """
    totalsupply: int
    price: float
    destroyed_in_region: int
    built_in_region: int
    new_orders: int
    date_of_entry: datetime.datetime
    note: str  # if something weird happens
    # precursor_delta


class Item:
    def __init__(self, precursors: list, total_supply, current_price, name, max_supply: int = None,
                 region="A Region"):
        self.precursors_required_for_build: list[Item] = precursors
        self.total_supply: int = total_supply
        self.current_price: float = current_price
        self.name: str = name
        self.precursors_pointer = []
        self.volatility_coef = 1
        self.region = region
        self.history: list[ItemHistoryEntry] = []
        if max_supply:
            self.max_supply = max_supply
        else:
            self.max_supply = self.total_supply

    def add_prec_pointer(self, prec):
        l0 = len(self.precursors_pointer)
        self.precursors_pointer.append(prec)
        l1 = len(self.precursors_pointer)
        if l0 < l1:
            return True
        return False

    def save_history_entry(self):

        self.history.append(ItemHistoryEntry(self.total_supply,
                                             self.current_price,
                                             self.get_destroyed_in_region,
                                             self.get_built_inregion))

    def recalculate_price(self):
        """
        Get item price recalculation
        :param item:
        :return:
        """
        if self.max_supply != 0:
            self.current_price *= self.volatility_coef * (1 + (self.max_supply - self.total_supply) / self.max_supply)

    def add_to_precursor_list(self):

class Item_Blueprint:
    def __init__(self, name, qty):
        self.name: str = name
        self.qty = qty


def recalculate_price(item):
    """
    Get item reduction
    :param item:
    :return:
    """
    item.current_price *= item.volatility_coef * (
            1 + (item.max_supply - item.total_supply) / item.max_supply)
    return


if __name__ == '__main__':
    market_items = []

    market_items.append(Item([Item_Blueprint("Iron", 5),
                              Item_Blueprint("Coal", 1)],
                             100, 10, "Steel"))
    market_items.append(Item([], 200, 1, "Iron"))
    market_items.append(Item([], 200, 2, "Coal"))
    pass
