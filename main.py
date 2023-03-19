
import simulation_engine.simulator as simulator
from market_models.item_model import *

sim_stepsize = 1


def add_items_to_market_component(market_component: simulator.MarketComponent):

    market_component.add_item(Item([], 200, 2, "Coal"))
    market_component.add_item(Item([], 200, 1, "Iron"))
    market_component.add_item(Item([ItemBlueprint("Iron", 5),
                                    ItemBlueprint("Coal", 1)],
                                   100, 10, "Steel"))

def main():
    sim = simulator.Simulator(stepsize=sim_stepsize)
    market_component = simulator.MarketComponent(stepsize=sim_stepsize)

    add_items_to_market_component(market_component)


    sim.add_component(market_component)



    sim.run()


if __name__ == '__main__':





    main()
