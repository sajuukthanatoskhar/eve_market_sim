import market_models.item_model
import simulation_engine.simulator as simulator

sim_stepsize = 1


def main():
    sim = simulator.Simulator(stepsize=sim_stepsize)

    sim.add_component(simulator.MarketComponent(stepsize=sim_stepsize))
    sim.run()


if __name__ == '__main__':
    main()
