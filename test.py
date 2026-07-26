import sys

sys.path.append(r"C:\EnergyPlusV26-1-0")

from pyenergyplus.api import EnergyPlusAPI

api = EnergyPlusAPI()
state = api.state_manager.new_state()

temp_handle = None


def callback(state):
    global temp_handle

    # Wait until API data is available
    if not api.exchange.api_data_fully_ready(state):
        return

    # Get handle only once
    if temp_handle is None:
        temp_handle = api.exchange.get_variable_handle(
            state,
            "Zone Mean Air Temperature",
            "ZONE ONE"
        )
        print("Handle:", temp_handle)

        if temp_handle == -1:
            print("Variable not found!")
            return

    temp = api.exchange.get_variable_value(state, temp_handle)
    print(f"Temperature: {temp:.2f} °C")


api.runtime.callback_end_zone_timestep_after_zone_reporting(
    state,
    callback
)

weather = r"C:\EnergyPlusV26-1-0\WeatherData\USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"

idf = r"C:\EnergyPlusV26-1-0\ExampleFiles\1ZoneUncontrolled.idf"

api.runtime.run_energyplus(
    state,
    [
        "-w",
        weather,
        idf,
    ],
)

print("Simulation Finished!")