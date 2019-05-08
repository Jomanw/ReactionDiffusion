from turing_system import OriginalSystem, GM, Schnakenberg, SimpleSystem, FiveElementCoupled, Oscillatory, GMSC, Latch, AnisotropicSchnakenberg, DoubleSC
from plotter import Plotter

# CONSTANTS
grid_size = 50
step_size = 2./grid_size
T = 5
fps = 60
dt = .0000035

update_frequency = 100
num_iterations = int(T / dt)
system_type = "dsc"
filetype = '.gif'

make_animation = True
run_experiment = False


# END OF CONSTANTS

systems = {
    "original":OriginalSystem(grid_size=grid_size, dt = dt),
    "sc": Schnakenberg(grid_size=grid_size, dt = .00001),
    "gm": GM(grid_size=grid_size, dt = .00001),
    "simple": SimpleSystem(grid_size=grid_size, dt = dt),
    "fec": FiveElementCoupled(grid_size=grid_size, dt = dt),
    "osc": Oscillatory(grid_size=grid_size, dt=.0001),
    "gmsc": GMSC(grid_size=grid_size, dt = .00001),
    "latch": Latch(grid_size=grid_size, dt = dt),
    "asc": AnisotropicSchnakenberg(grid_size=grid_size, dt=dt),
    "dsc": DoubleSC(grid_size=grid_size, dt = .00001)

}




# # system = OriginalSystem(grid_size=grid_size, dt = dt)
# system = Schnakenberg(grid_size=grid_size, dt = dt)
# # system = GM(grid_size=grid_size, dt = dt)
system = systems[system_type]
plotter = Plotter(system, step_size,
                    update_frequency=update_frequency,
                    save_name = system_type,
                    make_rgb=[0, 0, 2])
if run_experiment:
    count = 0
    system.remove_activator()
    while count < 25:
        plotter.run(50)
        count += 1
    system.expose_activator()
    count = 0
    while count < 25:
        plotter.run(50)
        count += 1
    count = 0
    system.unexpose_activator()
    while count < 25:
        plotter.run(50)
        count += 1


if make_animation:
    plotter.make_animation(num_iterations, fps=fps, frames = T * fps, filetype = filetype)
else:
    plotter.run(num_iterations)
