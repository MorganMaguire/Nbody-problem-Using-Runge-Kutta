import numpy as np 
import matplotlib.pyplot as plt 
import astropy.units as u 
import astropy.constants as c 
import sys 
import time
from mpl_toolkits.mplot3d import Axes3D

#making a class for Celestial Objects
class CelestialObjects():
    def __init__(self,mass,pos_vec,vel_vec,name=None, has_units=True):
        self.name=name
        self.has_units=has_units
        if self.has_units:
            self.mass=mass.cgs
            self.pos=pos_vec.cgs.value
            self.vel=vel_vec.cgs.value
        else:
            self.mass=mass 
            #3d array for position of body in 3d space in AU
            self.pos=pos_vec 
            #3d array for velocity of body in 3d space in km/s
            self.vel=vel_vec
        
    def return_vec(self):
        return np.concatenate((self.pos,self.vel))
    def return_name(self):
        return self.name
    def return_mass(self):
        if self.has_units:
            return self.mass.cgs.value
        else:
            return self.mass

v_earth=(((c.G*1.98892E30)/1.495978707E11)**0.5)/1000
v_jupiter=(((c.G*1.98892E30)/7.779089276E11)**0.5)/1000
v_saturn=(((c.G*1.98892E30)/1.421179772E12)**0.5)/1000
v_neptune=(((c.G*1.98892E30)/4.5028959E12)**0.5)/1000
v_uranus=(((c.G*1.98892E30)/2.872279117E12)**0.5)/1000
v_mars=(((c.G*1.98892E30)/2.243968061E11)**0.5)/1000
v_venus=(((c.G*1.98892E30)/1.047185095E11)**0.5)/1000
v_mercury=(((c.G*1.98892E30)/5.983914828E10)**0.5)/1000
v_asteroid=(((c.G*1.98892E30)/4.936729733E11)**0.5)/1000

Earth=CelestialObjects(name='Earth',
                       pos_vec=np.array([0,1,0])*u.AU,
                       vel_vec=np.array([v_earth.value,0,0])*u.km/u.s,
                       mass=1.0*c.M_earth)
Mercury=CelestialObjects(name='Mercury',
                        pos_vec=np.array([0,0.4,0])*u.AU, 
                        vel_vec=np.array([v_mercury.value,0,0])*u.km/u.s,
                        mass=0.055*c.M_earth)
Venus=CelestialObjects(name='Venus',
                        pos_vec=np.array([0,0.7,0])*u.AU, 
                        vel_vec=np.array([v_venus.value,0,0])*u.km/u.s,
                        mass=0.815*c.M_earth)
Mars=CelestialObjects(name='Mars',
                        pos_vec=np.array([0,1.5,0])*u.AU, 
                        vel_vec=np.array([v_mars.value,0,0])*u.km/u.s,
                        mass=0.1074*c.M_earth)
Sun=CelestialObjects(name='Sun',
                     pos_vec=np.array([0,0,0])*u.AU,
                     vel_vec=np.array([0,0,0])*u.km/u.s,
                     mass=1*u.Msun)
Jupiter=CelestialObjects(name='Jupiter', 
                         pos_vec=np.array([0,5.2,0])*u.AU, 
                         vel_vec=np.array([v_jupiter.value,0,0])*u.km/u.s,
                         mass=317.8*c.M_earth)
Saturn=CelestialObjects(name='Saturn',
                        pos_vec=np.array([0,9.5,0])*u.AU, 
                        vel_vec=np.array([v_saturn.value,0,0])*u.km/u.s,
                        mass=95.16*c.M_earth)
Uranus=CelestialObjects(name='Uranus',
                        pos_vec=np.array([0,19.2,0])*u.AU, 
                        vel_vec=np.array([v_uranus.value,0,0])*u.km/u.s,
                        mass=14.54*c.M_earth)
Neptune=CelestialObjects(name='Neptune',
                        pos_vec=np.array([0,30,0])*u.AU, 
                        vel_vec=np.array([v_neptune.value,0,0])*u.km/u.s,
                        mass=17.15*c.M_earth)
Asteroid=CelestialObjects(name='Asteroid',
                       pos_vec=np.array([0,3.3,0])*u.AU,
                       vel_vec=np.array([v_asteroid.value,0,0])*u.km/u.s,
                       mass=59E-4*c.M_earth)
                       
bodies=[Sun,Mercury,Venus,Earth,Mars,Jupiter,Asteroid]
#making a class for system
class Simulation():
    def __init__(self,bodies,has_units=True):
        self.has_units=has_units
        self.bodies=bodies
        self.Nbodies=len(self.bodies)
        self.Ndim=6
        self.quant_vec=np.concatenate(np.array([i.return_vec() for i in self.bodies]))
        self.mass_vec=np.array([i.return_mass() for i in self.bodies])
        self.name_vec=[i.return_name() for i in self.bodies]
        
    def set_diff_eqs(self,calc_diff_eqs,**kwargs):
        self.diff_eqs_kwargs=kwargs
        self.calc_diff_eqs=calc_diff_eqs
        
    def kinetic_energy(self, vel):
           # Calculate kinetic energy given velocities
         v_squared = np.sum(vel**2)
         return 0.5 * np.sum(self.mass_vec * v_squared)
        
    
    def rk4(self,t,dt):
        k1= dt* self.calc_diff_eqs(t,self.quant_vec,self.mass_vec,**self.diff_eqs_kwargs)
        k2=dt*self.calc_diff_eqs(t+dt*0.5,self.quant_vec+0.5*k1,self.mass_vec,**self.diff_eqs_kwargs)
        k3=dt*self.calc_diff_eqs(t+dt*0.5,self.quant_vec+0.5*k2,self.mass_vec,**self.diff_eqs_kwargs)
        k4=dt*self.calc_diff_eqs(t+dt,self.quant_vec+k3,self.mass_vec,**self.diff_eqs_kwargs)
        
        y_new=self.quant_vec+((k1+2*k2+2*k3+k4)/6)
        return y_new
    
    def run(self,T,dt,t0=0):
        if not hasattr(self,'calc_diff_eqs'):
            raise AttributeError('You must set a diff eq solver first.')
        if self.has_units:
            try:
                _=t0.unit
            except:
                t0=(t0*T.unit).cgs.value
            T=T.cgs.value
            dt=dt.cgs.value
        
        self.history=[self.quant_vec]
        kinetic_energy_history = []
        clock_time=t0
        nsteps=int((T-t0)/dt)
        start_time=time.time()
        count=0
        orbit_start_times = {}
        orbit_periods = {}
        orbit_cross_times = {}

        for step in range(nsteps):
            sys.stdout.flush()
            sys.stdout.write('Integrating: step = {}/{}| Simulation Time = {}'.format(step, nsteps, round(clock_time, 3)) + '\r')
            y_new = self.rk4(0, dt)
            self.history.append(y_new)
            self.quant_vec = y_new
            clock_time += dt
            asteroid_vel = y_new[-3:]
            kinetic_energy = self.kinetic_energy(asteroid_vel)
            kinetic_energy_history.append(kinetic_energy)


            for h, body in enumerate(self.bodies):
                x_coord = self.quant_vec[h * 6]
                if np.any(x_coord >= 0) and h not in orbit_start_times:
                    orbit_start_times[h] = clock_time
                elif np.any(x_coord < 0) and h in orbit_start_times:
                    if h not in orbit_cross_times:
                        orbit_cross_times[h] = []
                    orbit_cross_times[h].append(clock_time)
                    if h in orbit_start_times:
                        orbit_start_times.pop(h)

        print("\nOrbital Periods:")
        for h, cross_times in orbit_cross_times.items():
            if len(cross_times) >= 2:
                orbit_periods_h = np.diff(cross_times)
                avg_period = np.mean(orbit_periods_h)
                print(f"{self.name_vec[h]}: {avg_period:.3f} seconds ({avg_period / u.year:.3f} years)")

        
        # Plotting the kinetic energy over time
        plt.figure()
        plt.plot(np.arange(nsteps) * dt, kinetic_energy_history, label='Kinetic Energy')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Kinetic Energy')
        plt.title('Kinetic Energy of Asteroid Over Time')
        plt.legend()
        plt.show()        

        runtime=time.time()-start_time
        print(clock_time)
        print('\n')
        print('Simulation completed in {} seconds'.format(runtime))
        self.history=np.array(self.history)
    
def nbody_solver(t,y,masses):
    N_bodies=int(len(y)/6)
    solved_vector=np.zeros(y.size)
    distance=[]
    for i in range(N_bodies):
        ioffset=i * 6
        for j in range(N_bodies):
            joffset=j * 6
            solved_vector[ioffset]=y[ioffset+3]
            solved_vector[ioffset+1]=y[ioffset+4]
            solved_vector[ioffset+2]=y[ioffset+5]
            if i != j:
                dx= y[ioffset]-y[joffset]
                dy=y[ioffset+1]-y[joffset+1]
                dz=y[ioffset+2]-y[joffset+2]
                r=(dx**2+dy**2+dz**2)**0.5
                ax=(-c.G.cgs*masses[j]/r**3)*dx
                ay=(-c.G.cgs*masses[j]/r**3)*dy
                az=(-c.G.cgs*masses[j]/r**3)*dz
                ax=ax.value
                ay=ay.value
                az=az.value
                solved_vector[ioffset+3]+=ax
                solved_vector[ioffset+4]+=ay
                solved_vector[ioffset+5]+=az
    return solved_vector

simulation=Simulation(bodies)
simulation.set_diff_eqs(nbody_solver)
simulation.run(80*u.year,1*u.day)

sun_position = simulation.history[:, :3] # Extracting position for Sun
mercury_position = simulation.history[:, 6:9]
venus_position = simulation.history[:, 12:15]
earth_position=simulation.history[:, 18:21]
mars_position=simulation.history[:, 24:27]
jupiter_position=simulation.history[:,30:33] 
#saturn_position=simulation.history[:, 36:39]
#uranus_position=simulation.history[:, 42:45]
#neptune_position=simulation.history[:, 48:51] 
asteroid_position=simulation.history[:, 36:39] 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectories
ax.plot(sun_position[:, 0], sun_position[:, 1], sun_position[:, 2], label='Sun')
ax.plot(mercury_position[:, 0], mercury_position[:, 1], mercury_position[:, 2], label='Mercury')
ax.plot(venus_position[:, 0], venus_position[:, 1], venus_position[:, 2], label='Venus')
ax.plot(earth_position[:, 0], earth_position[:, 1], earth_position[:, 2], label='Earth')
ax.plot(mars_position[:, 0], mars_position[:, 1], mars_position[:, 2], label='Mars')
ax.plot(jupiter_position[:, 0], jupiter_position[:, 1], jupiter_position[:, 2], label='Jupiter')
#ax.plot(saturn_position[:, 0], saturn_position[:, 1], saturn_position[:, 2], label='Saturn')
#ax.plot(uranus_position[:, 0], uranus_position[:, 1], uranus_position[:, 2], label='Uranus')
#ax.plot(neptune_position[:, 0], neptune_position[:, 1], neptune_position[:, 2], label='Neptune')
ax.plot(asteroid_position[:, 0], asteroid_position[:, 1], asteroid_position[:, 2], label='Asteroid')

# Add labels and title
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.set_title('Terrestrial Planets and Jupiter Orbit of the Sun')
ax.scatter([0], [0], [0], marker='o', color='yellow', s=50, label='Sun')  # Marking the Sun at the origin
ax.scatter(mercury_position[0, 0], mercury_position[0, 1], mercury_position[0, 2], marker='x', color='black', s=10, label='Mercury') 
ax.scatter(venus_position[0, 0], venus_position[0, 1], venus_position[0, 2], marker='x', color='orange', s=10, label='Venus') 
ax.scatter(earth_position[0, 0], earth_position[0, 1], earth_position[0, 2], marker='x', color='blue', s=10, label='Earth')  
ax.scatter(mars_position[0, 0], mars_position[0, 1], mars_position[0, 2], marker='x', color='red', s=10, label='Mars')  
ax.scatter(jupiter_position[0, 0], jupiter_position[0, 1], jupiter_position[0, 2], marker='o', color='green', s=20, label='Jupiter') 
#ax.scatter(saturn_position[0, 0], saturn_position[0, 1], saturn_position[0, 2], marker='o', color='red', s=6, label='Saturn')  
#ax.scatter(uranus_position[0, 0], uranus_position[0, 1], uranus_position[0, 2], marker='o', color='black', s=6, label='Uranus') 
#ax.scatter(neptune_position[0, 0], neptune_position[0, 1], neptune_position[0, 2], marker='o', color='blue', s=6, label='Neptune')
ax.scatter(asteroid_position[0, 0], asteroid_position[0, 1], asteroid_position[0, 2], marker='o', color='blue', s=10, label='Asteroid')

ax.legend()

# Show the plot
plt.show()