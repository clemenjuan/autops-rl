#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import math
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from satellites import TargetSatellite, ObserverSatellite

def create_orbit_trace(semimajor_axis, inclination_deg, raan_deg, num_points=100):
    """Create complete orbital trace for one plane using actual orbital parameters"""
    inc_rad = np.radians(inclination_deg)
    raan_rad = np.radians(raan_deg)
    
    orbit_points = []
    for angle in np.linspace(0, 360, num_points):
        angle_rad = np.radians(angle)
        
        # Position in orbital plane (semimajor_axis is already in km)
        x_orbit = semimajor_axis * np.cos(angle_rad)
        y_orbit = semimajor_axis * np.sin(angle_rad)
        z_orbit = 0
        
        # Apply rotations for inclination and RAAN
        x_incl = x_orbit
        y_incl = y_orbit * np.cos(inc_rad) - z_orbit * np.sin(inc_rad)
        z_incl = y_orbit * np.sin(inc_rad) + z_orbit * np.cos(inc_rad)
        
        # Apply RAAN rotation (no unit conversion needed)
        x_final = x_incl * np.cos(raan_rad) - y_incl * np.sin(raan_rad)
        y_final = x_incl * np.sin(raan_rad) + y_incl * np.cos(raan_rad)
        z_final = z_incl
        
        orbit_points.append([x_final, y_final, z_final])
    
    return np.array(orbit_points)

def visualize_actual_constellation():
    """Show the EXACT constellation used in your simulations using actual satellite classes"""
    
    # Use the exact same parameters as your simulation
    num_targets = 100
    num_observers = 20
    
    # Create satellites EXACTLY like the simulator does (no workarounds needed now)
    print("Creating constellation: 20 observers + 100 targets")
    
    # Create target satellites first 
    target_satellites = [TargetSatellite(name=f"Target-{i+1}") for i in range(num_targets)]
    
    # Create observer satellites with proper Walker Delta formation
    observer_satellites = [ObserverSatellite(name=f"Observer-{i+1}", 
                                           num_targets=num_targets, 
                                           num_observers=num_observers)
                          for i in range(num_observers)]
    
    # Propagate orbits like the simulator does
    print("Propagating orbits...")
    for sat in target_satellites + observer_satellites:
        sat.propagate_orbit(0)
    
    # Debug: Print some orbital parameters before and after propagation
    print(f"Debug - First observer satellite:")
    obs = observer_satellites[0]
    print(f"  Semimajor axis: {obs.orbit['semimajoraxis']:.1f} km")
    print(f"  Position: ({obs.orbit['x']:.1f}, {obs.orbit['y']:.1f}, {obs.orbit['z']:.1f}) km")
    print(f"  Radius: {obs.orbit['radius']:.1f} km")
    print(f"  Altitude: {obs.orbit['radius'] - 6378.137:.1f} km")
    
    # Extract positions
    target_positions = np.array([[sat.orbit['x'], sat.orbit['y'], sat.orbit['z']] for sat in target_satellites])
    observer_positions = np.array([[sat.orbit['x'], sat.orbit['y'], sat.orbit['z']] for sat in observer_satellites])
    
    # Calculate altitude ranges for debugging
    target_radii = np.array([sat.orbit['radius'] for sat in target_satellites])
    observer_radii = np.array([sat.orbit['radius'] for sat in observer_satellites])
    
    earth_radius = 6378.137  # km
    target_altitudes = target_radii - earth_radius
    observer_altitudes = observer_radii - earth_radius
    
    print(f"Target altitude range: {target_altitudes.min():.0f} to {target_altitudes.max():.0f} km")
    print(f"Observer altitude range: {observer_altitudes.min():.0f} to {observer_altitudes.max():.0f} km")
    
    # Create 3D plot
    import matplotlib
    matplotlib.use('TkAgg')
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # NO EARTH - just the constellation
    
    # Plot targets (small gray dots)
    ax.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 
              c='gray', s=20, alpha=0.6, label=f'{num_targets} Targets')
    
    # Plot Walker Delta orbital planes FIRST
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    plane_labels = ['Plane 1', 'Plane 2', 'Plane 3', 'Plane 4', 'Plane 5']
    
    print("Drawing Walker Delta orbital planes...")
    # Group satellites by their actual RAAN values (this determines the orbital plane)
    unique_planes = {}
    for i, sat in enumerate(observer_satellites):
        # Round RAAN to nearest degree to group planes properly
        raan_rounded = round(sat.orbit['raan'])
        
        if raan_rounded not in unique_planes:
            unique_planes[raan_rounded] = {
                'semimajoraxis': sat.orbit['semimajoraxis'],
                'inclination': sat.orbit['inclination'], 
                'raan': sat.orbit['raan'],
                'satellites': [],
                'color_idx': len(unique_planes)  # Assign color index based on order discovered
            }
        unique_planes[raan_rounded]['satellites'].append(i+1)
    
    # Sort planes by RAAN for consistent coloring
    sorted_planes = sorted(unique_planes.items())
    
    # Draw orbital traces for each plane
    for plane_idx, (raan_key, params) in enumerate(sorted_planes):
        orbit_trace = create_orbit_trace(
            params['semimajoraxis'],
            params['inclination'], 
            params['raan']
        )
        
        color_idx = plane_idx % len(colors)
        ax.plot(orbit_trace[:, 0], orbit_trace[:, 1], orbit_trace[:, 2], 
               color=colors[color_idx], alpha=0.4, linewidth=2, linestyle='--',
               label=f'Plane {plane_idx+1} (RAAN {params["raan"]:.0f}°)')
    
    # Plot observers by their actual plane assignment
    for i in range(num_observers):
        sat = observer_satellites[i]
        raan_rounded = round(sat.orbit['raan'])
        
        # Find which plane this satellite belongs to
        plane_idx = next(idx for idx, (raan_key, _) in enumerate(sorted_planes) if raan_key == raan_rounded)
        color_idx = plane_idx % len(colors)
        
        ax.scatter(observer_positions[i, 0], observer_positions[i, 1], observer_positions[i, 2], 
                  c=colors[color_idx], s=200, alpha=1.0, edgecolor='black', linewidth=2)
    
    # Add observer labels only
    for i in range(num_observers):
        ax.text(observer_positions[i, 0]*1.1, observer_positions[i, 1]*1.1, observer_positions[i, 2]*1.1, 
               f'O{i+1}', fontsize=8, fontweight='bold')
    
    # Styling
    ax.set_xlabel('X (km)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (km)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (km)', fontsize=12, fontweight='bold')
    ax.set_title(f'Space Situational Awareness Example Scenario\n{num_observers} Observer Satellites + {num_targets} Target Objects', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    # Set proper scale
    all_positions = np.vstack([target_positions, observer_positions])
    max_range = np.max(np.abs(all_positions)) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Print simulation info
    print(f"\nSimulation Constellation:")
    print(f"- {num_observers} Observer satellites (Walker Delta)")
    print(f"- {num_targets} Target satellites (random orbits)")
    print(f"- Detection range: {observer_satellites[0].max_distance/1000:.0f} km")
    
    # Print plane distribution with actual orbital parameters
    for plane_idx, (raan_key, params) in enumerate(sorted_planes):
        print(f"- Plane {plane_idx+1}: Observers {params['satellites']}")
        print(f"  RAAN: {params['raan']:.1f}°, Inc: {params['inclination']:.1f}°, SMA: {params['semimajoraxis']:.0f} km")
    
    plt.show()

if __name__ == "__main__":
    visualize_actual_constellation() 