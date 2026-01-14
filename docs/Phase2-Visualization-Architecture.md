# Phase 2 Visualization Architecture: Data Flow & Components

## System Architecture Diagram

```
╔════════════════════════════════════════════════════════════════════════════╗
║                        PHASE 2: JEPA + VISUALIZATION                       ║
╚════════════════════════════════════════════════════════════════════════════╝

Phase 1 TLE history (hourly updates from Space-Track)
    ↓
Subphase 1: Propagation Pipeline (Orekit/GMAT)
    ├─ TLE → Cartesian → Propagate → Orbital elements
    └─ Output: fused_orbit_timeseries (PostgreSQL)
    ↓
Subphase 2: Synthetic maneuvers (50k sequences)
    ├─ GMAT/Orekit impulse model
    ├─ 10k per class (station-keeping, reboost, inc-change, decay, unknown)
    └─ Output: HDF5 dataset
    ↓
Subphase 3-5: Data prep + JEPA training
    ├─ Train/val/test split (70/15/15)
    ├─ JEPA encoder trained (>85% accuracy)
    └─ Output: model_v1.2.0.pth
    ↓
Subphase 6: CoALA integration
    ├─ /maneuvers endpoint (type, confidence, delta_a, delta_i)
    └─ Output: Real-time predictions in ml_predictions table
    ↓
**Subphase 7: ORBIT VISUALIZATION** ← NEW
    ├─ orbit_renderer.py: pre_orbit + post_orbit → GeoJSON
    ├─ cesium_formatter.py: GeoJSON + styling (green/yellow + opacity)
    ├─ API endpoints: /orbits/render/{norad_id}, /orbits/constellation/{const}
    ├─ Frontend: Cesium.js 3D globe, timeline, details modal
    └─ Output: Interactive 3D visualizations (< 100ms latency)
    ↓
Subphase 8: Monitoring
    ├─ Drift detection, retraining pipeline
    └─ Output: Continuous model improvement
```

Key files:
- agent/visualization/orbit_renderer.py
- agent/visualization/cesium_formatter.py
- agent/api/main.py (new /orbits/* endpoints)
- templates/visualization.html (Cesium.js UI)
