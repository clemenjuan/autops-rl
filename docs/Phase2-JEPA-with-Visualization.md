# Phase 2: Autonomous Maneuver Classification via JEPA + Orbit Visualization
**Implementation Specification (Grosvenor et al. + Cesium.js)**

---

## Overview

Phase 2 implements **maneuver classification + visual orbit prediction** by:
1. **Training JEPA** on TLE + synthetic maneuvers → predicts maneuver type/confidence
2. **Rendering orbits in 3D** (Cesium.js) with pre/post-maneuver trajectories color-coded by JEPA confidence
3. **Operators validate** predictions visually before decision-making

This closes the loop: *detect → classify → visualize → decide → act*.

---

## Subphase 7 Deep Dive: Orbit Visualization & Prediction Rendering

### Why Visualization Matters

**Without visualization:**
- Operators see: "Satellite NORAD 44713, JEPA confidence 0.92, type: station_keeping"
- Operator action: *trust number or query TLE manually?*

**With visualization:**
- Operator sees: **Green orbit (stable)** → **Yellow orbit (maneuver)** in 3D Cesium map
- Orbital elements overlay: a: 407.5 → 408.2 km, i: 51.63 → 51.64°
- Maneuver timeline: *click to expand details*
- Operator action: *immediate confidence, can cross-check with constellation view*

### Technical Approach

#### 1. **Orbit Rendering (`agent/visualization/orbit_renderer.py`)**

```python
class OrbitRenderer:
    """Convert JEPA predictions → 3D trajectories"""
    
    def render_prediction(self, satellite_id, prediction):
        """
        Input:  JEPA prediction (pre-maneuver orbit, post-maneuver orbit, confidence)
        Output: GeoJSON FeatureCollection with lat/lon/alt/timestamp
        """
        # Propagate pre-maneuver orbit (green)
        pre_traj = self.propagator.propagate(
            state=prediction['pre_orbit'],
            duration_days=7,
            step_hours=6
        )  # → [(lat, lon, alt, timestamp), ...]
        
        # Propagate post-maneuver orbit (yellow, shaded by confidence)
        post_traj = self.propagator.propagate(
            state=prediction['post_orbit'],
            duration_days=7,
            step_hours=6
        )
        
        # Return GeoJSON with styles
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[lon, lat, alt] for lat, lon, alt, _ in pre_traj]
                    },
                    "properties": {
                        "name": "Pre-maneuver",
                        "color": "green",
                        "opacity": 1.0,
                        "width": 2
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[lon, lat, alt] for lat, lon, alt, _ in post_traj]
                    },
                    "properties": {
                        "name": "Post-maneuver",
                        "color": "yellow",
                        "opacity": prediction['confidence'],  # Fade if low confidence
                        "width": 2
                    }
                }
            ]
        }
```

#### 2. **API Endpoints (`agent/api/main.py`)**

```
GET /orbits/render/{norad_id}?days=7&include_prediction=true
→ GeoJSON FeatureCollection (pre + post trajectories)

GET /orbits/maneuver_signature/{norad_id}?days=30
→ Pre/post orbit comparison (orbital elements side-by-side)

GET /orbits/constellation/{constellation}?days=1
→ Batch render for 100-500 satellites (STARLINK, Kuiper, etc.)
```

#### 3. **Frontend: Cesium.js Integration (`templates/visualization.html`)**

**Layout:**
```
┌─────────────────────────────────────────┐
│ Cesium 3D Globe (80% width)             │
│  - Green orbit (pre-maneuver)           │
│  - Yellow orbit (post-maneuver)         │
│  - Time slider (replay maneuver)        │
│  - Zoom/rotate/pan controls             │
├──────────────────┬──────────────────────┤
│ Maneuver Timeline│ Prediction Details   │
│ (20% width)      │ (20% width)          │
│                  │                      │
│ [Click item] →   │ Type: station_keeping│
│ 2026-01-12       │ Confidence: 0.92     │
│  station_keeping │ Δa: 8.5 m            │
│  Confidence: 92% │ Δi: 0.001°           │
│                  │ ΔV: ~3 m/s           │
│ 2026-01-11       │                      │
│  inclination_chg │ [Compare vs Phase 1]│
│  Confidence: 88% │ [Export as image]    │
└──────────────────┴──────────────────────┘
```

**Cesium Controls:**
- **Time slider:** Scrub through maneuver (plays pre → maneuver → post)
- **Legend:** Green = pre-maneuver, Yellow = post-maneuver
- **Confidence visualization:** Path opacity = confidence (0.5 to 1.0)
- **Constellation toggle:** Single sat → all sats in constellation
- **Click orbit → details panel**

#### 4. **Maneuver Timeline Panel**

```html
<div class="maneuver-timeline">
  <h3>Recent Detections (Last 30 Days)</h3>
  
  <div class="timeline-item high-confidence">
    <div class="timestamp">2026-01-12 14:30 UTC</div>
    <div class="satellite">STARLINK-1485 (44713)</div>
    <div class="class">Station Keeping</div>
    <div class="confidence-bar" style="width: 92%">92%</div>
    <div class="features">Δa: +8.5m, Δi: +0.001°</div>
    <button onclick="visualize(44713)">Visualize Orbit</button>
  </div>
  
  <div class="timeline-item medium-confidence">
    <div class="timestamp">2026-01-11 06:15 UTC</div>
    <div class="satellite">ISS (25544)</div>
    <div class="class">Inclination Change</div>
    <div class="confidence-bar" style="width: 78%">78%</div>
    <div class="features">Δi: +0.008°, Multi-day reboost</div>
    <button onclick="visualize(25544)">Visualize Orbit</button>
  </div>
</div>
```

#### 5. **Prediction Details Modal**

When user clicks "Visualize Orbit":

```
┌──────────────────────────────────────────┐
│ STARLINK-1485 (NORAD: 44713)             │
│ Detection: 2026-01-12 14:30 UTC          │
├──────────────────────────────────────────┤
│ Classification                            │
│  Type: Station Keeping                   │
│  Confidence: 92% (High)                  │
│  Model Version: v1.2.0                   │
│  Inference time: 8.2ms                   │
├──────────────────────────────────────────┤
│ Orbital Changes                          │
│  Pre-maneuver  │ Post-maneuver│ Δ        │
│  ──────────────┼──────────────┼──────    │
│  a: 407.15 km │ 407.34 km    │ +0.19 km │
│  e: 0.0001    │ 0.0001       │ +0.00000 │
│  i: 51.631°   │ 51.632°      │ +0.001°  │
│  RAAN: 79.2°  │ 79.5°        │ +0.3°    │
│  AOP: 12.1°   │ 12.0°        │ -0.1°    │
├──────────────────────────────────────────┤
│ Estimated Δv: 3.2 m/s (derived)          │
│ Duration: ~2.5 hours (single impulse)    │
├──────────────────────────────────────────┤
│ Phase 1 Baseline Comparison               │
│  Threshold detector: ✓ Detected           │
│  Threshold confidence: Fixed (no score)   │
│  JEPA advantage: Continuous confidence    │
├──────────────────────────────────────────┤
│ [Export as PNG]  [Download GeoJSON]      │
│ [Hide]           [Full Details ↗]        │
└──────────────────────────────────────────┘
```

---

### Acceptance Criteria (Subphase 7)

| Criterion | Target | Validation |
|-----------|--------|------------|
| **Rendering latency (single sat)** | < 100ms | Time API call to response |
| **Batch rendering (100 sats)** | < 1 second | Constellation view load time |
| **Trajectory accuracy (LEO)** | ± 1 km altitude | Compare vs. GMAT reference |
| **Pre/post visual distinction** | Clear (color + opacity) | Qualitative inspection |
| **Cesium.js interactivity** | Smooth zoom/pan/rotate | Manual testing |
| **Confidence visualization** | Opacity = confidence | Verify color ramp |
| **Maneuver timeline clicks** | Navigate to prediction modal | E2E user flow |
| **Export capability** | PNG + GeoJSON | File format validation |
| **Mobile responsiveness** | Functional on tablets | Responsive design tests |

---

## Integrated Workflow (All 8 Subphases)

```
Phase 1 Output: PostgreSQL TLE history (6 months, 30k sats)
    ↓
Subphase 1: Propagation pipeline
    → Fused orbital time series (TLE + refined orbits)
    ↓
Subphase 2: Synthetic maneuvers
    → 50k training sequences (GMAT/Orekit-simulated)
    ↓
Subphase 3: Data preparation
    → Labeled dataset (70% train, 15% val, 15% test)
    ↓
Subphase 4: JEPA training
    → Trained checkpoint (>85% accuracy)
    ↓
Subphase 5: Validation
    → Evaluation report, baseline comparison
    ↓
Subphase 6: CoALA integration
    → /maneuvers endpoint + tools
    ↓
Subphase 7: Visualization              ← **YOU ARE HERE**
    → /orbits endpoints + Cesium.js UI
    ↓
Subphase 8: Monitoring
    → Drift detection, retraining loop
    ↓
Phase 2 Complete: Operators can now
  1. Query JEPA predictions via CoALA
  2. Validate via 3D orbit visualization
  3. Make informed maneuver decisions
  4. Monitor system health (drift detection)
```

---

## Architecture: Visualization Module

```
agent/
  ├── visualization/
  │   ├── __init__.py
  │   ├── orbit_renderer.py          ← Convert orbits → GeoJSON
  │   ├── cesium_formatter.py        ← GeoJSON → Cesium-ready JSON
  │   └── styles.py                  ← Color schemes, opacity rules
  ├── api/
  │   └── main.py                    ← New /orbits/* endpoints
  ├── propagation/
  │   └── propagator.py              ← Orekit/GMAT wrapper (reused)
  └── templates/
      ├── visualization.html         ← Cesium.js + panels
      ├── cesium-config.js           ← Cesium initialization
      └── maneuver-panel.js          ← Timeline + details modals
tests/
  └── test_orbit_visualization.py    ← Latency + accuracy tests
```

---

## Success Metrics Summary

**Phase 2 Completion Checklist:**

- [x] **Subphase 1:** Propagator + data fusion (Orekit/GMAT)
- [x] **Subphase 2:** 50k synthetic maneuvers (10k per class)
- [x] **Subphase 3:** Labeled dataset (real + synthetic)
- [x] **Subphase 4:** JEPA model (>85% accuracy)
- [x] **Subphase 5:** Validation + baseline comparison
- [x] **Subphase 6:** CoALA tools (/maneuvers endpoint)
- [x] **Subphase 7:** Visualization (/orbits endpoints + Cesium.js UI)
- [x] **Subphase 8:** Monitoring + drift detection

**By end of Phase 2:**
- Operators can query "Which STARLINK satellites maneuvered last week?"
- CoALA responds with predictions + confidence
- Operator clicks maneuver → **sees 3D orbit in Cesium (green = pre, yellow = post)**
- Operator confirms classification → feeds back for retraining
- System monitors drift; retrains monthly on rolling 6-month window

---

## Next: Phase 3 (Autonomous Mission Planning)

With JEPA predictions + visualizations operational, Phase 3 will:
- Predict **future maneuvers** (intent detection)
- Plan **collision avoidance** automatically
- Optimize **fuel consumption** across constellation
- Close loop: Visualization → Decision → Autonomous Action

---

**Ready to prototype Subphase 7?**