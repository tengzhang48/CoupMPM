# Theory Manual

This manual documents the mathematical foundations of CoupMPM. Each section covers a
distinct physical or numerical algorithm implemented in the package.

## Contents

| Section | Topic |
| --- | --- |
| [Core MPM & APIC](theory_apic_transfer.md) | Material Point Method formulation and Affine Particle-In-Cell transfers |
| [B-Bar Anti-Locking](theory_bbar_locking.md) | Volumetric anti-locking via the B-bar projection method |
| [Multi-Body Contact](theory_bardenhagen_contact.md) | Bardenhagen–Kober impenetrability algorithm with friction and adhesion |
| [Dynamic Cohesive Zones](theory_dynamic_cohesive.md) | Runtime bond formation and rupture with traction-separation laws |
| [Particle Adaptivity](theory_adaptivity.md) | Jacobian-based particle splitting and nearest-neighbour merging |
