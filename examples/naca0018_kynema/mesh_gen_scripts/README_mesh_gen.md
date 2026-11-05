# Mesh preparation

Use the original Pointwise Glyph recipe and the supplied analytic coordinates.
Set the point count to the count required by your chosen outline and the
qualified mesh recipe; check generated inlet/outlet labels and angle rotation.
The output directory must be new. Pointwise is an external licensed dependency.

```bash
python create_grids.py --airfoil NACA0018 --aoa 84 --numpts 335 \
  --pointwise /path/to/pointwise --output /path/to/new_mesh_case
```

The Glyph recipe retains the original mesh resolution and wall-spacing setup.
Generating a mesh is separate from the paper's mesh-sensitivity assessment.
