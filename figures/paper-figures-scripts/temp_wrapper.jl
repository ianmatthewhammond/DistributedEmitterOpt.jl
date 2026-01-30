
# Wrapper to bridge old plot_geometry_for_inset calls to new save_geometry_snapshot
function plot_geometry_for_inset(np, pv, mesh, path; scalar_field="p", color="gray", flipy=true, window_size=(300, 300))
    if isnothing(mesh)
        return
    end
    xl, xr, yl, yr, _, _ = mesh.bounds
    save_geometry_snapshot(np, pv, mesh, path, xr - xl, yr - yl;
        color=color, flipy=flipy, window_size=window_size, design_field=scalar_field,
        axes_viewport=(0.66, 0.0, 1.0, 0.34))
end
