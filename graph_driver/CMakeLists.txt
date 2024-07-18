add_tool_target (xrays)

add_test (
    NAME xrays_test
    COMMAND $<TARGET_PROPERTY:xrays,BINARY_DIR>/xrays --absorption_model=weak_damping
                                                      --dispersion=ordinary_wave
                                                      --endtime=2.0
                                                      --equilibrium=efit
                                                      --equilibrium_file=${CMAKE_CURRENT_SOURCE_DIR}/../graph_tests/efit.nc
                                                      --init_kx_mean=-700.0
                                                      --init_ky_dist=normal
                                                      --init_ky_mean=-100.0
                                                      --init_ky_sigma=10.0
                                                      --init_kz_dist=normal
                                                      --init_kz_mean=0.0
                                                      --init_kz_sigma=10.0
                                                      --init_w_dist=normal
                                                      --init_w_mean=700
                                                      --init_w_sigma=10.0
                                                      --init_x_mean=1.0
                                                      --init_y_dist=normal
                                                      --init_y_mean=0.0
                                                      --init_y_sigma=0.05
                                                      --init_z_dist=normal
                                                      --init_z_mean=0.0
                                                      --init_z_sigma=0.05
                                                      --num_rays=1
                                                      --num_times=100000
                                                      --solver=rk4
                                                      --sub_steps=100
                                                      --use_cyl_xy
)
