# NonAxiSymmetrically Heated (NASH) Tube Stress
- steady-state temperature distribution (Gauss-Seidel iteration)
- biharmonic thermoelastic stress

As described in journal paper:
### Thermoelastic stress in concentrating solar receiver tubes: A retrospect on stress analysis methodology, and comparison of salt and sodium
Solar Energy 160 (2018) 368-379 - https://doi.org/10.1016/j.solener.2017.12.003

### Test output:

<code>
====================== HTC : 10e3 W/m^s/K ======================

                        Tbar_i:  750.204 (K)
                          B'_1:  41.834 (K)
                          D'_1:  0.000 (K)
                        Tbar_o:  771.501 (K)
                         B''_1:  75.027 (K)
                         D''_1: -0.000 (K)
                       sigma_r:  0.000 (MPa)
                  sigma_rTheta: -0.000 (MPa)
                   sigma_theta: -102.184 (MPa)
                       sigma_z: -378.225 (MPa)
                 max(sigma_Eq):  338.891 (MPa)
                          Time:  0.128 (sec)

====================== HTC : 40e3 W/m^s/K ======================

                        Tbar_i:  729.984 (K)
                          B'_1:  10.665 (K)
                          D'_1: -0.000 (K)
                        Tbar_o:  751.502 (K)
                         B''_1:  44.359 (K)
                         D''_1:  0.000 (K)
                       sigma_r:  0.000 (MPa)
                  sigma_rTheta:  0.000 (MPa)
                   sigma_theta: -110.425 (MPa)
                       sigma_z: -253.203 (MPa)
                 max(sigma_Eq):  219.877 (MPa)
                          Time:  0.052 (sec)

=================== ASTRI 2.0 REFERENCE CASE ===================


                       Inco625 at 650 degC                      

                             b:  12.700 (mm)
                             a:  11.050 (mm)
                             k:  19.150 (kg*m/s^3/K)
                         alpha:  18.815 (x1e6 K^-1)
                             E:  168.000 (GPa)
                            nu:  0.310 (-)
                            CG:  750.000 (kW/m^2)
                          mdot:  0.200 (kg/s)

                          Liquid Sodium                         

                             T:  887.000 (K)
                           rho:  807.871 (kg/m^3)
                            Cp:  1252.744 (m^2/s^2/K)
                            mu:  0.000 (kg/m/s)
                         kappa:  58.907 (kg*m/s^3/K)
                            Pr:  0.004 (-)
                             U:  0.645 (m/s)
                          mdot:  0.200 (kg/s)
                        deltaP: -155.117 (Pa/m)
                           HCR:  250.549 (J/K/s)
                         h_int:  17512.468 (W/m^2/K)
                            Bi:  1.509 (-)

                 Analytical thermoelastic stress                

                        Tbar_i:  902.514 (K)
                          B'_1:  23.702 (K)
                          D'_1: -0.000 (K)
                        Tbar_o:  924.482 (K)
                         B''_1:  57.574 (K)
                         D''_1:  0.000 (K)
                       sigma_r:  0.000 (MPa)
                  sigma_rTheta:  0.000 (MPa)
                   sigma_theta: -107.856 (MPa)
                       sigma_z: -313.452 (MPa)
                 max(sigma_Eq):  275.822 (MPa)
                          Time:  0.052 (sec)
</code>