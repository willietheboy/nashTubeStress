# NonAxiSymmetrically Heated (NASH) Tube Stress
- steady-state temperature distribution (Gauss-Seidel iteration)
- biharmonic thermoelastic stress

As described in journal paper:
### Thermoelastic stress in concentrating solar receiver tubes: A retrospect on stress analysis methodology, and comparison of salt and sodium
Solar Energy 160 (2018) 368-379 - https://doi.org/10.1016/j.solener.2017.12.003

### Test output:

<code>
====================== HTC : 10e3 W/m^s/K ======================

                        Tbar_i:  750.2038 (K)
                          B'_1:  41.8341 (K)
                          D'_1:  0.0000 (K)
                        Tbar_o:  771.5014 (K)
                         B''_1:  75.0273 (K)
                         D''_1: -0.0000 (K)
                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta: -0.0000 (MPa)
                   sigma_theta: -102.1835 (MPa)
                       sigma_z: -378.2248 (MPa)
                 max(sigma_Eq):  338.8910 (MPa)
                          Time:  0.1161 (sec)

====================== HTC : 40e3 W/m^s/K ======================

                        Tbar_i:  729.9837 (K)
                          B'_1:  10.6652 (K)
                          D'_1: -0.0000 (K)
                        Tbar_o:  751.5017 (K)
                         B''_1:  44.3586 (K)
                         D''_1:  0.0000 (K)
                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta:  0.0000 (MPa)
                   sigma_theta: -110.4248 (MPa)
                       sigma_z: -253.2035 (MPa)
                 max(sigma_Eq):  219.8765 (MPa)
                          Time:  0.0520 (sec)

=================== ASTRI 2.0 REFERENCE CASE ===================


                       Inco625 at 650 degC                      

                             b:  12.7000 (mm)
                             a:  11.0500 (mm)
                             k:  19.1500 (kg*m/s^3/K)
                         alpha:  18.1500 (x1e6 K^-1)
                             E:  168.0000 (GPa)
                            nu:  0.3100 (-)
                            CG:  750.0000 (kW/m^2)
                          mdot:  0.2000 (kg/s)

                          Liquid Sodium                         

                             T:  887.0000 (K)
                           rho:  807.8710 (kg/m^3)
                            Cp:  1252.7438 (m^2/s^2/K)
                            mu:  203.5792 (x1e6 kg/m/s)
                         kappa:  58.9065 (kg*m/s^3/K)
                            Pr:  0.0043 (-)
                             U:  0.6454 (m/s)
                          mdot:  0.2000 (kg/s)
                        deltaP: -155.1172 (Pa/m)
                           HCR:  250.5488 (J/K/s)
                         h_int:  17512.4682 (W/m^2/K)
                            Bi:  1.5089 (-)

                    Generalised plane strain                    

                        Tbar_i:  902.5141 (K)
                          B'_1:  23.7020 (K)
                          D'_1: -0.0000 (K)
                        Tbar_o:  924.4818 (K)
                         B''_1:  57.5737 (K)
                         D''_1:  0.0000 (K)
                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta:  0.0000 (MPa)
                   sigma_theta: -104.0440 (MPa)
                       sigma_z: -302.3736 (MPa)
                 max(sigma_Eq):  266.0729 (MPa)
                          Time:  0.0511 (sec)

      Generalised plane strain with annulled bending moment      

                        Tbar_i:  902.5141 (K)
                          B'_1:  23.7020 (K)
                          D'_1: -0.0000 (K)
                        Tbar_o:  924.4818 (K)
                         B''_1:  57.5737 (K)
                         D''_1:  0.0000 (K)
                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta:  0.0000 (MPa)
                   sigma_theta: -104.0440 (MPa)
                       sigma_z: -166.6702 (MPa)
                 max(sigma_Eq):  145.8186 (MPa)
</code>