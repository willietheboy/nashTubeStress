# NonAxiSymmetrically Heated (NASH) Tube Stress
- steady-state temperature distribution (Gauss-Seidel iteration)
- biharmonic thermoelastic stress

As described in journal paper:
### Thermoelastic stress in concentrating solar receiver tubes: A retrospect on stress analysis methodology, and comparison of salt and sodium
Solar Energy 160 (2018) 368-379 - https://doi.org/10.1016/j.solener.2017.12.003

### Test output:

<code>
================== Holms (1952), NACA-TR-1059 ==================

                             E:  120.6583 (GPa)
                         alpha:  14.4000 (x1e6 K^-1)
                        Tbar_i:  0.0000 (K)
                          B'_1:  0.0000 (K)
                          D'_1:  0.0000 (K)
                        Tbar_o:  277.7778 (K)
                         B''_1:  555.5556 (K)
                         D''_1:  0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta:  0.0000 (MPa)
                   sigma_theta: -365.5046 (MPa)
                       sigma_z: -1234.2440 (MPa)
                      sigma_Eq:  1098.1029 (MPa)

================ NPS Sch. 5S 1" SS316 at 450degC ===============


                        HTC: 10e3 W/m^s/K                       

                        Tbar_i:  750.2038 (K)
                          B'_1:  41.8341 (K)
                          D'_1:  0.0000 (K)
                        Tbar_o:  771.5014 (K)
                         B''_1:  75.0273 (K)
                         D''_1: -0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta: -0.0000 (MPa)
                   sigma_theta: -102.1835 (MPa)
                       sigma_z: -378.2248 (MPa)
                      sigma_Eq:  338.8910 (MPa)
                          Time:  0.1126 (sec)

                        HTC: 40e3 W/m^s/K                       

                        Tbar_i:  729.9837 (K)
                          B'_1:  10.6652 (K)
                          D'_1: -0.0000 (K)
                        Tbar_o:  751.5017 (K)
                         B''_1:  44.3586 (K)
                         D''_1:  0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta:  0.0000 (MPa)
                   sigma_theta: -110.4248 (MPa)
                       sigma_z: -253.2035 (MPa)
                      sigma_Eq:  219.8765 (MPa)
                          Time:  0.0518 (sec)

============= NPS Sch. 5S 3/4" Inco625 at 650 degC =============

                             b:  12.7000 (mm)
                             a:  11.0500 (mm)
                             k:  19.1500 (kg*m/s^3/K)
                         alpha:  18.1500 (x1e6 K^-1)
                             E:  168.0000 (GPa)
                            nu:  0.3100 (-)
                            CG:  750.0000 (kW/m^2)
                          mdot:  0.2000 (kg/s)

                          Liquid Sodium                         

                             T:  888.0000 (K)
                           rho:  807.6339 (kg/m^3)
                            Cp:  1252.6951 (m^2/s^2/K)
                            mu:  203.3446 (x1e6 kg/m/s)
                         kappa:  58.8628 (kg*m/s^3/K)
                            Pr:  0.0043 (-)
                             U:  0.6456 (m/s)
                          mdot:  0.2000 (kg/s)
                        deltaP: -155.1224 (Pa/m)
                           HCR:  250.5390 (J/K/s)
                         h_int:  17502.1730 (W/m^2/K)
                            Bi:  1.5080 (-)

                    Generalised plane strain                    

                        Tbar_i:  903.5232 (K)
                          B'_1:  23.7157 (K)
                          D'_1:  0.0000 (K)
                        Tbar_o:  925.4909 (K)
                         B''_1:  57.5873 (K)
                         D''_1: -0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta: -0.0000 (MPa)
                   sigma_theta: -104.0399 (MPa)
                       sigma_z: -302.4280 (MPa)
                      sigma_Eq:  266.1248 (MPa)
                          Time:  0.0507 (sec)

      Generalised plane strain with annulled bending moment     

                        Tbar_i:  903.5232 (K)
                          B'_1:  23.7157 (K)
                          D'_1:  0.0000 (K)
                        Tbar_o:  925.4909 (K)
                         B''_1:  57.5873 (K)
                         D''_1: -0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta: -0.0000 (MPa)
                   sigma_theta: -104.0399 (MPa)
                       sigma_z: -166.6801 (MPa)
                      sigma_Eq:  145.8259 (MPa)
</code>