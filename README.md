# NonAxiSymmetrically Heated (NASH) Tube Stress
- steady-state temperature distribution (Gauss-Seidel iteration)
- biharmonic thermoelastic stress

As described in journal paper:
### Thermoelastic stress in concentrating solar receiver tubes: A retrospect on stress analysis methodology, and comparison of salt and sodium
Solar Energy 160 (2018) 368-379 - https://doi.org/10.1016/j.solener.2017.12.003

### Holms1952()
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

           Table II, p89: radius vs. tangential stress          

                     4.0 (in.):  126988.0387 (psi)
                     5.0 (in.):  69864.6296 (psi)
                     6.0 (in.):  35464.0469 (psi)
                     7.0 (in.):  11712.7526 (psi)
                     8.0 (in.): -6252.1660 (psi)
                     9.0 (in.): -20725.2239 (psi)
                    10.0 (in.): -32920.4657 (psi)
                    11.0 (in.): -43538.4796 (psi)
                    12.0 (in.): -53011.9613 (psi)
</code>

### SE6413()
<code>
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
                          Time:  0.1137 (sec)

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
                          Time:  0.0517 (sec)
</code>

### ASTRI2()
<code>
============== NPS Sch. 5S 3/4" N06625 at 650 degC =============

                             b:  12.7000 (mm)
                             a:  11.0500 (mm)
                             k:  19.1000 (kg*m/s^3/K)
                         alpha:  18.2000 (x1e6 K^-1)
                             E:  169.0000 (GPa)
                            nu:  0.3100 (-)
                            CG:  750.0000 (kW/m^2)
                          mdot:  0.1000 (kg/s)

                          Liquid Sodium                         

                             T:  888.0000 (K)
                           rho:  807.6339 (kg/m^3)
                            Cp:  1252.6951 (m^2/s^2/K)
                            mu:  203.3446 (x1e6 kg/m/s)
                         kappa:  58.8628 (kg*m/s^3/K)
                            Pr:  0.0043 (-)
                             U:  0.3228 (m/s)
                          mdot:  0.1000 (kg/s)
                            Re:  28332.5225 (-)
                            Pe:  122.6091 (-)
                        deltaP: -45.6350 (Pa/m)
                           HCR:  125.2695 (J/K/s)
                         h_int:  17395.9490 (W/m^2/K)
                            Bi:  1.5028 (-)

                    Generalised plane strain                    

                        Tbar_i:  903.7004 (K)
                          B'_1:  23.8937 (K)
                          D'_1: -0.0000 (K)
                        Tbar_o:  925.7961 (K)
                         B''_1:  57.8371 (K)
                         D''_1: -0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta: -0.0000 (MPa)
                   sigma_theta: -105.3066 (MPa)
                       sigma_z: -306.6446 (MPa)
                      sigma_Eq:  269.8679 (MPa)
                          Time:  0.0944 (sec)

      Generalised plane strain with annulled bending moment     

                        Tbar_i:  903.7004 (K)
                          B'_1:  23.8937 (K)
                          D'_1: -0.0000 (K)
                        Tbar_o:  925.7961 (K)
                         B''_1:  57.8371 (K)
                         D''_1: -0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta: -0.0000 (MPa)
                   sigma_theta: -105.3066 (MPa)
                       sigma_z: -169.0042 (MPa)
                      sigma_Eq:  147.8331 (MPa)

                    Determining peak flux...                    

                          Time:  27.7873 (sec)
</code>