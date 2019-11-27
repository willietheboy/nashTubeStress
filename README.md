# NonAxiSymmetrically Heated (NASH) Tube Stress
- steady-state temperature distribution (Gauss-Seidel iteration)
- biharmonic thermoelastic stress

As described in journal paper:
### Thermoelastic stress in concentrating solar receiver tubes: A retrospect on stress analysis methodology, and comparison of salt and sodium
Solar Energy 160 (2018) 368-379 - https://doi.org/10.1016/j.solener.2017.12.003

### Timoshenko1951()
<code>
=========== Case 135 p412 Timoshenko & Goodier, 1951 ===========

                             E:  200.0000 (GPa)
                         alpha:  10.0000 (x1e6 K^-1)
                        Tbar_i:  0.0000 (K)
                          B'_1:  0.0000 (K)
                          D'_1:  0.0000 (K)
                        Tbar_o:  100.0000 (K)
                         B''_1: -0.0000 (K)
                         D''_1:  0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta:  0.0000 (MPa)
                   sigma_theta: -126.9543 (MPa)
                       sigma_z: -126.9543 (MPa)
                      sigma_Eq:  126.9543 (MPa)
</code>

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
=============== NPS Sch. 5S 1" S31609 at 450degC ===============


                          Nitrate Salt                          

                             T:  723.1500 (K)
                           rho:  1803.8000 (kg/m^3)
                            Cp:  1520.4000 (m^2/s^2/K)
                            mu:  1472.4232 (x1e6 kg/m/s)
                         kappa:  0.5285 (kg*m/s^3/K)
                            Pr:  4.2359 (-)
                             U:  3.8960 (m/s)
                          mdot:  5.0000 (kg/s)
                            Re:  143651.3945 (-)
                            Pe:  608492.6747 (-)
                        deltaP: -7589.5960 (Pa/m)
                           HCR:  7602.0000 (J/K/s)
                         h_int:  9613.0515 (W/m^2/K)
                            Bi:  0.7936 (-)

                   HTC: 9613.05152527 W/m^s/K                   

                        Tbar_i:  750.0697 (K)
                          B'_1:  44.9933 (K)
                          D'_1:  0.0000 (K)
                        Tbar_o:  770.4418 (K)
                         B''_1:  79.3203 (K)
                         D''_1: -0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta: -0.0000 (MPa)
                   sigma_theta: -101.7681 (MPa)
                       sigma_z: -389.8666 (MPa)
                      sigma_Eq:  350.2524 (MPa)
                          Time:  0.1098 (sec)

                          Liquid Sodium                         

                             T:  723.1500 (K)
                           rho:  846.2179 (kg/m^3)
                            Cp:  1272.2439 (m^2/s^2/K)
                            mu:  254.4561 (x1e6 kg/m/s)
                         kappa:  66.7702 (kg*m/s^3/K)
                            Pr:  0.0048 (-)
                             U:  6.6437 (m/s)
                          mdot:  4.0000 (kg/s)
                            Re:  664996.9491 (-)
                            Pe:  3224.1846 (-)
                        deltaP: -7742.9078 (Pa/m)
                           HCR:  5088.9755 (J/K/s)
                         h_int:  43402.8029 (W/m^2/K)
                            Bi:  3.5829 (-)

                   HTC: 43402.8029066 W/m^s/K                   

                        Tbar_i:  729.1806 (K)
                          B'_1:  10.1771 (K)
                          D'_1:  0.0000 (K)
                        Tbar_o:  749.7852 (K)
                         B''_1:  45.0594 (K)
                         D''_1: -0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta: -0.0000 (MPa)
                   sigma_theta: -110.9377 (MPa)
                       sigma_z: -252.2070 (MPa)
                      sigma_Eq:  218.9436 (MPa)
                          Time:  0.0475 (sec)
</code>

### ASTRI2()
<code>
=========== 25.4mm OD x 1.65mm WT N06625 at 650 degC ===========

                             b:  12.7000 (mm)
                             a:  11.0500 (mm)
                             k:  19.1000 (kg*m/s^3/K)
                         alpha:  18.2000 (x1e6 K^-1)
                             E:  169.0000 (GPa)
                           TSP:  161.0366 (kPa.m/W)
                            nu:  0.3100 (-)
                            CG:  765.0000 (kW/m^2)
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
                   dead-weight:  1.3489 (kg/m)
                           P_i:  6.0000 (bar (x1e-5 Pa))

            Generalised plane strain (pressure only)            

                        Tbar_i:  888.0000 (K)
                          B'_1:  0.0000 (K)
                          D'_1: -0.0000 (K)
                        Tbar_o:  888.0000 (K)
                         B''_1:  0.0000 (K)
                         D''_1: -0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta:  0.0000 (MPa)
                   sigma_theta:  3.7390 (MPa)
                       sigma_z:  0.0000 (MPa)
                      sigma_Eq:  3.7390 (MPa)

             Generalised plane strain (thermal only)            

                        Tbar_i:  904.0144 (K)
                          B'_1:  24.3716 (K)
                          D'_1: -0.0000 (K)
                        Tbar_o:  926.5520 (K)
                         B''_1:  58.9939 (K)
                         D''_1:  0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta:  0.0000 (MPa)
                   sigma_theta: -107.4128 (MPa)
                       sigma_z: -312.7775 (MPa)
                      sigma_Eq:  275.2653 (MPa)
                          Time:  0.0951 (sec)

      Generalised plane strain with annulled bending moment     

                        Tbar_i:  904.0144 (K)
                          B'_1:  24.3716 (K)
                          D'_1: -0.0000 (K)
                        Tbar_o:  926.5520 (K)
                         B''_1:  58.9939 (K)
                         D''_1:  0.0000 (K)

                  Stress at outside tube crown:                 

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta:  0.0000 (MPa)
                   sigma_theta: -107.4128 (MPa)
                       sigma_z: -172.3842 (MPa)
                      sigma_Eq:  150.7898 (MPa)

                Determining peak flux for N06625                

                    Time taken:  23.2500 (sec)

=========== 33.4mm OD x 1.32mm WT N06320 at 650 degC ===========

                             b:  16.7000 (mm)
                             a:  15.3800 (mm)
                             k:  21.4000 (kg*m/s^3/K)
                         alpha:  16.5000 (x1e6 K^-1)
                             E:  172.0000 (GPa)
                           TSP:  132.6168 (kPa.m/W)
                            nu:  0.3100 (-)
                            CG:  765.0000 (kW/m^2)
                          mdot:  0.1000 (kg/s)

                          Liquid Sodium                         

                             T:  888.0000 (K)
                           rho:  807.6339 (kg/m^3)
                            Cp:  1252.6951 (m^2/s^2/K)
                            mu:  203.3446 (x1e6 kg/m/s)
                         kappa:  58.8628 (kg*m/s^3/K)
                            Pr:  0.0043 (-)
                             U:  0.1666 (m/s)
                          mdot:  0.1000 (kg/s)
                            Re:  20355.9410 (-)
                            Pe:  88.0904 (-)
                        deltaP: -9.4882 (Pa/m)
                           HCR:  125.2695 (J/K/s)
                         h_int:  12061.7567 (W/m^2/K)
                            Bi:  0.7440 (-)
                   dead-weight:  1.6937 (kg/m)
                           P_i:  0.0000 (bar (x1e-5 Pa))

                Determining peak flux for N06230                

                    Time taken:  30.7456 (sec)
</code>