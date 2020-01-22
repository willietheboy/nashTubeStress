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


__________________________Nitrate Salt__________________________

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

                    Biharmonic coefficients:

                        Tbar_i:  749.6892 (K)
                          B'_1:  45.1191 (K)
                          D'_1: -0.0000 (K)
                        Tbar_o:  769.7119 (K)
                         B''_1:  79.4518 (K)
                         D''_1:  0.0000 (K)

                  Stress at outside tube crown:

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta:  0.0000 (MPa)
                   sigma_theta: -101.0056 (MPa)
                       sigma_z: -389.5197 (MPa)
                      sigma_Eq:  350.1201 (MPa)
                          Time:  1.0306 (sec)

__________________________Liquid Sodium_________________________

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

                    Biharmonic coefficients:

                        Tbar_i:  729.1010 (K)
                          B'_1:  10.2208 (K)
                          D'_1: -0.0000 (K)
                        Tbar_o:  749.3712 (K)
                         B''_1:  45.1490 (K)
                         D''_1:  0.0000 (K)

                  Stress at outside tube crown:

                       sigma_r:  0.0000 (MPa)
                  sigma_rTheta:  0.0000 (MPa)
                   sigma_theta: -110.3052 (MPa)
                       sigma_z: -251.8567 (MPa)
                      sigma_Eq:  218.6731 (MPa)
                          Time:  0.4356 (sec)
</code>