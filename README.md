# Vapor-Liquid-Equilibrium
Simple algorithms for fugacity and activity coefficients calculation
Must work using numpy matrices, algorithm for liquid fugacities include Grayson Streed (phigs), Peng-Robinson (philr), Soave Redlich Kwong (phil) whose parameters are Temperature, Pressure, Critical Temperature, Critical Pressure and acentric factor
Algorithm for vapor phase fugacity (phisrk) is Soave Redlich Kwong for gas mixtures, the parameters being Temperature, Pressure Critical Temperature, Critical Pressure and acentric factor
Algorithms for activity coefficient include UNIFAC method (UNIQUAC), whose parameters are composition (x), Rk (subgroup's relative volume), Qk (subgroup's relative surface area), v (matrix containing the quantity of every subgroups in a molecule of given species), a (group interaction parameters, taken from Hansen, Rasmusen, Fredenslund, Schiller and Gmehling, IEC Research,vol. 30, pp. 2352-2355, 1991) and T (Temperature)
For activity coefficient it includes Chao-Seader with Flory-Huggins' correction Parameters being composition (x), temperature (T), Solubility Parameter(di) in sqrt(cal/cm3) and liquid molar volume (vil) in cm3/gmol
