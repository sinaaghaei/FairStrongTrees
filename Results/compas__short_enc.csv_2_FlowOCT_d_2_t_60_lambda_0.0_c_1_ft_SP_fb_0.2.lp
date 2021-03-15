\ Model FlowOCT
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  zeta[0,1,1] + zeta[0,2,1] + zeta[0,3,1] + zeta[0,4,1] + zeta[0,5,1]
   + zeta[0,6,1] + zeta[0,7,1] + zeta[1,1,1] + zeta[1,2,1] + zeta[1,3,1]
   + zeta[1,4,1] + zeta[1,5,1] + zeta[1,6,1] + zeta[1,7,1] + zeta[2,1,1]
   + zeta[2,2,1] + zeta[2,3,1] + zeta[2,4,1] + zeta[2,5,1] + zeta[2,6,1]
   + zeta[2,7,1] + zeta[3,1,1] + zeta[3,2,1] + zeta[3,3,1] + zeta[3,4,1]
   + zeta[3,5,1] + zeta[3,6,1] + zeta[3,7,1] + zeta[4,1,2] + zeta[4,2,2]
   + zeta[4,3,2] + zeta[4,4,2] + zeta[4,5,2] + zeta[4,6,2] + zeta[4,7,2]
   + zeta[5,1,1] + zeta[5,2,1] + zeta[5,3,1] + zeta[5,4,1] + zeta[5,5,1]
   + zeta[5,6,1] + zeta[5,7,1] + zeta[6,1,1] + zeta[6,2,1] + zeta[6,3,1]
   + zeta[6,4,1] + zeta[6,5,1] + zeta[6,6,1] + zeta[6,7,1] + zeta[7,1,1]
   + zeta[7,2,1] + zeta[7,3,1] + zeta[7,4,1] + zeta[7,5,1] + zeta[7,6,1]
   + zeta[7,7,1] + zeta[8,1,2] + zeta[8,2,2] + zeta[8,3,2] + zeta[8,4,2]
   + zeta[8,5,2] + zeta[8,6,2] + zeta[8,7,2] + zeta[9,1,1] + zeta[9,2,1]
   + zeta[9,3,1] + zeta[9,4,1] + zeta[9,5,1] + zeta[9,6,1] + zeta[9,7,1]
   - 70 Constant
Subject To
 R0: - zeta[0,1,1] - zeta[0,1,2] + z[0,1] - z[0,2] - z[0,3] = 0
 R1: - zeta[1,1,1] - zeta[1,1,2] + z[1,1] - z[1,2] - z[1,3] = 0
 R2: - zeta[2,1,1] - zeta[2,1,2] + z[2,1] - z[2,2] - z[2,3] = 0
 R3: - zeta[3,1,1] - zeta[3,1,2] + z[3,1] - z[3,2] - z[3,3] = 0
 R4: - zeta[4,1,1] - zeta[4,1,2] + z[4,1] - z[4,2] - z[4,3] = 0
 R5: - zeta[5,1,1] - zeta[5,1,2] + z[5,1] - z[5,2] - z[5,3] = 0
 R6: - zeta[6,1,1] - zeta[6,1,2] + z[6,1] - z[6,2] - z[6,3] = 0
 R7: - zeta[7,1,1] - zeta[7,1,2] + z[7,1] - z[7,2] - z[7,3] = 0
 R8: - zeta[8,1,1] - zeta[8,1,2] + z[8,1] - z[8,2] - z[8,3] = 0
 R9: - zeta[9,1,1] - zeta[9,1,2] + z[9,1] - z[9,2] - z[9,3] = 0
 R10: - zeta[0,2,1] - zeta[0,2,2] + z[0,2] - z[0,4] - z[0,5] = 0
 R11: - zeta[1,2,1] - zeta[1,2,2] + z[1,2] - z[1,4] - z[1,5] = 0
 R12: - zeta[2,2,1] - zeta[2,2,2] + z[2,2] - z[2,4] - z[2,5] = 0
 R13: - zeta[3,2,1] - zeta[3,2,2] + z[3,2] - z[3,4] - z[3,5] = 0
 R14: - zeta[4,2,1] - zeta[4,2,2] + z[4,2] - z[4,4] - z[4,5] = 0
 R15: - zeta[5,2,1] - zeta[5,2,2] + z[5,2] - z[5,4] - z[5,5] = 0
 R16: - zeta[6,2,1] - zeta[6,2,2] + z[6,2] - z[6,4] - z[6,5] = 0
 R17: - zeta[7,2,1] - zeta[7,2,2] + z[7,2] - z[7,4] - z[7,5] = 0
 R18: - zeta[8,2,1] - zeta[8,2,2] + z[8,2] - z[8,4] - z[8,5] = 0
 R19: - zeta[9,2,1] - zeta[9,2,2] + z[9,2] - z[9,4] - z[9,5] = 0
 R20: - zeta[0,3,1] - zeta[0,3,2] + z[0,3] - z[0,6] - z[0,7] = 0
 R21: - zeta[1,3,1] - zeta[1,3,2] + z[1,3] - z[1,6] - z[1,7] = 0
 R22: - zeta[2,3,1] - zeta[2,3,2] + z[2,3] - z[2,6] - z[2,7] = 0
 R23: - zeta[3,3,1] - zeta[3,3,2] + z[3,3] - z[3,6] - z[3,7] = 0
 R24: - zeta[4,3,1] - zeta[4,3,2] + z[4,3] - z[4,6] - z[4,7] = 0
 R25: - zeta[5,3,1] - zeta[5,3,2] + z[5,3] - z[5,6] - z[5,7] = 0
 R26: - zeta[6,3,1] - zeta[6,3,2] + z[6,3] - z[6,6] - z[6,7] = 0
 R27: - zeta[7,3,1] - zeta[7,3,2] + z[7,3] - z[7,6] - z[7,7] = 0
 R28: - zeta[8,3,1] - zeta[8,3,2] + z[8,3] - z[8,6] - z[8,7] = 0
 R29: - zeta[9,3,1] - zeta[9,3,2] + z[9,3] - z[9,6] - z[9,7] = 0
 R30: - b[1,age_factor.1] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] + z[0,2] <= 0
 R31: - b[2,age_factor.1] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] + z[0,4] <= 0
 R32: - b[3,age_factor.1] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] + z[0,6] <= 0
 R33: - b[1,two_year_recid.0] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] + z[1,2] <= 0
 R34: - b[2,two_year_recid.0] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.1] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] + z[1,4] <= 0
 R35: - b[3,two_year_recid.0] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.1] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] + z[1,6] <= 0
 R36: - b[1,two_year_recid.0] - b[1,age_factor.1] - b[1,age_factor.2]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] + z[2,2] <= 0
 R37: - b[2,two_year_recid.0] - b[2,age_factor.1] - b[2,age_factor.2]
   - b[2,race_factor.1] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] + z[2,4] <= 0
 R38: - b[3,two_year_recid.0] - b[3,age_factor.1] - b[3,age_factor.2]
   - b[3,race_factor.1] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] + z[2,6] <= 0
 R39: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,crime_factor.1] + z[3,2] <= 0
 R40: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,crime_factor.1] + z[3,4] <= 0
 R41: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,crime_factor.1] + z[3,6] <= 0
 R42: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,two_year_recid.0]
   - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.2]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] + z[4,2] <= 0
 R43: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,two_year_recid.0]
   - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.2]
   - b[2,race_factor.3] - b[2,race_factor.4] - b[2,race_factor.5]
   - b[2,race_factor.6] + z[4,4] <= 0
 R44: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,two_year_recid.0]
   - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.2]
   - b[3,race_factor.3] - b[3,race_factor.4] - b[3,race_factor.5]
   - b[3,race_factor.6] + z[4,6] <= 0
 R45: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] + z[5,2] <= 0
 R46: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] + z[5,4] <= 0
 R47: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] + z[5,6] <= 0
 R48: - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[6,2]
   <= 0
 R49: - b[2,gender_factor.1] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1] + z[6,4]
   <= 0
 R50: - b[3,gender_factor.1] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1] + z[6,6]
   <= 0
 R51: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.2]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] + z[7,2] <= 0
 R52: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.2]
   - b[2,race_factor.3] - b[2,race_factor.4] - b[2,race_factor.5]
   - b[2,race_factor.6] + z[7,4] <= 0
 R53: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.2]
   - b[3,race_factor.3] - b[3,race_factor.4] - b[3,race_factor.5]
   - b[3,race_factor.6] + z[7,6] <= 0
 R54: - b[1,two_year_recid.0] - b[1,age_factor.1] - b[1,age_factor.2]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[8,2]
   <= 0
 R55: - b[2,two_year_recid.0] - b[2,age_factor.1] - b[2,age_factor.2]
   - b[2,race_factor.1] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1] + z[8,4]
   <= 0
 R56: - b[3,two_year_recid.0] - b[3,age_factor.1] - b[3,age_factor.2]
   - b[3,race_factor.1] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1] + z[8,6]
   <= 0
 R57: - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[9,2]
   <= 0
 R58: - b[2,gender_factor.1] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1] + z[9,4]
   <= 0
 R59: - b[3,gender_factor.1] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1] + z[9,6]
   <= 0
 R60: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,race_factor.6]
   - b[1,crime_factor.1] + z[0,3] <= 0
 R61: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.2] - b[2,race_factor.6]
   - b[2,crime_factor.1] + z[0,5] <= 0
 R62: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.2] - b[3,race_factor.6]
   - b[3,crime_factor.1] + z[0,7] <= 0
 R63: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.1] - b[1,race_factor.2] - b[1,crime_factor.1] + z[1,3]
   <= 0
 R64: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.1] - b[2,race_factor.2] - b[2,crime_factor.1] + z[1,5]
   <= 0
 R65: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.1] - b[3,race_factor.2] - b[3,crime_factor.1] + z[1,7]
   <= 0
 R66: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.3] - b[1,race_factor.2] - b[1,crime_factor.1] + z[2,3]
   <= 0
 R67: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.3] - b[2,race_factor.2] - b[2,crime_factor.1] + z[2,5]
   <= 0
 R68: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.3] - b[3,race_factor.2] - b[3,crime_factor.1] + z[2,7]
   <= 0
 R69: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.6] + z[3,3]
   <= 0
 R70: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.6] + z[3,5]
   <= 0
 R71: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.6] + z[3,7]
   <= 0
 R72: - b[1,priors_count.3] - b[1,priors_count.4] - b[1,priors_count.5]
   - b[1,priors_count.6] - b[1,priors_count.7] - b[1,priors_count.8]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.1]
   - b[1,crime_factor.1] + z[4,3] <= 0
 R73: - b[2,priors_count.3] - b[2,priors_count.4] - b[2,priors_count.5]
   - b[2,priors_count.6] - b[2,priors_count.7] - b[2,priors_count.8]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.1]
   - b[2,crime_factor.1] + z[4,5] <= 0
 R74: - b[3,priors_count.3] - b[3,priors_count.4] - b[3,priors_count.5]
   - b[3,priors_count.6] - b[3,priors_count.7] - b[3,priors_count.8]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.1]
   - b[3,crime_factor.1] + z[4,7] <= 0
 R75: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.6]
   - b[1,crime_factor.1] + z[5,3] <= 0
 R76: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.6]
   - b[2,crime_factor.1] + z[5,5] <= 0
 R77: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.6]
   - b[3,crime_factor.1] + z[5,7] <= 0
 R78: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,age_factor.1] - b[1,race_factor.1] + z[6,3] <= 0
 R79: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,age_factor.1] - b[2,race_factor.1] + z[6,5] <= 0
 R80: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,age_factor.1] - b[3,race_factor.1] + z[6,7] <= 0
 R81: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.1]
   - b[1,crime_factor.1] + z[7,3] <= 0
 R82: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.1]
   - b[2,crime_factor.1] + z[7,5] <= 0
 R83: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.1]
   - b[3,crime_factor.1] + z[7,7] <= 0
 R84: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.3] - b[1,race_factor.2] + z[8,3] <= 0
 R85: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.3] - b[2,race_factor.2] + z[8,5] <= 0
 R86: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.3] - b[3,race_factor.2] + z[8,7] <= 0
 R87: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,age_factor.1] - b[1,race_factor.1] + z[9,3] <= 0
 R88: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,age_factor.1] - b[2,race_factor.1] + z[9,5] <= 0
 R89: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,age_factor.1] - b[3,race_factor.1] + z[9,7] <= 0
 R90: b[1,priors_count.1] + b[1,priors_count.2] + b[1,priors_count.3]
   + b[1,priors_count.4] + b[1,priors_count.5] + b[1,priors_count.6]
   + b[1,priors_count.7] + b[1,priors_count.8] + b[1,two_year_recid.0]
   + b[1,gender_factor.1] + b[1,age_factor.1] + b[1,age_factor.2]
   + b[1,age_factor.3] + b[1,race_factor.1] + b[1,race_factor.2]
   + b[1,race_factor.3] + b[1,race_factor.4] + b[1,race_factor.5]
   + b[1,race_factor.6] + b[1,crime_factor.1] + p[1] = 1
 R91: b[2,priors_count.1] + b[2,priors_count.2] + b[2,priors_count.3]
   + b[2,priors_count.4] + b[2,priors_count.5] + b[2,priors_count.6]
   + b[2,priors_count.7] + b[2,priors_count.8] + b[2,two_year_recid.0]
   + b[2,gender_factor.1] + b[2,age_factor.1] + b[2,age_factor.2]
   + b[2,age_factor.3] + b[2,race_factor.1] + b[2,race_factor.2]
   + b[2,race_factor.3] + b[2,race_factor.4] + b[2,race_factor.5]
   + b[2,race_factor.6] + b[2,crime_factor.1] + p[1] + p[2] = 1
 R92: b[3,priors_count.1] + b[3,priors_count.2] + b[3,priors_count.3]
   + b[3,priors_count.4] + b[3,priors_count.5] + b[3,priors_count.6]
   + b[3,priors_count.7] + b[3,priors_count.8] + b[3,two_year_recid.0]
   + b[3,gender_factor.1] + b[3,age_factor.1] + b[3,age_factor.2]
   + b[3,age_factor.3] + b[3,race_factor.1] + b[3,race_factor.2]
   + b[3,race_factor.3] + b[3,race_factor.4] + b[3,race_factor.5]
   + b[3,race_factor.6] + b[3,crime_factor.1] + p[1] + p[3] = 1
 R93: p[1] + p[2] + p[4] = 1
 R94: p[1] + p[2] + p[5] = 1
 R95: p[1] + p[3] + p[6] = 1
 R96: p[1] + p[3] + p[7] = 1
 R97: - beta[1,1] + zeta[0,1,1] <= 0
 R98: - beta[1,2] + zeta[0,1,2] <= 0
 R99: - beta[1,1] + zeta[1,1,1] <= 0
 R100: - beta[1,2] + zeta[1,1,2] <= 0
 R101: - beta[1,1] + zeta[2,1,1] <= 0
 R102: - beta[1,2] + zeta[2,1,2] <= 0
 R103: - beta[1,1] + zeta[3,1,1] <= 0
 R104: - beta[1,2] + zeta[3,1,2] <= 0
 R105: - beta[1,1] + zeta[4,1,1] <= 0
 R106: - beta[1,2] + zeta[4,1,2] <= 0
 R107: - beta[1,1] + zeta[5,1,1] <= 0
 R108: - beta[1,2] + zeta[5,1,2] <= 0
 R109: - beta[1,1] + zeta[6,1,1] <= 0
 R110: - beta[1,2] + zeta[6,1,2] <= 0
 R111: - beta[1,1] + zeta[7,1,1] <= 0
 R112: - beta[1,2] + zeta[7,1,2] <= 0
 R113: - beta[1,1] + zeta[8,1,1] <= 0
 R114: - beta[1,2] + zeta[8,1,2] <= 0
 R115: - beta[1,1] + zeta[9,1,1] <= 0
 R116: - beta[1,2] + zeta[9,1,2] <= 0
 R117: - beta[2,1] + zeta[0,2,1] <= 0
 R118: - beta[2,2] + zeta[0,2,2] <= 0
 R119: - beta[2,1] + zeta[1,2,1] <= 0
 R120: - beta[2,2] + zeta[1,2,2] <= 0
 R121: - beta[2,1] + zeta[2,2,1] <= 0
 R122: - beta[2,2] + zeta[2,2,2] <= 0
 R123: - beta[2,1] + zeta[3,2,1] <= 0
 R124: - beta[2,2] + zeta[3,2,2] <= 0
 R125: - beta[2,1] + zeta[4,2,1] <= 0
 R126: - beta[2,2] + zeta[4,2,2] <= 0
 R127: - beta[2,1] + zeta[5,2,1] <= 0
 R128: - beta[2,2] + zeta[5,2,2] <= 0
 R129: - beta[2,1] + zeta[6,2,1] <= 0
 R130: - beta[2,2] + zeta[6,2,2] <= 0
 R131: - beta[2,1] + zeta[7,2,1] <= 0
 R132: - beta[2,2] + zeta[7,2,2] <= 0
 R133: - beta[2,1] + zeta[8,2,1] <= 0
 R134: - beta[2,2] + zeta[8,2,2] <= 0
 R135: - beta[2,1] + zeta[9,2,1] <= 0
 R136: - beta[2,2] + zeta[9,2,2] <= 0
 R137: - beta[3,1] + zeta[0,3,1] <= 0
 R138: - beta[3,2] + zeta[0,3,2] <= 0
 R139: - beta[3,1] + zeta[1,3,1] <= 0
 R140: - beta[3,2] + zeta[1,3,2] <= 0
 R141: - beta[3,1] + zeta[2,3,1] <= 0
 R142: - beta[3,2] + zeta[2,3,2] <= 0
 R143: - beta[3,1] + zeta[3,3,1] <= 0
 R144: - beta[3,2] + zeta[3,3,2] <= 0
 R145: - beta[3,1] + zeta[4,3,1] <= 0
 R146: - beta[3,2] + zeta[4,3,2] <= 0
 R147: - beta[3,1] + zeta[5,3,1] <= 0
 R148: - beta[3,2] + zeta[5,3,2] <= 0
 R149: - beta[3,1] + zeta[6,3,1] <= 0
 R150: - beta[3,2] + zeta[6,3,2] <= 0
 R151: - beta[3,1] + zeta[7,3,1] <= 0
 R152: - beta[3,2] + zeta[7,3,2] <= 0
 R153: - beta[3,1] + zeta[8,3,1] <= 0
 R154: - beta[3,2] + zeta[8,3,2] <= 0
 R155: - beta[3,1] + zeta[9,3,1] <= 0
 R156: - beta[3,2] + zeta[9,3,2] <= 0
 R157: - beta[4,1] + zeta[0,4,1] <= 0
 R158: - beta[4,2] + zeta[0,4,2] <= 0
 R159: - beta[4,1] + zeta[1,4,1] <= 0
 R160: - beta[4,2] + zeta[1,4,2] <= 0
 R161: - beta[4,1] + zeta[2,4,1] <= 0
 R162: - beta[4,2] + zeta[2,4,2] <= 0
 R163: - beta[4,1] + zeta[3,4,1] <= 0
 R164: - beta[4,2] + zeta[3,4,2] <= 0
 R165: - beta[4,1] + zeta[4,4,1] <= 0
 R166: - beta[4,2] + zeta[4,4,2] <= 0
 R167: - beta[4,1] + zeta[5,4,1] <= 0
 R168: - beta[4,2] + zeta[5,4,2] <= 0
 R169: - beta[4,1] + zeta[6,4,1] <= 0
 R170: - beta[4,2] + zeta[6,4,2] <= 0
 R171: - beta[4,1] + zeta[7,4,1] <= 0
 R172: - beta[4,2] + zeta[7,4,2] <= 0
 R173: - beta[4,1] + zeta[8,4,1] <= 0
 R174: - beta[4,2] + zeta[8,4,2] <= 0
 R175: - beta[4,1] + zeta[9,4,1] <= 0
 R176: - beta[4,2] + zeta[9,4,2] <= 0
 R177: - beta[5,1] + zeta[0,5,1] <= 0
 R178: - beta[5,2] + zeta[0,5,2] <= 0
 R179: - beta[5,1] + zeta[1,5,1] <= 0
 R180: - beta[5,2] + zeta[1,5,2] <= 0
 R181: - beta[5,1] + zeta[2,5,1] <= 0
 R182: - beta[5,2] + zeta[2,5,2] <= 0
 R183: - beta[5,1] + zeta[3,5,1] <= 0
 R184: - beta[5,2] + zeta[3,5,2] <= 0
 R185: - beta[5,1] + zeta[4,5,1] <= 0
 R186: - beta[5,2] + zeta[4,5,2] <= 0
 R187: - beta[5,1] + zeta[5,5,1] <= 0
 R188: - beta[5,2] + zeta[5,5,2] <= 0
 R189: - beta[5,1] + zeta[6,5,1] <= 0
 R190: - beta[5,2] + zeta[6,5,2] <= 0
 R191: - beta[5,1] + zeta[7,5,1] <= 0
 R192: - beta[5,2] + zeta[7,5,2] <= 0
 R193: - beta[5,1] + zeta[8,5,1] <= 0
 R194: - beta[5,2] + zeta[8,5,2] <= 0
 R195: - beta[5,1] + zeta[9,5,1] <= 0
 R196: - beta[5,2] + zeta[9,5,2] <= 0
 R197: - beta[6,1] + zeta[0,6,1] <= 0
 R198: - beta[6,2] + zeta[0,6,2] <= 0
 R199: - beta[6,1] + zeta[1,6,1] <= 0
 R200: - beta[6,2] + zeta[1,6,2] <= 0
 R201: - beta[6,1] + zeta[2,6,1] <= 0
 R202: - beta[6,2] + zeta[2,6,2] <= 0
 R203: - beta[6,1] + zeta[3,6,1] <= 0
 R204: - beta[6,2] + zeta[3,6,2] <= 0
 R205: - beta[6,1] + zeta[4,6,1] <= 0
 R206: - beta[6,2] + zeta[4,6,2] <= 0
 R207: - beta[6,1] + zeta[5,6,1] <= 0
 R208: - beta[6,2] + zeta[5,6,2] <= 0
 R209: - beta[6,1] + zeta[6,6,1] <= 0
 R210: - beta[6,2] + zeta[6,6,2] <= 0
 R211: - beta[6,1] + zeta[7,6,1] <= 0
 R212: - beta[6,2] + zeta[7,6,2] <= 0
 R213: - beta[6,1] + zeta[8,6,1] <= 0
 R214: - beta[6,2] + zeta[8,6,2] <= 0
 R215: - beta[6,1] + zeta[9,6,1] <= 0
 R216: - beta[6,2] + zeta[9,6,2] <= 0
 R217: - beta[7,1] + zeta[0,7,1] <= 0
 R218: - beta[7,2] + zeta[0,7,2] <= 0
 R219: - beta[7,1] + zeta[1,7,1] <= 0
 R220: - beta[7,2] + zeta[1,7,2] <= 0
 R221: - beta[7,1] + zeta[2,7,1] <= 0
 R222: - beta[7,2] + zeta[2,7,2] <= 0
 R223: - beta[7,1] + zeta[3,7,1] <= 0
 R224: - beta[7,2] + zeta[3,7,2] <= 0
 R225: - beta[7,1] + zeta[4,7,1] <= 0
 R226: - beta[7,2] + zeta[4,7,2] <= 0
 R227: - beta[7,1] + zeta[5,7,1] <= 0
 R228: - beta[7,2] + zeta[5,7,2] <= 0
 R229: - beta[7,1] + zeta[6,7,1] <= 0
 R230: - beta[7,2] + zeta[6,7,2] <= 0
 R231: - beta[7,1] + zeta[7,7,1] <= 0
 R232: - beta[7,2] + zeta[7,7,2] <= 0
 R233: - beta[7,1] + zeta[8,7,1] <= 0
 R234: - beta[7,2] + zeta[8,7,2] <= 0
 R235: - beta[7,1] + zeta[9,7,1] <= 0
 R236: - beta[7,2] + zeta[9,7,2] <= 0
 R237: - p[1] + beta[1,1] + beta[1,2] = 0
 R238: - p[2] + beta[2,1] + beta[2,2] = 0
 R239: - p[3] + beta[3,1] + beta[3,2] = 0
 R240: - p[4] + beta[4,1] + beta[4,2] = 0
 R241: - p[5] + beta[5,1] + beta[5,2] = 0
 R242: - p[6] + beta[6,1] + beta[6,2] = 0
 R243: - p[7] + beta[7,1] + beta[7,2] = 0
 R244: zeta[0,4,1] + zeta[0,4,2] - z[0,4] = 0
 R245: zeta[1,4,1] + zeta[1,4,2] - z[1,4] = 0
 R246: zeta[2,4,1] + zeta[2,4,2] - z[2,4] = 0
 R247: zeta[3,4,1] + zeta[3,4,2] - z[3,4] = 0
 R248: zeta[4,4,1] + zeta[4,4,2] - z[4,4] = 0
 R249: zeta[5,4,1] + zeta[5,4,2] - z[5,4] = 0
 R250: zeta[6,4,1] + zeta[6,4,2] - z[6,4] = 0
 R251: zeta[7,4,1] + zeta[7,4,2] - z[7,4] = 0
 R252: zeta[8,4,1] + zeta[8,4,2] - z[8,4] = 0
 R253: zeta[9,4,1] + zeta[9,4,2] - z[9,4] = 0
 R254: zeta[0,5,1] + zeta[0,5,2] - z[0,5] = 0
 R255: zeta[1,5,1] + zeta[1,5,2] - z[1,5] = 0
 R256: zeta[2,5,1] + zeta[2,5,2] - z[2,5] = 0
 R257: zeta[3,5,1] + zeta[3,5,2] - z[3,5] = 0
 R258: zeta[4,5,1] + zeta[4,5,2] - z[4,5] = 0
 R259: zeta[5,5,1] + zeta[5,5,2] - z[5,5] = 0
 R260: zeta[6,5,1] + zeta[6,5,2] - z[6,5] = 0
 R261: zeta[7,5,1] + zeta[7,5,2] - z[7,5] = 0
 R262: zeta[8,5,1] + zeta[8,5,2] - z[8,5] = 0
 R263: zeta[9,5,1] + zeta[9,5,2] - z[9,5] = 0
 R264: zeta[0,6,1] + zeta[0,6,2] - z[0,6] = 0
 R265: zeta[1,6,1] + zeta[1,6,2] - z[1,6] = 0
 R266: zeta[2,6,1] + zeta[2,6,2] - z[2,6] = 0
 R267: zeta[3,6,1] + zeta[3,6,2] - z[3,6] = 0
 R268: zeta[4,6,1] + zeta[4,6,2] - z[4,6] = 0
 R269: zeta[5,6,1] + zeta[5,6,2] - z[5,6] = 0
 R270: zeta[6,6,1] + zeta[6,6,2] - z[6,6] = 0
 R271: zeta[7,6,1] + zeta[7,6,2] - z[7,6] = 0
 R272: zeta[8,6,1] + zeta[8,6,2] - z[8,6] = 0
 R273: zeta[9,6,1] + zeta[9,6,2] - z[9,6] = 0
 R274: zeta[0,7,1] + zeta[0,7,2] - z[0,7] = 0
 R275: zeta[1,7,1] + zeta[1,7,2] - z[1,7] = 0
 R276: zeta[2,7,1] + zeta[2,7,2] - z[2,7] = 0
 R277: zeta[3,7,1] + zeta[3,7,2] - z[3,7] = 0
 R278: zeta[4,7,1] + zeta[4,7,2] - z[4,7] = 0
 R279: zeta[5,7,1] + zeta[5,7,2] - z[5,7] = 0
 R280: zeta[6,7,1] + zeta[6,7,2] - z[6,7] = 0
 R281: zeta[7,7,1] + zeta[7,7,2] - z[7,7] = 0
 R282: zeta[8,7,1] + zeta[8,7,2] - z[8,7] = 0
 R283: zeta[9,7,1] + zeta[9,7,2] - z[9,7] = 0
 R284: z[0,1] = 1
 R285: z[1,1] = 1
 R286: z[2,1] = 1
 R287: z[3,1] = 1
 R288: z[4,1] = 1
 R289: z[5,1] = 1
 R290: z[6,1] = 1
 R291: z[7,1] = 1
 R292: z[8,1] = 1
 R293: z[9,1] = 1
 R294: 0.3333333333333333 zeta[0,1,2] + 0.3333333333333333 zeta[0,2,2]
   + 0.3333333333333333 zeta[0,3,2] + 0.3333333333333333 zeta[0,4,2]
   + 0.3333333333333333 zeta[0,5,2] + 0.3333333333333333 zeta[0,6,2]
   + 0.3333333333333333 zeta[0,7,2] + 0.3333333333333333 zeta[3,1,2]
   + 0.3333333333333333 zeta[3,2,2] + 0.3333333333333333 zeta[3,3,2]
   + 0.3333333333333333 zeta[3,4,2] + 0.3333333333333333 zeta[3,5,2]
   + 0.3333333333333333 zeta[3,6,2] + 0.3333333333333333 zeta[3,7,2]
   - 0.25 zeta[4,1,2] - 0.25 zeta[4,2,2] - 0.25 zeta[4,3,2]
   - 0.25 zeta[4,4,2] - 0.25 zeta[4,5,2] - 0.25 zeta[4,6,2]
   - 0.25 zeta[4,7,2] + 0.3333333333333333 zeta[5,1,2]
   + 0.3333333333333333 zeta[5,2,2] + 0.3333333333333333 zeta[5,3,2]
   + 0.3333333333333333 zeta[5,4,2] + 0.3333333333333333 zeta[5,5,2]
   + 0.3333333333333333 zeta[5,6,2] + 0.3333333333333333 zeta[5,7,2]
   - 0.25 zeta[6,1,2] - 0.25 zeta[6,2,2] - 0.25 zeta[6,3,2]
   - 0.25 zeta[6,4,2] - 0.25 zeta[6,5,2] - 0.25 zeta[6,6,2]
   - 0.25 zeta[6,7,2] - 0.25 zeta[7,1,2] - 0.25 zeta[7,2,2]
   - 0.25 zeta[7,3,2] - 0.25 zeta[7,4,2] - 0.25 zeta[7,5,2]
   - 0.25 zeta[7,6,2] - 0.25 zeta[7,7,2] - 0.25 zeta[9,1,2]
   - 0.25 zeta[9,2,2] - 0.25 zeta[9,3,2] - 0.25 zeta[9,4,2]
   - 0.25 zeta[9,5,2] - 0.25 zeta[9,6,2] - 0.25 zeta[9,7,2] <= 0.2
 R295: 0.3333333333333333 zeta[0,1,2] + 0.3333333333333333 zeta[0,2,2]
   + 0.3333333333333333 zeta[0,3,2] + 0.3333333333333333 zeta[0,4,2]
   + 0.3333333333333333 zeta[0,5,2] + 0.3333333333333333 zeta[0,6,2]
   + 0.3333333333333333 zeta[0,7,2] + 0.3333333333333333 zeta[3,1,2]
   + 0.3333333333333333 zeta[3,2,2] + 0.3333333333333333 zeta[3,3,2]
   + 0.3333333333333333 zeta[3,4,2] + 0.3333333333333333 zeta[3,5,2]
   + 0.3333333333333333 zeta[3,6,2] + 0.3333333333333333 zeta[3,7,2]
   - 0.25 zeta[4,1,2] - 0.25 zeta[4,2,2] - 0.25 zeta[4,3,2]
   - 0.25 zeta[4,4,2] - 0.25 zeta[4,5,2] - 0.25 zeta[4,6,2]
   - 0.25 zeta[4,7,2] + 0.3333333333333333 zeta[5,1,2]
   + 0.3333333333333333 zeta[5,2,2] + 0.3333333333333333 zeta[5,3,2]
   + 0.3333333333333333 zeta[5,4,2] + 0.3333333333333333 zeta[5,5,2]
   + 0.3333333333333333 zeta[5,6,2] + 0.3333333333333333 zeta[5,7,2]
   - 0.25 zeta[6,1,2] - 0.25 zeta[6,2,2] - 0.25 zeta[6,3,2]
   - 0.25 zeta[6,4,2] - 0.25 zeta[6,5,2] - 0.25 zeta[6,6,2]
   - 0.25 zeta[6,7,2] - 0.25 zeta[7,1,2] - 0.25 zeta[7,2,2]
   - 0.25 zeta[7,3,2] - 0.25 zeta[7,4,2] - 0.25 zeta[7,5,2]
   - 0.25 zeta[7,6,2] - 0.25 zeta[7,7,2] - 0.25 zeta[9,1,2]
   - 0.25 zeta[9,2,2] - 0.25 zeta[9,3,2] - 0.25 zeta[9,4,2]
   - 0.25 zeta[9,5,2] - 0.25 zeta[9,6,2] - 0.25 zeta[9,7,2] >= -0.2
Bounds
 Constant = 1
Binaries
 b[1,priors_count.1] b[1,priors_count.2] b[1,priors_count.3]
 b[1,priors_count.4] b[1,priors_count.5] b[1,priors_count.6]
 b[1,priors_count.7] b[1,priors_count.8] b[1,two_year_recid.0]
 b[1,gender_factor.1] b[1,age_factor.1] b[1,age_factor.2] b[1,age_factor.3]
 b[1,race_factor.1] b[1,race_factor.2] b[1,race_factor.3]
 b[1,race_factor.4] b[1,race_factor.5] b[1,race_factor.6]
 b[1,crime_factor.1] b[2,priors_count.1] b[2,priors_count.2]
 b[2,priors_count.3] b[2,priors_count.4] b[2,priors_count.5]
 b[2,priors_count.6] b[2,priors_count.7] b[2,priors_count.8]
 b[2,two_year_recid.0] b[2,gender_factor.1] b[2,age_factor.1]
 b[2,age_factor.2] b[2,age_factor.3] b[2,race_factor.1] b[2,race_factor.2]
 b[2,race_factor.3] b[2,race_factor.4] b[2,race_factor.5]
 b[2,race_factor.6] b[2,crime_factor.1] b[3,priors_count.1]
 b[3,priors_count.2] b[3,priors_count.3] b[3,priors_count.4]
 b[3,priors_count.5] b[3,priors_count.6] b[3,priors_count.7]
 b[3,priors_count.8] b[3,two_year_recid.0] b[3,gender_factor.1]
 b[3,age_factor.1] b[3,age_factor.2] b[3,age_factor.3] b[3,race_factor.1]
 b[3,race_factor.2] b[3,race_factor.3] b[3,race_factor.4]
 b[3,race_factor.5] b[3,race_factor.6] b[3,crime_factor.1] p[1] p[2] p[3]
 p[4] p[5] p[6] p[7] zeta[0,1,1] zeta[0,1,2] zeta[0,2,1] zeta[0,2,2]
 zeta[0,3,1] zeta[0,3,2] zeta[0,4,1] zeta[0,4,2] zeta[0,5,1] zeta[0,5,2]
 zeta[0,6,1] zeta[0,6,2] zeta[0,7,1] zeta[0,7,2] zeta[1,1,1] zeta[1,1,2]
 zeta[1,2,1] zeta[1,2,2] zeta[1,3,1] zeta[1,3,2] zeta[1,4,1] zeta[1,4,2]
 zeta[1,5,1] zeta[1,5,2] zeta[1,6,1] zeta[1,6,2] zeta[1,7,1] zeta[1,7,2]
 zeta[2,1,1] zeta[2,1,2] zeta[2,2,1] zeta[2,2,2] zeta[2,3,1] zeta[2,3,2]
 zeta[2,4,1] zeta[2,4,2] zeta[2,5,1] zeta[2,5,2] zeta[2,6,1] zeta[2,6,2]
 zeta[2,7,1] zeta[2,7,2] zeta[3,1,1] zeta[3,1,2] zeta[3,2,1] zeta[3,2,2]
 zeta[3,3,1] zeta[3,3,2] zeta[3,4,1] zeta[3,4,2] zeta[3,5,1] zeta[3,5,2]
 zeta[3,6,1] zeta[3,6,2] zeta[3,7,1] zeta[3,7,2] zeta[4,1,1] zeta[4,1,2]
 zeta[4,2,1] zeta[4,2,2] zeta[4,3,1] zeta[4,3,2] zeta[4,4,1] zeta[4,4,2]
 zeta[4,5,1] zeta[4,5,2] zeta[4,6,1] zeta[4,6,2] zeta[4,7,1] zeta[4,7,2]
 zeta[5,1,1] zeta[5,1,2] zeta[5,2,1] zeta[5,2,2] zeta[5,3,1] zeta[5,3,2]
 zeta[5,4,1] zeta[5,4,2] zeta[5,5,1] zeta[5,5,2] zeta[5,6,1] zeta[5,6,2]
 zeta[5,7,1] zeta[5,7,2] zeta[6,1,1] zeta[6,1,2] zeta[6,2,1] zeta[6,2,2]
 zeta[6,3,1] zeta[6,3,2] zeta[6,4,1] zeta[6,4,2] zeta[6,5,1] zeta[6,5,2]
 zeta[6,6,1] zeta[6,6,2] zeta[6,7,1] zeta[6,7,2] zeta[7,1,1] zeta[7,1,2]
 zeta[7,2,1] zeta[7,2,2] zeta[7,3,1] zeta[7,3,2] zeta[7,4,1] zeta[7,4,2]
 zeta[7,5,1] zeta[7,5,2] zeta[7,6,1] zeta[7,6,2] zeta[7,7,1] zeta[7,7,2]
 zeta[8,1,1] zeta[8,1,2] zeta[8,2,1] zeta[8,2,2] zeta[8,3,1] zeta[8,3,2]
 zeta[8,4,1] zeta[8,4,2] zeta[8,5,1] zeta[8,5,2] zeta[8,6,1] zeta[8,6,2]
 zeta[8,7,1] zeta[8,7,2] zeta[9,1,1] zeta[9,1,2] zeta[9,2,1] zeta[9,2,2]
 zeta[9,3,1] zeta[9,3,2] zeta[9,4,1] zeta[9,4,2] zeta[9,5,1] zeta[9,5,2]
 zeta[9,6,1] zeta[9,6,2] zeta[9,7,1] zeta[9,7,2] z[0,1] z[0,2] z[0,3]
 z[0,4] z[0,5] z[0,6] z[0,7] z[1,1] z[1,2] z[1,3] z[1,4] z[1,5] z[1,6]
 z[1,7] z[2,1] z[2,2] z[2,3] z[2,4] z[2,5] z[2,6] z[2,7] z[3,1] z[3,2]
 z[3,3] z[3,4] z[3,5] z[3,6] z[3,7] z[4,1] z[4,2] z[4,3] z[4,4] z[4,5]
 z[4,6] z[4,7] z[5,1] z[5,2] z[5,3] z[5,4] z[5,5] z[5,6] z[5,7] z[6,1]
 z[6,2] z[6,3] z[6,4] z[6,5] z[6,6] z[6,7] z[7,1] z[7,2] z[7,3] z[7,4]
 z[7,5] z[7,6] z[7,7] z[8,1] z[8,2] z[8,3] z[8,4] z[8,5] z[8,6] z[8,7]
 z[9,1] z[9,2] z[9,3] z[9,4] z[9,5] z[9,6] z[9,7]
End
